"""Bounded decompression for retained ZIP ranges and complete ZIP archives."""

from contextlib import contextmanager
import csv
import hashlib
import io
import json
from pathlib import Path
import struct
import zlib
import zipfile

CHUNK = 64 * 1024


class SourceError(ValueError):
    pass


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(CHUNK):
            digest.update(block)
    return digest.hexdigest()


def integer(value, name):
    if type(value) is not int or value < 0:
        raise SourceError(f"{name} must be a nonnegative integer")
    return value


def digest_value(value, name):
    if not isinstance(value, str) or len(value) != 64 or set(value) - set("0123456789abcdef"):
        raise SourceError(f"invalid {name}")
    return value


def bound_path(root, value):
    root = Path(root).resolve()
    if not isinstance(value, str) or not value:
        raise SourceError("missing source path")
    path = (root / value).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise SourceError(f"source path is missing or outside project: {value}")
    return path


class DeflateReader(io.RawIOBase):
    """Return at most 64 KiB expanded bytes per read, including high-ratio input."""

    def __init__(self, handle, compressed):
        self.handle = handle
        self.remaining = compressed
        self.decoder = zlib.decompressobj(-15)
        self.pending = b""

    def readable(self):
        return True

    def readinto(self, buffer):
        if not buffer:
            return 0
        while True:
            if self.decoder.eof:
                if self.remaining or self.pending or self.decoder.unused_data:
                    raise SourceError("compressed size includes trailing data")
                return 0
            if not self.pending and self.remaining:
                block = self.handle.read(min(CHUNK, self.remaining))
                if not block:
                    raise SourceError("truncated compressed member")
                self.remaining -= len(block)
                self.pending = block
            if not self.pending:
                raise SourceError("DEFLATE stream has no end marker")
            try:
                block = self.decoder.decompress(self.pending, min(CHUNK, len(buffer)))
            except zlib.error as exc:
                raise SourceError("invalid DEFLATE stream") from exc
            self.pending = self.decoder.unconsumed_tail
            if block:
                buffer[:len(block)] = block
                return len(block)


class CheckedReader(io.RawIOBase):
    def __init__(self, source, size, crc, sha=None):
        self.source, self.size, self.expected_crc, self.expected_sha = source, size, crc, sha
        self.count = self.crc = 0
        self.sha = hashlib.sha256()
        self.verified = False

    def readable(self):
        return True

    def readinto(self, buffer):
        block = self.source.read(min(CHUNK, len(buffer)))
        if block:
            self.count += len(block)
            if self.count > self.size:
                raise SourceError("expanded member exceeds declared size")
            self.crc = zlib.crc32(block, self.crc)
            self.sha.update(block)
            buffer[:len(block)] = block
            return len(block)
        if self.count != self.size or self.crc & 0xffffffff != self.expected_crc:
            raise SourceError("expanded size or CRC mismatch")
        if self.expected_sha is not None and self.sha.hexdigest() != self.expected_sha:
            raise SourceError("expanded member SHA256 mismatch")
        self.verified = True
        return 0


@contextmanager
def csv_rows(source, entry, source_sha, *, expanded_sha=None):
    checked = CheckedReader(source, integer(entry["size"], "size"), integer(entry["crc"], "crc"), expanded_sha)
    with io.TextIOWrapper(io.BufferedReader(checked, buffer_size=CHUNK), encoding="utf-8-sig", newline="") as text:
        reader = csv.reader(text, strict=True)
        try:
            columns = next(reader)
        except StopIteration as exc:
            raise SourceError("empty CSV member") from exc
        expected = entry["expected_counts"]
        schema_sha = hashlib.sha256(json.dumps(columns, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
        if len(columns) != len(set(columns)) or len(columns) != expected["schema_columns"] or schema_sha != expected["schema_sha256"]:
            raise SourceError("ordered CSV header/schema hash mismatch")
        facts = {"expanded_bytes": entry["size"], "crc32": entry["crc"], "chunk_bytes": CHUNK}
        yield entry["date"], source_sha, columns, enumerate(reader, 2), facts
        if not checked.verified:
            raise SourceError("CSV consumer did not exhaust and verify the member")


@contextmanager
def range_rows(entry, root):
    path = bound_path(root, entry["local"])
    sha = digest_value(entry["sha256"], "member SHA256")
    if path.stat().st_size != integer(entry["range_bytes"], "range bytes") or fingerprint(path) != sha:
        raise SourceError("retained member bytes or SHA256 mismatch")
    with path.open("rb") as handle:
        header = handle.read(30)
        if len(header) != 30:
            raise SourceError("truncated ZIP local header")
        magic, version, flags, method, _, _, crc, compressed, size, namesize, extrasize = struct.unpack("<4s5H3L2H", header)
        if magic != b"PK\x03\x04" or method != 8 or flags & ~0x808 or version > 20:
            raise SourceError("unsupported ZIP local header")
        if handle.read(namesize).decode("utf-8") != entry["name"]:
            raise SourceError("ZIP local filename mismatch")
        if len(handle.read(extrasize)) != extrasize:
            raise SourceError("truncated ZIP extra field")
        compressed_expected = integer(entry["compressed"], "compressed size")
        if handle.tell() + compressed_expected > path.stat().st_size:
            raise SourceError("compressed member extends beyond retained range")
        if not flags & 8 and (crc, compressed, size) != (entry["crc"], compressed_expected, entry["size"]):
            raise SourceError("ZIP local sizes/CRC disagree with contract")
        with io.BufferedReader(DeflateReader(handle, compressed_expected), buffer_size=CHUNK) as raw:
            with csv_rows(raw, entry, sha) as data:
                yield data
    if fingerprint(path) != sha:
        raise SourceError("retained member changed while reading")


class ArchiveSource:
    """One archive binding per build; each member is verified while streamed."""

    def __init__(self, receipt, root):
        self.receipt = receipt
        self.path = bound_path(root, receipt["archive_path"])
        self.sha = digest_value(receipt["archive_sha256"], "archive SHA256")
        self.archive = None

    def __enter__(self):
        if self.path.stat().st_size != integer(self.receipt["archive_bytes"], "archive bytes") or fingerprint(self.path) != self.sha:
            raise SourceError("archive bytes or SHA256 mismatch")
        self.archive = zipfile.ZipFile(self.path)
        try:
            infos = [info for info in self.archive.infolist() if info.filename.endswith(".csv") and not info.filename.startswith("__MACOSX/")]
            names = [info.filename for info in infos]
            expected = [entry["name"] for entry in self.receipt["members"]]
            if len(names) != len(set(names)) or len(expected) != len(set(expected)) or set(names) != set(expected):
                raise SourceError("archive member inventory mismatch")
            if len(expected) != self.receipt["member_count"]:
                raise SourceError("archive member count mismatch")
            return self
        except BaseException:
            self.archive.close()
            raise

    def __exit__(self, exc_type, exc, tb):
        self.archive.close()
        if exc_type is None and fingerprint(self.path) != self.sha:
            raise SourceError("archive changed while reading")

    @contextmanager
    def rows(self, member):
        info = self.archive.getinfo(member["name"])
        if (info.compress_size, info.file_size, info.CRC) != (member["compressed_bytes"], member["expanded_bytes"], member["crc"]) or info.compress_type != zipfile.ZIP_DEFLATED or info.flag_bits & 1:
            raise SourceError("ZIP directory disagrees with receipt")
        entry = dict(member, size=member["expanded_bytes"], expected_counts={key: member[key] for key in ("source_rows", "selected_rows", "failure_rows")})
        entry["expected_counts"].update(schema_columns=self.receipt["schema_columns"], schema_sha256=self.receipt["schema_sha256"])
        with self.archive.open(info) as raw:
            with csv_rows(raw, entry, member["sha256"], expanded_sha=digest_value(member["sha256"], "expanded SHA256")) as data:
                yield entry, data
