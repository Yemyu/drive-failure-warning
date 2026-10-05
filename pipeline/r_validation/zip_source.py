"""Local ZIP reader for first observation of a bound archive.

No download or release bypass is exposed here. The production coordinator must
validate its release before opening any real archive. Receipts certify bytes
and CSV structure only; they are not a completed panel or scoring authorisation.
"""
from contextlib import contextmanager
import csv
import datetime as dt
import hashlib
import io
import json
from pathlib import Path
import zipfile

from pipeline.reproducible.source_streams import CheckedReader, CHUNK, SourceError, fingerprint, digest_value
from tools.run_small_replay import SOURCE_COLUMNS


class RArchiveSource:
    def __init__(self, path, *, expected_sha256, start, end, prefix,
                 max_member_bytes=256*1024**2, max_expanded_bytes=12*1024**3):
        self.path = Path(path).resolve()
        self.expected_sha = digest_value(expected_sha256, 'archive SHA256')
        self.start, self.end = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
        if self.start > self.end or self.start.isoformat() != start or self.end.isoformat() != end:
            raise SourceError('invalid archive calendar')
        if not prefix or '/' in prefix or '\\' in prefix or prefix in {'.', '..'}:
            raise SourceError('invalid archive prefix')
        for value in (max_member_bytes, max_expanded_bytes):
            if type(value) is not int or value <= 0:
                raise SourceError('invalid expansion budget')
        self.member_limit, self.total_limit = max_member_bytes, max_expanded_bytes
        self.names = [f'{prefix}/{(self.start+dt.timedelta(days=i)).isoformat()}.csv'
                      for i in range((self.end-self.start).days+1)]
        self.finder_metadata = f'{prefix}/.DS_Store'
        self.ignored_members = []
        self.archive = None
        self.facts = []
        self.columns = None
        self.identities = {}
        self.complete = False
        self.entered = False

    def __enter__(self):
        if self.entered:
            raise SourceError('archive reader cannot be reused')
        self.entered = True
        if fingerprint(self.path) != self.expected_sha:
            raise SourceError('archive SHA256 mismatch')
        self.archive = zipfile.ZipFile(self.path)
        try:
            infos = self.archive.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)):
                raise SourceError('duplicate ZIP member')
            data = [info for info in infos if not info.is_dir() and not info.filename.startswith('__MACOSX/')
                    and info.filename != self.finder_metadata]
            self.ignored_members = [{'name':info.filename,'expanded_bytes':info.file_size,
                                     'compressed_bytes':info.compress_size,'crc32':info.CRC,
                                     'reason':'directory' if info.is_dir() else 'macos_metadata'}
                                    for info in infos if info not in data]
            if {info.filename for info in data} != set(self.names):
                raise SourceError('archive calendar/member inventory mismatch')
            if sum(info.file_size for info in infos) > self.total_limit:
                raise SourceError('archive expansion budget exceeded')
            if any(info.file_size > self.member_limit for info in infos):
                raise SourceError('member expansion budget exceeded')
            for info in data:
                if info.file_size > self.member_limit:
                    raise SourceError('member expansion budget exceeded')
                if info.compress_type != zipfile.ZIP_DEFLATED or info.flag_bits & 1:
                    raise SourceError('unsupported or encrypted member')
            return self
        except BaseException:
            self.archive.close()
            raise

    def __exit__(self, kind, value, traceback):
        self.archive.close()
        if kind is None:
            if len(self.facts) != len(self.names):
                raise SourceError('not all archive members were verified')
            if fingerprint(self.path) != self.expected_sha:
                raise SourceError('archive changed while reading')
            self.complete = True

    @contextmanager
    def rows(self, name):
        if self.archive is None or self.complete or len(self.facts) >= len(self.names) or name != self.names[len(self.facts)]:
            raise SourceError('members must be consumed once in calendar order')
        info = self.archive.getinfo(name)
        date = Path(name).stem
        with self.archive.open(info) as raw:
            checked = CheckedReader(raw, info.file_size, info.CRC)
            with io.TextIOWrapper(io.BufferedReader(checked, buffer_size=CHUNK), encoding='utf-8-sig', newline='') as text:
                reader = csv.reader(text, strict=True)
                columns = next(reader, None)
                if columns is None or len(columns) != len(set(columns)) or not set(SOURCE_COLUMNS).issubset(columns):
                    raise SourceError('missing or duplicated CSV fields')
                if self.columns is not None and columns != self.columns:
                    raise SourceError('within-quarter CSV header changed')
                self.columns = columns
                count = selected = 0
                seen = set()
                exhausted = False

                def validated_rows():
                    nonlocal count, selected, exhausted
                    for values in reader:
                        if len(values) != len(columns):
                            raise SourceError('CSV row width mismatch')
                        row = dict(zip(columns, values))
                        if row['date'] != date or not row['serial_number'] or not row['model']:
                            raise SourceError('invalid member row date or identity')
                        serial, model = row['serial_number'], row['model']
                        if serial in seen:
                            raise SourceError('duplicate daily source serial')
                        seen.add(serial)
                        if serial in self.identities and self.identities[serial] != model:
                            raise SourceError('full-source serial/model conflict')
                        self.identities[serial] = model
                        count += 1
                        selected += int(model == 'ST4000DM000')
                        yield row
                    exhausted = True

                yield validated_rows()
                if not exhausted or not checked.verified:
                    raise SourceError('CSV consumer did not exhaust member')
                self.facts.append({'name': name, 'date': date, 'source_rows': count,
                                   'selected_rows': selected, 'expanded_bytes': checked.count,
                                   'compressed_bytes': info.compress_size, 'crc32': info.CRC,
                                   'sha256': checked.sha.hexdigest()})

    def receipt(self):
        if not self.complete:
            raise SourceError('archive receipt is unavailable before complete verification')
        return {'status': 'verified_source', 'scope': 'bytes_csv_structure_and_full_source_identity',
                'ignored_members':self.ignored_members,
                'archive_sha256': self.expected_sha, 'members': [dict(item) for item in self.facts],
                'schema_columns': list(self.columns),
                'schema_sha256': hashlib.sha256(json.dumps(self.columns, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest(),
                'full_source_serial_count': len(self.identities)}
