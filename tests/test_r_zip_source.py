import hashlib
import io
from pathlib import Path
import struct
import tempfile
import unittest
import warnings
import zipfile

from pipeline.r_validation.zip_source import RArchiveSource
from pipeline.reproducible.source_streams import SourceError
from tests.test_source_streams import csv_bytes, COLUMNS

ROOT = Path(__file__).resolve().parents[1]


class RZipSourceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=ROOT/'.tmp'); self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name)/'synthetic.zip'

    def archive(self, entries=None):
        if entries is None:
            entries=[('fixture/2023-10-01.csv',csv_bytes('2023-10-01',[{}])),
                     ('fixture/2023-10-02.csv',csv_bytes('2023-10-02',[{}]))]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',UserWarning)
            with zipfile.ZipFile(self.path,'w',compression=zipfile.ZIP_DEFLATED) as archive:
                for name,data in entries: archive.writestr(name,data)
        return hashlib.sha256(self.path.read_bytes()).hexdigest()

    def reader(self, sha=None, **kw):
        return RArchiveSource(self.path,expected_sha256=sha or hashlib.sha256(self.path.read_bytes()).hexdigest(),
                              start='2023-10-01',end='2023-10-02',prefix='fixture',**kw)

    def consume(self, source):
        with source:
            for name in source.names:
                with source.rows(name) as rows: list(rows)
        return source.receipt()

    def test_stream_receipts_and_no_early_completion(self):
        self.archive(); source=self.reader()
        with self.assertRaises(SourceError): source.receipt()
        receipt=self.consume(source)
        self.assertEqual([x['source_rows'] for x in receipt['members']],[1,1])
        self.assertEqual(receipt['full_source_serial_count'],1)
        self.assertEqual(receipt['members'][0]['sha256'],hashlib.sha256(csv_bytes('2023-10-01',[{}])).hexdigest())
        with self.assertRaisesRegex(SourceError,'cannot be reused'):
            with source: pass

    def test_missing_duplicate_and_unexpected_members_refused(self):
        first=('fixture/2023-10-01.csv',csv_bytes('2023-10-01',[{}]))
        for entries in ([first],[first,first],[first,('../2023-10-02.csv',b'x')]):
            self.archive(entries)
            with self.assertRaises(SourceError): self.consume(self.reader())

    def test_exact_finder_metadata_is_recorded_without_csv_decoding(self):
        entries=[('fixture/2023-10-01.csv',csv_bytes('2023-10-01',[{}])),
                 ('fixture/2023-10-02.csv',csv_bytes('2023-10-02',[{}])),
                 ('fixture/.DS_Store',b'\xff\x00not CSV')]
        self.archive(entries);receipt=self.consume(self.reader())
        self.assertEqual(len(receipt['members']),2)
        self.assertEqual(receipt['ignored_members'][0]['name'],'fixture/.DS_Store')
        self.assertEqual(receipt['ignored_members'][0]['expanded_bytes'],len(entries[-1][1]))

    def test_finder_exception_does_not_accept_other_extra_files_or_missing_date(self):
        first=('fixture/2023-10-01.csv',csv_bytes('2023-10-01',[{}]))
        second=('fixture/2023-10-02.csv',csv_bytes('2023-10-02',[{}]))
        for entries in ([first,('fixture/.DS_Store',b'x')],
                        [first,second,('fixture/.DS_Store.csv',b'x')],
                        [first,second,('other/.DS_Store',b'x')],
                        [first,second,('fixture/.DS_Store',b'x'),('fixture/.DS_Store',b'y')]):
            with self.subTest(entries=[x[0] for x in entries]):
                self.archive(entries)
                with self.assertRaises(SourceError):self.consume(self.reader())

    def test_ignored_metadata_still_counts_toward_expansion_budget(self):
        self.archive([('fixture/2023-10-01.csv',csv_bytes('2023-10-01',[{}])),
                      ('fixture/2023-10-02.csv',csv_bytes('2023-10-02',[{}])),('fixture/.DS_Store',b'x'*1000)])
        with self.assertRaisesRegex(SourceError,'member expansion'):
            self.consume(self.reader(max_member_bytes=999))

    def test_header_change_row_date_and_width_refused(self):
        first=('fixture/2023-10-01.csv',csv_bytes('2023-10-01',[{}]))
        for body in (csv_bytes('2023-10-02',[{}],COLUMNS+['extra']),csv_bytes('2023-10-01',[{}]),
                     csv_bytes('2023-10-02',[{}])+b'bad,row\n'):
            self.archive([first,('fixture/2023-10-02.csv',body)])
            with self.assertRaises(SourceError): self.consume(self.reader())

    def test_non_target_model_conflict_and_daily_duplicate_refused(self):
        for second in ([{'model':'OTHER'}],[{},{}]):
            self.archive([('fixture/2023-10-01.csv',csv_bytes('2023-10-01',[{}])),
                          ('fixture/2023-10-02.csv',csv_bytes('2023-10-02',second))])
            with self.assertRaises(SourceError): self.consume(self.reader())

    def test_expansion_limits_and_wrong_sha_refused(self):
        self.archive()
        for source in (self.reader(max_member_bytes=1),self.reader(max_expanded_bytes=1),self.reader('0'*64)):
            with self.assertRaises(SourceError): self.consume(source)

    def test_partial_consumption_refused(self):
        self.archive(); source=self.reader()
        with self.assertRaises(SourceError):
            with source:
                with source.rows(source.names[0]) as rows: next(rows)
        with self.assertRaises(SourceError): source.receipt()

    def test_corrupt_crc_refused_even_with_current_archive_hash(self):
        self.archive()
        raw=bytearray(self.path.read_bytes())
        position=raw.index(b'PK\x01\x02')
        crc=struct.unpack_from('<I',raw,position+16)[0]
        struct.pack_into('<I',raw,position+16,crc^1)
        self.path.write_bytes(raw)
        with self.assertRaises((SourceError,zipfile.BadZipFile)):
            self.consume(self.reader())


if __name__=='__main__': unittest.main()
