"""Public page rebuilding must need only repository files and preserve saved data."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import build_signal_dashboard as dashboard


class PublicDashboardTests(unittest.TestCase):
    def test_rebuild_is_deterministic_and_does_not_modify_data(self):
        before = dashboard.DATA.read_bytes()
        with tempfile.TemporaryDirectory(dir=dashboard.ROOT) as directory:
            target = Path(directory)
            with contextlib.redirect_stdout(io.StringIO()):
                first = dashboard.build(target)
                page = (target / "index.html").read_bytes()
                second = dashboard.build(target)
            self.assertEqual(first, second)
            self.assertEqual(page, (target / "index.html").read_bytes())
            self.assertEqual(first["outputs"]["index.html"], hashlib.sha256(page).hexdigest())
            self.assertEqual(list(first["sources"]), ["dashboard/signal_v1/data.json"])
        self.assertEqual(dashboard.DATA.read_bytes(), before)

    def test_changed_data_is_rejected_before_any_page_is_written(self):
        with tempfile.TemporaryDirectory(dir=dashboard.ROOT) as directory:
            root = Path(directory)
            data = root / "changed.json"
            data.write_text('{}\n', encoding="utf-8")
            source = root / "binding.json"
            source.write_text(json.dumps({"schema_version": 1, "data_path": data.relative_to(dashboard.ROOT).as_posix(),
                                          "data_sha256": "0" * 64}), encoding="utf-8")
            target = root / "build"
            with patch.object(dashboard, "DATA", data), patch.object(dashboard, "SOURCE", source):
                with self.assertRaisesRegex(ValueError, "does not match"):
                    dashboard.build(target)
            self.assertFalse(target.exists())

    def test_inline_json_cannot_close_its_script(self):
        with tempfile.TemporaryDirectory(dir=dashboard.ROOT) as directory:
            root = Path(directory)
            data = root / "data.json"
            value = {"cases": [], "note": "</script><script>unexpected()</script>"}
            data.write_text(json.dumps(value), encoding="utf-8")
            source = root / "source.json"
            source.write_text(json.dumps({"schema_version": 1, "data_path": data.relative_to(dashboard.ROOT).as_posix(),
                                          "data_sha256": dashboard.sha256(data)}), encoding="utf-8")
            template = root / "template.html"
            template.write_text('<script type="application/json">__DATA__</script>', encoding="utf-8")
            with patch.object(dashboard, "DATA", data), patch.object(dashboard, "SOURCE", source), patch.object(dashboard, "TEMPLATE", template):
                with contextlib.redirect_stdout(io.StringIO()):
                    dashboard.build(root / "build")
            page = (root / "build/index.html").read_text(encoding="utf-8")
            self.assertEqual(page.count('</script>'), 1)
            payload = page.split('>', 1)[1].rsplit('</script>', 1)[0]
            self.assertEqual(json.loads(payload), value)


if __name__ == "__main__":
    unittest.main()
