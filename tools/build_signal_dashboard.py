"""Build the static dashboard from the committed result snapshot, without scoring."""
from pathlib import Path
import argparse
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dashboard/signal_v1/data.json"
SOURCE = ROOT / "dashboard/signal_v1/source_manifest.json"
TEMPLATE = ROOT / "dashboard/template_signal.html"
OUT = DATA.parent


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(output=None):
    destination = Path(output).resolve() if output else OUT
    if not destination.is_relative_to(ROOT):
        raise ValueError("output directory must be inside the project")
    binding = json.loads(SOURCE.read_text(encoding="utf-8"))
    if binding.get("schema_version") != 1 or binding.get("data_path") != DATA.relative_to(ROOT).as_posix():
        raise ValueError("invalid public data binding")
    if sha256(DATA) != binding.get("data_sha256"):
        raise ValueError("saved dashboard data does not match its source manifest")
    payload = DATA.read_text(encoding="utf-8").strip()
    data = json.loads(payload)
    template = TEMPLATE.read_text(encoding="utf-8")
    if template.count("__DATA__") != 1:
        raise ValueError("template must contain exactly one data placeholder")
    # Preserve JSON values while preventing an embedded closing script tag.
    html = template.replace("__DATA__", payload.replace("<", "\\u003c"))
    destination.mkdir(parents=True, exist_ok=True)
    page = destination / "index.html"
    page.write_text(html, encoding="utf-8")
    manifest = {
        "schema_version": 2,
        "scope": "Static presentation build from committed Q3/Q4 results; no training, scoring or raw data access",
        "source_manifest_sha256": sha256(SOURCE),
        "sources": {DATA.relative_to(ROOT).as_posix(): sha256(DATA)},
        "template_sha256": sha256(TEMPLATE),
        "outputs": {"index.html": sha256(page)},
        "tabler": {"version": "1.0.0", "stylesheet_sha256": sha256(ROOT / "dashboard/vendor/tabler/tabler.min.css"),
                   "asset_manifest_sha256": sha256(ROOT / "dashboard/vendor/tabler/ASSETS.json")},
    }
    (destination / "export_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Built {page.relative_to(ROOT)}; {page.stat().st_size:,} bytes; {len(data['cases'])} anonymous cases")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional build directory inside the project")
    args = parser.parse_args()
    try:
        build(args.output)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        parser.exit(2, f"Build failed: {exc}\n")
