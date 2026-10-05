"""Build the selected-model SQLite panel from an existing member manifest."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

from .member_io import load_manifest, manifest_entries
from .panel import SCHEMA_VERSION, PanelWriter


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--model", default="ST4000DM000")
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    entries = manifest_entries(manifest)
    writer = PanelWriter(args.output, reset=args.reset)
    writer.set_metadata(
        {
            "schema_version": SCHEMA_VERSION,
            "selected_model": args.model,
            "source_url": manifest.get("source_url", ""),
            "retrieved_at_utc": manifest.get("retrieved_at_utc", ""),
            "archive_bytes": str(manifest.get("archive_bytes", "")),
            "manifest": str(args.manifest),
        }
    )
    try:
        for index, entry in enumerate(entries, start=1):
            result = writer.append_member(entry, pathlib.Path(__file__).resolve().parents[1], args.model)
            print(json.dumps(result, ensure_ascii=False), flush=True)
            if index % 10 == 0:
                print(f"progress={index}/{len(entries)} panel={writer.summary()['panel_rows']}", flush=True)
        summary = writer.summary()
    finally:
        writer.close()
    print(json.dumps({"summary": summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
