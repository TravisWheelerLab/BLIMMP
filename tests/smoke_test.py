#!/usr/bin/env python3
"""Smoke test: run BLIMMP end-to-end and assert the results are non-trivial.

This guards against mispackaged builds. BLIMMP builds its KO universe from data
files that ship inside the package; when those files are missing, every observed
annotation is dropped by a left join and the run still exits 0 while reporting
every module absent. The output looks structurally complete, so nothing short of
inspecting the values catches it.

Usage:
    python tests/smoke_test.py [--blimmp BLIMMP] [--input Examples/example.domtblout]
"""

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def fail(message):
    print(f"FAIL: {message}", file=sys.stderr)
    sys.exit(1)


def read_rows(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blimmp", default="BLIMMP", help="BLIMMP executable (default: BLIMMP)")
    parser.add_argument(
        "--input",
        default=str(REPO_ROOT / "Examples" / "example.domtblout"),
        help="domtblout file to run against",
    )
    args = parser.parse_args()

    if not Path(args.input).is_file():
        fail(f"test input not found: {args.input}")

    with tempfile.TemporaryDirectory() as tmpdir:
        prefix = str(Path(tmpdir) / "smoke")
        cmd = [args.blimmp, args.input, "-f", "domtblout", "--sigma", "1.0", "--output", prefix]
        print("Running:", " ".join(cmd), flush=True)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(result.stdout)
            print(result.stderr, file=sys.stderr)
            fail(f"BLIMMP exited {result.returncode}")

        # A mispackaged build warns instead of failing; treat that as fatal here.
        combined = result.stdout + result.stderr
        for marker in ("file not found", "Module descriptions file not found"):
            if marker in combined:
                print(combined)
                fail(f"BLIMMP reported a missing data file (matched {marker!r})")

        dk_path = Path(f"{prefix}_BLIMMP_dk.csv")
        modules_csv = Path(f"{prefix}_BLIMMP_module_probabilities.csv")
        modules_json = Path(f"{prefix}_BLIMMP_modules.json")
        for path in (dk_path, modules_csv, modules_json):
            if not path.is_file():
                fail(f"expected output missing: {path.name}")

        # 1. The per-KO table must not be empty. This is the direct symptom of an
        #    empty KO universe -- the bug that produced all-zero results in v0.1.2.
        dk_rows = read_rows(dk_path)
        if not dk_rows:
            fail("per-KO table (_BLIMMP_dk.csv) has no data rows -- KO universe was empty")
        print(f"  per-KO rows: {len(dk_rows)}")

        module_rows = read_rows(modules_csv)
        if not module_rows:
            fail("module table has no data rows")

        # 2. At least one module must carry non-zero confidence. The example input
        #    is a real genome's search results, so an all-zero result is a bug.
        def as_float(value):
            try:
                return float(value)
            except (TypeError, ValueError):
                return 0.0

        confident = [r for r in module_rows if as_float(r.get("module_confidence")) > 0]
        if not confident:
            fail(
                f"all {len(module_rows)} modules scored 0.0 confidence -- "
                "this is the signature of missing KEGG module graphs"
            )
        print(f"  modules with non-zero confidence: {len(confident)} / {len(module_rows)}")

        # 3. Module descriptions come from kegg_bacteria_modules.json, which was
        #    omitted from package_data in v0.1.2.
        described = [r for r in module_rows if (r.get("module_description") or "").strip()]
        if not described:
            fail("no module has a description -- kegg_bacteria_modules.json was not packaged")
        print(f"  modules with descriptions: {len(described)} / {len(module_rows)}")

        with open(modules_json) as handle:
            payload = json.load(handle)
        if not payload:
            fail("module JSON is empty")
        print(f"  modules in JSON: {len(payload)}")

    print("PASS: BLIMMP produced non-trivial results")


if __name__ == "__main__":
    main()
