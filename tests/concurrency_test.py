#!/usr/bin/env python3
"""Several BLIMMP processes must be able to extract the module graphs at once.

Nextflow dispatches one BLIMMP task per genome, so a run with N genomes puts N
processes on a node at the same time.  When the graphs are not already
extracted, every one of them extracts into the same directory.

Extracting straight into that shared directory made them destroy each other's
work: one process removed __MACOSX or flattened the nested folder while another
was still reading from it, and the resulting OSError looked to the caller like
an unwritable destination.  On a 9-genome run, 4 tasks died with a FATAL
"failed to extract" before any analysis ran.  A single-process test cannot see
this, which is how it shipped.
"""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

WORKERS = 8
EXPECTED_GRAPHS = 340

CHILD = """
import sys
from BLIMMP_Scripts.module_detection import ensure_module_graphs
print(ensure_module_graphs(sys.argv[1]))
"""


def main() -> int:
    import BLIMMP_Scripts

    packaged = Path(BLIMMP_Scripts.__file__).parent / "Graph_Dependencies"
    zip_name = "KEGG_Graphs_Generated_March26.zip"
    if not (packaged / zip_name).is_file():
        print(f"FAIL: {zip_name} is not in the installed package at {packaged}")
        return 1

    scratch = Path(tempfile.mkdtemp(prefix="blimmp-concurrency-"))
    try:
        # A Graph_Dependencies directory holding only the archive, so every
        # worker has to extract rather than finding graphs already in place.
        graph_dir = scratch / "Graph_Dependencies"
        graph_dir.mkdir()
        shutil.copy2(packaged / zip_name, graph_dir)

        procs = [
            subprocess.Popen(
                [sys.executable, "-c", CHILD, str(graph_dir)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            for _ in range(WORKERS)
        ]
        results = [(p.wait(), *p.communicate()) for p in procs]

        failed = [(i, err or out) for i, (rc, out, err) in enumerate(results) if rc != 0]
        if failed:
            print(f"FAIL: {len(failed)}/{WORKERS} workers exited non-zero")
            for i, msg in failed[:3]:
                last = msg.strip().splitlines()[-1] if msg.strip() else "(no output)"
                print(f"  worker {i}: {last}")
            return 1

        extracted = graph_dir / "KEGG_Graphs_Generated_March26"
        graphs = list(extracted.glob("module_*_nodes.json"))
        if len(graphs) != EXPECTED_GRAPHS:
            print(f"FAIL: expected {EXPECTED_GRAPHS} module graphs, found {len(graphs)}")
            return 1

        # Every worker must agree on where the graphs ended up. The extracting
        # worker also prints a progress line, so read the path off the last one.
        returned = {out.strip().splitlines()[-1] for _, out, _ in results if out.strip()}
        if returned != {str(extracted)}:
            print(f"FAIL: workers disagreed on the graph directory: {sorted(returned)}")
            return 1

        # A crashed or abandoned extraction leaves its staging directory behind.
        leftovers = [p.name for p in graph_dir.iterdir() if p.name.startswith(".")]
        if leftovers:
            print(f"FAIL: staging directories left behind: {leftovers}")
            return 1

        print(f"PASS: {WORKERS} concurrent workers, {len(graphs)} graphs, no leftovers")
        return 0
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
