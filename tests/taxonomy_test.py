#!/usr/bin/env python
"""--taxonomy must change which priors are used.

It did not.  lineage_paths() required One_Hop_Refilled_*.json and
Two_Hop_Refilled_*.json under two directories that appear never to have been
part of the code, so the check failed for every taxonomy, every run fell back
to domain priors, and four runs differing only in -t produced byte identical
output.  Nothing crashed, because the two paths were returned and never read.

The integration section below would have caught that: run one input under
several taxonomies and require the results to differ.  The unit section covers
the name resolution that picks the priors.

Run inside the container, against the installed package:
    docker run --rm blimmp:ci python /app/tests/taxonomy_test.py
"""

import hashlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from BLIMMP_Scripts.module_detection import (
    available_priors,
    describe_priors,
    resolve_taxonomy,
)
import BLIMMP_Scripts.module_detection as md

PKG = Path(md.__file__).parent
COUNTS_DIR = PKG / "Data_Dependencies" / "ATB_Taxonomy_Frequency"
GRAPH_DIR = PKG / "Graph_Dependencies" / "MODULE_ALL_NEIGHBOR_DATA"

failures = []


def check(condition, message):
    if condition:
        print(f"  ok   {message}")
    else:
        print(f"  FAIL {message}")
        failures.append(message)


def expect_exit(taxonomy, available, why):
    try:
        got = resolve_taxonomy(taxonomy, available)
    except SystemExit:
        print(f"  ok   {taxonomy!r} rejected ({why})")
        return
    print(f"  FAIL {taxonomy!r} resolved to {got!r}, expected rejection ({why})")
    failures.append(f"{taxonomy!r} not rejected")


print("== discovery ==")
priors = available_priors(COUNTS_DIR, GRAPH_DIR)
check(len(priors) >= 20, f"found {len(priors)} prior sets in the installed package")
check("README.md" not in priors, "README.md is not offered as a taxon")
check("README" not in " ".join(priors), "no README-derived name leaked in")

by_rank = {}
for name, (_tag, rank, _c, _g) in priors.items():
    by_rank.setdefault(rank, []).append(name)
check(len(by_rank.get("domain", [])) == 1, "exactly one domain-level set")
check(by_rank.get("domain") == ["bacteria"], "the domain-level taxon is named bacteria, not domain")
check(len(by_rank.get("kingdom", [])) == 4, f"4 kingdoms, got {len(by_rank.get('kingdom', []))}")
check(len(by_rank.get("phylum", [])) >= 14, f"{len(by_rank.get('phylum', []))} phyla")

# Every -ati name is a kingdom and every -ota name a phylum.  The list this
# replaced filed mycoplasmatota as a kingdom, which the suffix contradicts.
for name, (_tag, rank, _c, _g) in priors.items():
    if name.endswith("ati"):
        check(rank == "kingdom", f"{name} is a kingdom")
    elif name.endswith("ota"):
        check(rank == "phylum", f"{name} is a phylum")

print("== every discovered name resolves to itself ==")
bad = [n for n in priors if resolve_taxonomy(n, priors) != n]
check(not bad, f"all {len(priors)} names round-trip" if not bad else f"these did not: {bad}")

print("== the two sets that used to be unreachable ==")
check("cyanobacteriota_melainabacteria_group" in priors, "cyanobacteria priors are reachable")
check("fusobacteriota" in priors, "fusobacteriota priors are reachable")
check(
    resolve_taxonomy("cyanobacteriota", priors) == "cyanobacteriota_melainabacteria_group",
    "'cyanobacteriota' reaches the melainabacteria group set",
)

print("== shortening, only where unambiguous ==")
check(resolve_taxonomy("fcb", priors) == "fcb_group", "'fcb' reaches fcb_group")
check(resolve_taxonomy("pvc", priors) == "pvc_group", "'pvc' reaches pvc_group")
for name in ("", "bacteria", "domain", "BACTERIA"):
    check(resolve_taxonomy(name, priors) == "bacteria", f"{name!r} means the whole domain")

expect_exit("cyano", priors, "an arbitrary truncation")
expect_exit("bacillat", priors, "an arbitrary truncation")
expect_exit("nonsense", priors, "nothing like a taxon name")

print("== a near miss points at the candidates ==")
# fusobacteri sits one letter from a kingdom and a phylum. Guessing between
# them would pick priors built from a different set of genomes, so it is
# refused, but the message should name the two rather than reprint all twenty.
for stem, expected in (
    ("fusobacteri", ("fusobacteriati", "fusobacteriota")),
    ("pseudomonad", ("pseudomonadati", "pseudomonadota")),
    ("thermotog",   ("thermotogati", "thermotogota")),
    ("cyano",       ("cyanobacteriota_melainabacteria_group",)),
    ("bacillat",    ("bacillati",)),
):
    try:
        resolve_taxonomy(stem, priors)
        check(False, f"{stem!r} should have been refused")
    except SystemExit as exc:
        message = str(exc)
        check(all(name in message for name in expected),
              f"{stem!r} is refused and named {', '.join(expected)}")
        check(any(f"({rank})" in message for rank in ("kingdom", "phylum")),
              f"{stem!r} suggestion carries the rank")

print("== the listing names the rank ==")
described = describe_priors(priors)
check("kingdom:" in described and "phylum:" in described, "output is grouped by rank")
check("bacillati" in described and "bacillota" in described, "both confusable names are listed")

print("== end to end: different taxonomy, different result ==")
example = Path("/app/Examples/example.domtblout")
if not example.exists():
    example = PKG.parent / "Examples" / "example.domtblout"

if not example.exists():
    print(f"  SKIP no example domtblout at {example}")
else:
    digests = {}
    with tempfile.TemporaryDirectory() as tmp:
        for taxonomy in ("bacteria", "cyanobacteriota", "actinomycetota", "bacillati"):
            prefix = os.path.join(tmp, taxonomy)
            result = subprocess.run(
                ["BLIMMP", str(example), "-f", "domtblout", "--sigma", "1.0",
                 "-t", taxonomy, "-o", prefix],
                capture_output=True, text=True,
            )
            if result.returncode != 0:
                print(f"  FAIL -t {taxonomy} exited {result.returncode}")
                print(result.stderr[-2000:])
                failures.append(f"-t {taxonomy} failed")
                continue
            out = Path(f"{prefix}_BLIMMP_module_probabilities.csv")
            if not out.exists():
                print(f"  FAIL -t {taxonomy} wrote no probabilities file")
                failures.append(f"-t {taxonomy} produced nothing")
                continue
            digests[taxonomy] = hashlib.md5(out.read_bytes()).hexdigest()

    for taxonomy, digest in digests.items():
        print(f"       {taxonomy:20s} {digest}")
    check(
        len(set(digests.values())) == len(digests),
        f"all {len(digests)} taxonomies gave distinct results "
        f"(got {len(set(digests.values()))} distinct)",
    )

print("== an unknown taxonomy stops the run ==")
if example.exists():
    with tempfile.TemporaryDirectory() as tmp:
        result = subprocess.run(
            ["BLIMMP", str(example), "-f", "domtblout", "--sigma", "1.0",
             "-t", "not_a_real_taxon", "-o", os.path.join(tmp, "x")],
            capture_output=True, text=True,
        )
        check(result.returncode != 0, "an invalid --taxonomy exits non-zero")
        combined = result.stdout + result.stderr
        check("unrecognized taxonomic group" in combined,
              "the error calls the name unrecognized")
        check("kingdom:" in combined, "the error lists the valid names by rank")

if failures:
    print(f"\nFAILED: {len(failures)}")
    for item in failures:
        print(f"  - {item}")
    sys.exit(1)

print("\nAll taxonomy checks passed.")
