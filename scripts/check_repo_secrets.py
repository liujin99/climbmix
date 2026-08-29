#!/usr/bin/env python3
"""Secret/internal-value guard for the (PUBLIC) climbmix repo.

Scans git-tracked files for sensitive patterns and fails loudly on a hit.
Two pattern sources:

  1. Built-in GENERIC patterns (safe to live in the repo): private keys,
     credential assignments with values, JWT-looking blobs.
  2. An optional external patterns file (~/.config/climbmix/
     guard_patterns.json — gitignored, NEVER committed) holding the
     internal-specific values: gateway hostnames, pool/workspace/project
     IDs, image repo ids, bucket names. Since listing those values here
     would itself be a leak, the file lives outside the repo; a missing
     file just means the internal-specific scan is skipped.

Usage:
    python3 scripts/check_repo_secrets.py                 # tracked files
    python3 scripts/check_repo_secrets.py --staged        # + staged diff
    python3 scripts/check_repo_secrets.py --patterns-file /path/x.json

Exit 0 = clean; exit 1 = hits (printed as file:line). Run before every
push that touches remote/ or docs/ (wire into pre-push hooks or CI).
"""

import argparse
import json
import os
import re
import subprocess
import sys

REPO_ROOT = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".."))

DEFAULT_PATTERNS_FILE = os.path.join(
    os.path.expanduser("~"), ".config", "climbmix", "guard_patterns.json")

# Generic secret shapes. The internal-specific values (gateway hosts,
# pool/project IDs, bucket names) come from the external patterns file —
# see module docstring.
BUILTIN_PATTERNS = [
    # PEM private key blocks
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    # Credential-looking assignments with non-empty values
    r"(?i)\b(access_key|secret_key|api[_-]?key|passwd|password|token)\b"
    r"\s*[:=]\s*['\"][^'\"{}<>\s]{8,}['\"]",
    # Long JWT-style blobs (header.payload.signature, base64url segments)
    r"\beyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\b",
    # Huawei-style AK/SK pairs (15-25 uppercase-alnum secrets)
    r"\b[A-Z0-9]{20}\b.*\b[A-Za-z0-9+/]{32,}\b",
]

SKIP_SUFFIXES = (
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".pdf", ".zip", ".gz",
    ".tar", ".tgz", ".pt", ".npz", ".npy", ".parquet", ".bin", ".ckpt",
    ".so", ".dylib", ".lock",
)


def load_patterns(patterns_file: str):
    patterns = list(BUILTIN_PATTERNS)
    if patterns_file and os.path.isfile(patterns_file):
        with open(patterns_file, "r", encoding="utf-8") as f:
            extra = json.load(f)
        if not isinstance(extra, list):
            print(f"✗ {patterns_file}: expected a JSON list of regex "
                  f"strings", file=sys.stderr)
            sys.exit(2)
        patterns.extend(extra)
        print(f"· external patterns: {len(extra)} (from {patterns_file})")
    elif patterns_file:
        print(f"· no external patterns file at {patterns_file} — skipping "
              f"internal-specific scan")
    return patterns


def tracked_files(root: str, include_staged: bool):
    files = subprocess.run(
        ["git", "-C", root, "ls-files", "-z"],
        capture_output=True, check=True).stdout.decode().split("\0")
    files = [f for f in files if f]
    if include_staged:
        staged = subprocess.run(
            ["git", "-C", root, "diff", "--cached", "--name-only", "-z"],
            capture_output=True, check=True).stdout.decode().split("\0")
        for f in (f for f in staged if f):
            if f not in files:
                files.append(f)
    return files


def scan_file(path: str, regexes):
    """Yield (line_no, line, pattern_str) hits for one file.

    `regexes` is a list of (pattern_str, compiled_regex) pairs.
    """
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f, 1):
                for p, rx in regexes:
                    if rx.search(line):
                        yield i, line.rstrip(), p
                        break  # one hit per line is enough
    except OSError:
        pass


def scan(root: str, patterns, files=None, include_staged: bool = False):
    """Scan `files` (default: git-tracked under root). Returns hit list of
    (relpath, line_no, line, pattern_source)."""
    regexes = [(p, re.compile(p)) for p in patterns]
    if files is None:
        files = tracked_files(root, include_staged)
    hits = []
    for rel in files:
        if rel.endswith(SKIP_SUFFIXES):
            continue
        full = os.path.join(root, rel)
        if not os.path.isfile(full):
            continue
        for i, line, p in scan_file(full, regexes):
            src = "builtin" if p in BUILTIN_PATTERNS else "external"
            hits.append((rel, i, line, src))
    return hits


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--patterns-file", default=DEFAULT_PATTERNS_FILE)
    ap.add_argument("--staged", action="store_true",
                    help="also scan files staged for commit")
    ap.add_argument("--root", default=REPO_ROOT)
    args = ap.parse_args()

    patterns = load_patterns(args.patterns_file)
    hits = scan(args.root, patterns, include_staged=args.staged)
    if hits:
        print(f"\n✗ {len(hits)} potential secret/internal-value hit(s):")
        for rel, i, line, src in hits:
            print(f"  {rel}:{i}  [{src}]  {line.strip()[:120]}")
        print("\nIf these are REAL values: remove them, rotate the "
              "credential (it must be considered exposed), and re-run. "
              "If they are false positives: refine the patterns in "
              f"{args.patterns_file}.")
        return 1
    print("✓ clean — no secret/internal-value patterns found")
    return 0


if __name__ == "__main__":
    sys.exit(main())
