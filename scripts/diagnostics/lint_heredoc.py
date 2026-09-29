#!/usr/bin/env python3
r"""Syntax-lint every python heredoc embedded in the repo's shell scripts.

Why (TODO:134-4, the 2026-09-12 44d9c4e lesson): a Python syntax error inside
a <<'PYEOF' block kills every subsequent launch, and none of the existing
gates see it — bash -n only parses the shell layer, the LAUNCH=0 dry-run gate
exits before heredoc bodies execute (and dry-runs never run most blocks
anyway). This tool closes the gap statically: extract each python-feeding
heredoc and compile() its body.

Scope: *.sh files under the roots (default: runs/ + scripts/diagnostics/,
resolved from the repo root regardless of cwd). A heredoc is linted when its
start line invokes python (python3 / python / $PY ...), quoted or not —
unquoted bodies undergo shell expansion, but in practice the expansions live
inside python string literals, and a body that does NOT compile as-written is
exactly the class of authoring mistake worth flagging. Non-python heredocs
(cat <<MSG, read <<EOF ...) are skipped and counted.

Usage:
    python3 scripts/diagnostics/lint_heredoc.py                 # default roots
    python3 scripts/diagnostics/lint_heredoc.py path [path ...] # explicit roots/files
    python3 scripts/diagnostics/lint_heredoc.py --selftest      # built-in fixtures

Run alongside check_repo_secrets.py before every push.
Exit 0 = every block compiles; exit 1 = any failure.
"""
import argparse
import os
import re
import sys
import tempfile

HEREDOC_START = re.compile(r"<<(-?)\s*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\2")
PY_INVOCATION = re.compile(r"\bpython|\$PY")


def _is_terminator(line, tag, dash):
    # bash: non-dash terminators start at column 0; <<- tolerates leading tabs.
    if dash:
        return line.lstrip("\t").rstrip() == tag
    return line.rstrip() == tag and line[:1] not in (" ", "\t")


def extract_blocks(text):
    """Yield (tag, start_lineno_1based, start_line, body_lines, terminated)."""
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        m = HEREDOC_START.search(lines[i])
        if not m:
            i += 1
            continue
        dash, _, tag = m.groups()
        body = []
        j = i + 1
        terminated = False
        while j < len(lines):
            if _is_terminator(lines[j], tag, dash):
                terminated = True
                break
            body.append(lines[j])
            j += 1
        yield tag, i + 1, lines[i], body, terminated
        i = (j if terminated else j) + 1


def _command_context(lines, i):
    """Join line i with its backslash-continuation ancestors (multi-line
    commands put the python3 invocation lines above the <<'PY' line)."""
    ctx = lines[i]
    k = i - 1
    while k >= 0 and lines[k].rstrip().endswith("\\"):
        ctx = lines[k] + " " + ctx
        k -= 1
    return ctx


def lint_file(path):
    """Return (n_linted, n_skipped, failures); failures = (path, lineno, msg)."""
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read()
    except (OSError, UnicodeDecodeError) as e:
        return 0, 0, [(path, 1, f"unreadable: {e}")]
    n_linted = n_skipped = 0
    failures = []
    lines = text.splitlines()
    for tag, start, _start_line, body, terminated in extract_blocks(text):
        if not PY_INVOCATION.search(_command_context(lines, start - 1)):
            n_skipped += 1
            continue
        n_linted += 1
        if not terminated:
            failures.append((path, start, f"unterminated heredoc <<{tag}"))
            continue
        src = "\n".join(body) + "\n"
        try:
            compile(src, f"{path}:{start}", "exec")
        except SyntaxError as e:
            # body line k lives at file line start + k (start = heredoc opener)
            lineno = start + (e.lineno or 1)
            failures.append((path, lineno, f"SyntaxError: {e.msg}"))
    return n_linted, n_skipped, failures


def collect(paths):
    files = []
    for p in paths:
        if os.path.isdir(p):
            for root, _dirs, names in os.walk(p):
                files.extend(os.path.join(root, n) for n in names
                             if n.endswith(".sh"))
        else:
            files.append(p)
    return sorted(set(files))


def run(paths):
    # scripts/diagnostics/lint_heredoc.py -> three dirnames up = repo root
    repo = os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))
    if not paths:
        paths = [os.path.join(repo, "runs"),
                 os.path.join(repo, "scripts", "diagnostics")]
    files = collect(paths)
    if not files:
        print("lint_heredoc: no .sh files found under the given roots")
        return 1
    total_linted = total_skipped = 0
    failures = []
    for f in files:
        n_linted, n_skipped, fails = lint_file(f)
        total_linted += n_linted
        total_skipped += n_skipped
        failures.extend(fails)
    for path, lineno, msg in failures:
        print(f"  FAIL {path}:{lineno}: {msg}")
    print(f"lint_heredoc: {len(files)} files, {total_linted} python heredocs "
          f"linted, {total_skipped} non-python heredocs skipped, "
          f"{len(failures)} failures")
    return 1 if failures else 0


GOOD_SH = """#!/usr/bin/env bash
python3 - <<'PYEOF'
import json
print(json.dumps({"ok": 1}))
PYEOF
cat <<MSG
not python at all: {{{ $HOME
MSG
"""

BAD_SH = """#!/usr/bin/env bash
$PY - <<PYEOF
x = (
PYEOF
echo done
"""

UNTERMINATED_SH = """#!/usr/bin/env bash
python3 - <<'PY'
import os
# terminator never comes
"""

NONPY_SH = """#!/usr/bin/env bash
IFS='|' read -r A B <<EOF
x|y
EOF
"""

MULTILINE_SH = """#!/usr/bin/env bash
python3 - "$OUT" \\
    "$KEY" <<'PYEOF'
import json, sys
print(json.dumps({"argv": len(sys.argv)}))
PYEOF
cat file \\
    <<'MSG'
prose, not python: {{{
MSG
"""


def selftest():
    ok = True

    def expect(name, cond):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        ok = ok and cond

    with tempfile.TemporaryDirectory(prefix="lint_heredoc_") as td:
        good = os.path.join(td, "good.sh")
        bad = os.path.join(td, "bad.sh")
        unterm = os.path.join(td, "unterm.sh")
        nonpy = os.path.join(td, "nonpy.sh")
        for p, content in ((good, GOOD_SH), (bad, BAD_SH),
                           (unterm, UNTERMINATED_SH), (nonpy, NONPY_SH)):
            with open(p, "w") as f:
                f.write(content)
        n, s, fails = lint_file(good)
        expect("good.sh: 1 linted, 1 skipped, 0 failures",
               n == 1 and s == 1 and not fails)
        n, s, fails = lint_file(bad)
        expect("bad.sh: unquoted $PY block linted, 1 failure",
               n == 1 and s == 0 and len(fails) == 1)
        expect("bad.sh: failure points at the broken line",
               fails and fails[0][1] == 3 and "SyntaxError" in fails[0][2])
        n, s, fails = lint_file(unterm)
        expect("unterminated heredoc reported",
               n == 1 and len(fails) == 1 and "unterminated" in fails[0][2])
        n, s, fails = lint_file(nonpy)
        expect("read <<EOF skipped (not python-feeding)",
               n == 0 and s == 1 and not fails)
        multi = os.path.join(td, "multi.sh")
        with open(multi, "w") as f:
            f.write(MULTILINE_SH)
        n, s, fails = lint_file(multi)
        expect("multi.sh: continuation-line heredoc linted, cat <<MSG skipped",
               n == 1 and s == 1 and not fails)
        rc = run([td])
        expect("run() on the fixture dir exits 1", rc == 1)
    print("lint_heredoc selftest:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="*",
                    help="roots or .sh files (default: runs/ + scripts/diagnostics/)")
    ap.add_argument("--selftest", action="store_true",
                    help="run built-in fixtures and exit")
    args = ap.parse_args()
    return selftest() if args.selftest else run(args.paths)


if __name__ == "__main__":
    sys.exit(main())
