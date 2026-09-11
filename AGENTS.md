# AGENTS.md — climbmix

## Server (ModelArts master) conventions

- **Temp files & one-off scripts: `~/work/tmp/<name>`** — never `~/work/`
  directly (keeps the workspace clean; one `rm -rf ~/work/tmp/*` reclaims
  everything). Do NOT use the system `/tmp` on the server (can be tmpfs,
  size-limited, wiped on reboot; smoke mini-data is GB-scale).
- Long-lived repos stay at `~/work/<repo>` (climbmix, nanochat-npu,
  climbmix-ma; the latter is symlinked into climbmix/climbmix-ma as the
  vendored backend the dispatch scripts bootstrap onto sys.path).
- User's terminal intermittently mangles pastes (hyphen→U+2011, dropped
  chars): deliver multi-line content as single-paste heredocs, then verify
  with `bash -n` + `grep -cF` expected-count checks before executing.
- Communication with the user is in Chinese.

## Local dev machine conventions

- Temp/test scratch work: `/tmp/opencode/` (pre-created, outside the repo).
- Test entrypoints: `python3 scripts/diagnostics/test_prod2_runtime.py`,
  `python3 scripts/diagnostics/test_prod2_fixes.py` (standalone, no pytest),
  plus legacy suites under `/tmp/opencode/test_*.py`.
- Run `python3 scripts/check_repo_secrets.py` before every push.

## Dual-session commit discipline (two agents share ONE working tree)

- **Always `git add <explicit paths>`** — never `git add -A` / `-u` /
  directory-level adds: they sweep the OTHER session's uncommitted hunks
  into your commit (already happened twice: e5c4c5f committed a mid-state,
  c169896 had to hotfix the missing module).
- **Verify before committing**: `git diff --cached --stat` must list ONLY
  files this session intends to change.
- **Commit small & fast**: land a change once it's tested — minimize the
  dirty window the other session can accidentally sweep.
- **Shared files** (runs/run_climbmix.sh, scripts/dispatch_target_arm.py,
  src/climbmix/pipeline/target_runner.py, src/climbmix/utils/fingerprint.py,
  …): check `git status` for the other session's dirty edits BEFORE
  editing; coordinate through the user when both need the same file.
- **New .py on an executable path** must be classified the same commit:
  add to `utils/fingerprint.py` `GLOBAL_EXCLUDE` (observability/launcher
  tools) or its stage's file set — an unclassified new module shifts BOTH
  stage fingerprints.
- **WIP claims (hook-enforced)**: while a file is in flight, list it in
  `.git/wip_claims/<session>.list` (repo-relative, one per line) and
  commit with `AGENT_ID=<session>` — the pre-commit hook blocks commits
  sweeping files claimed by another session (unidentified committers are
  blocked on ANY claim; deliberate cross-claims go through the user or
  `--no-verify`). Clear your claim file after landing.
