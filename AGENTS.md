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
