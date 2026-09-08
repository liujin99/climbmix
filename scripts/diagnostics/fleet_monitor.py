#!/usr/bin/env python3
# ═══════════════════════════════════════════════════════════════════════
#  fleet_monitor.py — prod2 舰队监控探针 (我方远端作业状态时间线)
#
#  用法 (服务器, 仓库根目录):
#    python3 scripts/diagnostics/fleet_monitor.py result/prod2_k15bal_current
#    python3 scripts/diagnostics/fleet_monitor.py result/prod2_k15bal_current \
#        --interval 300          # 常驻轮询 (Ctrl-C 退出)
#
#  作业 id 来源 (自动合并去重):
#    run_dir/fleet_timeline.jsonl — 本探针的历史 (跨调用持久)
#    prod2_k15bal.log / run_dir/search.log / run_dir/dispatch_*.log 中
#    "[Exp N] submitted job <id>" / "[arm] submitted job <id>" 行
#
#  输出:
#    stdout — 当前快照表 (状态 + 本状态持续时长 + 排队等待) + 派生统计
#             (PENDING/RUNNING 计数, 峰值并发, 利用率 = RUNNING/max_concurrent_jobs,
#              已启动作业的排队等待 mean/max)
#    run_dir/fleet_timeline.jsonl — 每次轮询追加 {"t": epoch, "s": {job: status}}
#
#  诚实性: 后端 status() 无时间戳 → 一切时刻是"首次观测到该状态"的时刻;
#  探针中途才加入的作业 (首见即 RUNNING) 的排队等待标 "?" 不估。
#  平台全池状态 (他人作业) 本探针看不到 — 只盯自己的舰队。
# ═══════════════════════════════════════════════════════════════════════
import argparse
import glob as globmod
import json
import os
import re
import sys
import time

SUBMIT_RE = re.compile(r"\[([^\]]+)\] submitted job ([0-9a-f-]{8,})")


def _bootstrap_syspath():
    """让脚本在未 source env 的裸 shell 里也能 import climbmix:
    仓库根 = 本文件上两级; 补 src 与 vendored climbmix-ma。"""
    repo = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    for p in (os.path.join(repo, "src"),
              os.path.join(repo, "climbmix-ma")):
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)


def collect_jobs(sources):
    """log 文件列表 → {job_id: label}。"Exp 12" → "exp0012"。"""
    jobs = {}
    for path in sources:
        if not path or not os.path.isfile(path):
            continue
        try:
            with open(path, errors="replace") as f:
                for line in f:
                    m = SUBMIT_RE.search(line)
                    if not m:
                        continue
                    label, jid = m.group(1).strip(), m.group(2).strip()
                    if label.lower().startswith("exp "):
                        label = f"exp{int(label.split()[1]):04d}"
                    jobs[jid] = label
        except OSError:
            continue
    return jobs


def load_timeline(path):
    if not os.path.isfile(path):
        return []
    out = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if "t" in rec and "s" in rec:
                        out.append(rec)
                except ValueError:
                    continue
    except OSError:
        return []
    return out


def fmt_dur(sec):
    if sec is None:
        return "?"
    sec = max(0, int(sec))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def fmt_ts(epoch):
    if not epoch:
        return "?"
    return time.strftime("%H:%M:%S", time.localtime(epoch))


def derive_job_history(jid, timeline):
    """时间线 → {first_seen, pending_since, started, ended}。"""
    h = {"first_seen": None, "pending_since": None,
         "started": None, "ended": None}
    for rec in timeline:
        st = rec["s"].get(jid)
        if st is None:
            continue
        if h["first_seen"] is None:
            h["first_seen"] = rec["t"]
        if st == "PENDING" and h["pending_since"] is None:
            h["pending_since"] = rec["t"]
        if st == "RUNNING" and h["started"] is None:
            h["started"] = rec["t"]
        if st in ("SUCCEEDED", "FAILED", "CANCELLED") and h["ended"] is None:
            h["ended"] = rec["t"]
    return h


def main():
    ap = argparse.ArgumentParser(description="prod2 fleet monitor probe")
    ap.add_argument("run_dir", nargs="?",
                    default="result/prod2_k15bal_current")
    ap.add_argument("--main-log", default="prod2_k15bal.log",
                    help="主脚本 stdout 日志 (submitted job 行)")
    ap.add_argument("--interval", type=float, default=0.0,
                    help=">0: 常驻轮询间隔秒; 0: 单次快照")
    args = ap.parse_args()

    _bootstrap_syspath()
    try:
        from climbmix.remote.remote_executor import RemoteConfig
        from climbmix.remote.backends import resolve_backend
    except ImportError as e:
        print(f"[!] cannot import climbmix remote stack ({e}).\n"
              f"    Run from the repo root, or: "
              f"source /tmp/prod2_remote.env")
        return 1
    rc_path = os.path.join(args.run_dir, "remote_config.json")
    if not os.path.isfile(rc_path):
        print(f"[!] {rc_path} not found — wrong run dir?")
        return 1
    rc = RemoteConfig.from_json_file(rc_path)
    try:
        api = resolve_backend(rc).make_job_api(rc)
    except Exception as e:
        print(f"[!] backend init failed ({type(e).__name__}: {e}) — "
              f"check credentials / platform config")
        return 1

    timeline_path = os.path.join(args.run_dir, "fleet_timeline.jsonl")
    sources = [args.main_log,
               os.path.join(args.run_dir, "search.log")]
    sources += sorted(globmod.glob(
        os.path.join(args.run_dir, "dispatch_*.log")))
    sources += [os.path.join(args.run_dir, "arm_watcher.log")]

    while True:
        now = time.time()
        known = collect_jobs(sources)
        timeline = load_timeline(timeline_path)
        seen_ids = set()
        for rec in timeline:
            seen_ids.update(rec["s"].keys())
        all_ids = set(known.keys()) | seen_ids

        if not all_ids:
            print("no jobs found (no 'submitted job' lines yet, "
                  "empty timeline) — nothing to monitor")
        else:
            statuses = {}
            errors = {}
            for jid in all_ids:
                try:
                    statuses[jid] = api.status(jid).value
                except Exception as e:
                    statuses[jid] = "UNKNOWN"
                    errors[jid] = f"{type(e).__name__}: {e}"
            with open(timeline_path, "a") as f:
                f.write(json.dumps({"t": int(now), "s": statuses}) + "\n")
            timeline.append({"t": now, "s": statuses})

            counts = {}
            for st in statuses.values():
                counts[st] = counts.get(st, 0) + 1

            print("═" * 66)
            print(f"  fleet monitor — {args.run_dir}  "
                  f"({time.strftime('%F %H:%M:%S')})")
            print(f"  {len(all_ids)} jobs "
                  f"({', '.join(f'{k}={v}' for k, v in sorted(counts.items()))})"
                  f"  timeline: {len(timeline)} polls")
            print("═" * 66)

            order = sorted(all_ids, key=lambda j: (
                derive_job_history(j, timeline)["first_seen"] or now, j))
            hdr = (f"  {'label':<16} {'job':>10}  {'status':<10} "
                   f"{'state age':>9}  {'queue wait':>10}  note")
            print(hdr)
            print("  " + "-" * (len(hdr) - 2))
            queue_waits = []
            for jid in order:
                label = known.get(jid, "(unknown)")
                st = statuses[jid]
                h = derive_job_history(jid, timeline)
                # 本状态已持续多久: 最近一次状态变化 → now
                last_change = max(x for x in (
                    h["pending_since"], h["started"], h["ended"],
                    h["first_seen"]) if x is not None)
                age = now - last_change if last_change else None
                note = ""
                qw = None
                if h["started"] is not None:
                    base = h["pending_since"] or h["first_seen"]
                    if h["pending_since"] is not None:
                        qw = h["started"] - h["pending_since"]
                        queue_waits.append(qw)
                        note = f"started {fmt_ts(h['started'])}"
                    else:
                        note = "late-join (monitor saw it RUNNING already)"
                elif st == "PENDING":
                    base = h["pending_since"] or h["first_seen"]
                    qw = now - base if base else None
                    note = ("still queued — ≥ shown (monitor may have "
                            "joined late)" if h["pending_since"] is None
                            else "still queued")
                elif st in ("SUCCEEDED", "FAILED", "CANCELLED"):
                    note = (f"ended {fmt_ts(h['ended'])}"
                            if h["ended"] else "terminal")
                if jid in errors:
                    note = f"status() error: {errors[jid][:60]}"
                print(f"  {label:<16} {jid[:10]:>10}  {st:<10} "
                      f"{fmt_dur(age):>9}  {fmt_dur(qw):>10}  {note}")

            print("  " + "-" * (len(hdr) - 2))
            running = counts.get("RUNNING", 0)
            util = (f"{running}/{rc.max_concurrent_jobs} slots "
                    f"({100.0 * running / max(1, rc.max_concurrent_jobs):.0f}%)")
            peak = 0
            for rec in timeline:
                n = sum(1 for v in rec["s"].values() if v == "RUNNING")
                peak = max(peak, n)
            print(f"  realized concurrency now: {util}, peak observed: {peak}")
            if queue_waits:
                print(f"  queue wait (jobs that started, n={len(queue_waits)}): "
                      f"mean {fmt_dur(sum(queue_waits) / len(queue_waits))}, "
                      f"max {fmt_dur(max(queue_waits))}")
            print("═" * 66)

        if args.interval <= 0:
            break
        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n(bye)")
            break
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n(bye)")
        sys.exit(0)
