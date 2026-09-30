#!/usr/bin/env python3
# ═══════════════════════════════════════════════════════════════════════
#  mfu_probe.py — 20-step 训练计时探针 (mfu #11 / TODO:132)
#
#  目的: 拆解 d28 生产训练的每步耗时构成, 验证"优化器小路径 ~550+ 条
#  逐参数 collectives 是大头"假设, 为通信合并候选 (~570→~40) 定优先级。
#  不改 nanochat 任何代码 (零代码漂移) — 复刻 scripts/mid_train.py 的
#  步骤体, 数据换 synthetic (randint), 逐段计时:
#
#    fwd+bwd (×grad_accum) / NaN 链 (逐参数 isnan→bool 同步, 生产同款) /
#    clip_grad_norm / nan_flag all_reduce / optimizer.step / zero_grad
#
#  + 通信普查: monkey-patch dist.{all_reduce, reduce_scatter_tensor,
#    all_gather_into_tensor, broadcast} — 每步调用数 / 张量字节 /
#    host 提交时间 / future.wait 等待时间, 按算子分桶。
#  + 优化器组 census: 每组 (kind/shape/n/路径), 预测 vs 实测 collectives。
#  + MFU = estimate_flops × total_batch / (dt × peak × ws)。
#
#  用法 (服务器本地 ws=8 验证):
#    source $CLIMBMIX_DIR/runs/lib/npu_env.sh
#    cd $NANOCHAT_DIR && torchrun --standalone --nproc_per_node=8 \
#        $CLIMBMIX_DIR/scripts/diagnostics/mfu_probe.py \
#        --nanochat-dir $NANOCHAT_DIR --steps 20 --warmup 4 \
#        --out $CLIMBMIX_DIR/result/mfu_probe_ws8.json
#
#  生产形状 ws=64 (db=1 → grad_accum=8): 走远端 dispatch (本地验证后接)。
#  注意: 数据侧 (tokenizing loader) 不在本探针内 — 生产 dt − 探针 dt ≈
#  数据侧 + 评估间歇; 探针给的是计算/通信/步骤体的纯账。
# ═══════════════════════════════════════════════════════════════════════
import argparse
import json
import os
import statistics
import sys
import time

parser = argparse.ArgumentParser(
    description="mfu probe: dissect one training step (fwd/bwd, nan-chain, "
                "clip, optimizer, comm census) — no nanochat code changes")
parser.add_argument("--nanochat-dir", default="",
                    help="nanochat-npu 树 (缺省取 env NANOCHAT_DIR)")
parser.add_argument("--model-tag", default="",
                    help="base 模型 tag (缺省 = base_checkpoints 里最大模型 = d28)")
parser.add_argument("--model-step", type=int, default=-1,
                    help="-1 = 最后一步")
parser.add_argument("--device-batch-size", type=int, default=1,
                    help="生产臂锁定 db=1 (内存墙, dispatch rev3)")
parser.add_argument("--total-batch-size", type=int, default=1048576,
                    help="每步总 token (2^20, 与生产臂一致)")
parser.add_argument("--max-seq-len", type=int, default=2048)
parser.add_argument("--steps", type=int, default=20)
parser.add_argument("--warmup", type=int, default=4)
parser.add_argument("--out", default="mfu_probe.json")
parser.add_argument("--png", default="",
                    help="饼图输出 (缺省 = --out 同名 .png; 'no' 关闭)")
# torchrun 兼容: 部分版本向脚本注入 --local-rank (mid_train 同环境实测
# 未注入, 防御性接住两种行为)
parser.add_argument("--local-rank", "--local_rank", type=int, default=0)
args = parser.parse_args()

# ── 自举 nanochat 路径 (必须在 nanochat 导入前) ──
_nanochat_dir = args.nanochat_dir or os.environ.get("NANOCHAT_DIR", "")
if not _nanochat_dir:
    raise SystemExit("需要 --nanochat-dir 或环境变量 NANOCHAT_DIR")
if _nanochat_dir not in sys.path:
    sys.path.insert(0, _nanochat_dir)

import torch                                   # noqa: E402
import torch.distributed as dist               # noqa: E402
from nanochat.common import (compute_init, compute_cleanup, print0,   # noqa: E402
                             autodetect_device_type, get_peak_flops,
                             COMPUTE_DTYPE, COMPUTE_DTYPE_REASON)
from nanochat.checkpoint_manager import load_model   # noqa: E402

# ═══════════════════════════════════════════════════════════════════════
# 计算初始化 (与 mid_train 同款)
# ═══════════════════════════════════════════════════════════════════════
device_type = autodetect_device_type()
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0
print0(f"COMPUTE_DTYPE: {COMPUTE_DTYPE} ({COMPUTE_DTYPE_REASON})")

if device_type == "npu":
    synchronize = torch.npu.synchronize
    peak_flops = get_peak_flops(torch.npu.get_device_name(0))
    print0(f"NPU: {torch.npu.get_device_name(0)} | Peak FLOPS (BF16): {peak_flops:.2e}")
elif device_type == "cuda":
    synchronize = torch.cuda.synchronize
    peak_flops = get_peak_flops(torch.cuda.get_device_name(0))
else:
    synchronize = lambda: None   # noqa: E731
    peak_flops = float("inf")

# ═══════════════════════════════════════════════════════════════════════
# 通信普查 (在 compute_init 之后安装, 吃不到启动期的组网噪声)
# ═══════════════════════════════════════════════════════════════════════
_DIST_OPS = ("all_reduce", "reduce_scatter_tensor",
             "all_gather_into_tensor", "broadcast")
_NOW = {op: {"calls": 0, "sync_calls": 0, "bytes": 0.0,
             "launch_s": 0.0, "wait_s": 0.0} for op in _DIST_OPS}


class _TimedFuture:
    __slots__ = ("_fut", "_op")

    def __init__(self, fut, op):
        self._fut, self._op = fut, op

    def wait(self):
        t0 = time.perf_counter()
        self._fut.wait()
        _NOW[self._op]["wait_s"] += time.perf_counter() - t0

    def __getattr__(self, name):
        return getattr(self._fut, name)


class _TimedWork:
    __slots__ = ("_work", "_op")

    def __init__(self, work, op):
        self._work, self._op = work, op

    def get_future(self):
        return _TimedFuture(self._work.get_future(), self._op)

    def __getattr__(self, name):
        return getattr(self._work, name)


def _install_dist_probe():
    for op in _DIST_OPS:
        real = getattr(dist, op)

        def _make(op, real):
            def _wrapper(*a, **kw):
                nbytes = sum(t.numel() * t.element_size()
                             for t in a if isinstance(t, torch.Tensor))
                t0 = time.perf_counter()
                ret = real(*a, **kw)
                launch = time.perf_counter() - t0
                d = _NOW[op]
                d["calls"] += 1
                d["bytes"] += nbytes
                d["launch_s"] += launch
                if kw.get("async_op") and ret is not None:
                    return _TimedWork(ret, op)
                if not kw.get("async_op"):
                    d["sync_calls"] += 1
                return ret
            return _wrapper
        setattr(dist, op, _make(op, real))


def _snapshot():
    return {op: dict(v) for op, v in _NOW.items()}


def _delta(before, after):
    return {op: {k: after[op][k] - before[op][k] for k in before[op]}
            for op in before}


_install_dist_probe()

# ═══════════════════════════════════════════════════════════════════════
# 模型 / 优化器 / 形状 (生产臂同款: db=1, total=2^20, GA 推导)
# ═══════════════════════════════════════════════════════════════════════
model, tokenizer, meta = load_model(
    "base", device, phase="train",
    model_tag=(args.model_tag or None),
    step=(None if args.model_step < 0 else args.model_step))
vocab = tokenizer.get_vocab_size()
num_flops_per_token = model.estimate_flops()
mss, db, tbs = args.max_seq_len, args.device_batch_size, args.total_batch_size
world_tokens = db * mss * ddp_world_size
assert tbs % world_tokens == 0, (
    f"total_batch {tbs:,} 不能被 db×mss×ws={world_tokens:,} 整除 — "
    f"形状与生产不一致, 拒跑")
grad_accum = tbs // world_tokens
print0(f"shape: ws={ddp_world_size} db={db} mss={mss} "
       f"grad_accum={grad_accum} total_batch={tbs:,} "
       f"flops/token={num_flops_per_token:,.0f}")

# LR 数值不影响计时; 用 mid_train 继承缺省量级
optimizer = model.setup_optimizer(unembedding_lr=0.008,
                                  embedding_lr=0.3,
                                  matrix_lr=0.02,
                                  weight_decay=0.05)

# ── 优化器组 census (预测口径 vs 实测对账) ──


def _group_census():
    ws = ddp_world_size
    small_min = getattr(type(optimizer), "_SMALL_PATH_MIN_WS", None)
    rows, predicted = [], 0
    for g in optimizer.param_groups:
        params = g.get("params", [])
        if not params:
            continue
        kind, n = g.get("kind"), len(params)
        shape = tuple(params[0].shape)
        if kind == "muon":
            is_small = (small_min is not None and ws > small_min and n < ws)
            path = ("small: per-param all_reduce + broadcast"
                    if is_small else "stacked: reduce_scatter + all_gather")
            colls = 2 * n if is_small else 2
        elif kind == "adamw":
            n_small = sum(1 for p in params if p.numel() < 1024)
            path = f"adamw: {n_small} small(ar) + {n - n_small} large(rs+ag)"
            colls = n_small + 2 * (n - n_small)
        else:
            path, colls = kind or "?", 0
        predicted += colls
        rows.append({"kind": kind, "n_params": n, "shape": list(shape),
                     "path": path, "predicted_collectives": colls})
    return rows, predicted


group_rows, predicted_colls = _group_census()
if master_process:
    print0(f"optimizer groups ({len(group_rows)}):")
    for r in group_rows:
        print0(f"  {r['kind']:>5} n={r['n_params']:>3} "
               f"shape={tuple(r['shape'])} → {r['path']}")
    print0(f"predicted collectives/step ≈ {predicted_colls} "
           f"(+1 nan_flag sync all_reduce)")

# ═══════════════════════════════════════════════════════════════════════
# 步骤体 (复刻 mid_train :371-414; 数据换 synthetic)
# ═══════════════════════════════════════════════════════════════════════
records = []
total_steps = args.warmup + args.steps

for step in range(total_steps):
    x = torch.randint(1, vocab, (db, mss), device=device)
    y = torch.randint(1, vocab, (db, mss), device=device)

    synchronize()
    t0 = time.perf_counter()
    snap0 = _snapshot()

    # fwd+bwd × grad_accum (含生产同款 loss.detach().item() 逐步同步)
    t_fb = time.perf_counter()
    for _ in range(grad_accum):
        loss = model(x, y)
        loss = loss / grad_accum
        loss.backward()
        loss.detach().item()
    synchronize()
    t_fb = time.perf_counter() - t_fb

    # NaN 链 (生产逐参数同款: isnan→any→bool 逐参数设备同步)
    t_nan = time.perf_counter()
    has_nan = any(p.grad is not None and torch.isnan(p.grad).any()
                  for p in model.parameters())
    if dist.is_initialized():
        dev = next(model.parameters()).device
        nan_flag = torch.tensor([1.0 if has_nan else 0.0], device=dev)
        dist.all_reduce(nan_flag, op=dist.ReduceOp.MAX)
        has_nan = nan_flag.item() > 0
    synchronize()
    t_nan = time.perf_counter() - t_nan
    if has_nan:
        print0(f"[probe] step {step}: NaN grads (synthetic 数据不该出现) — "
               f"继续计时但结果标记 suspect")
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        model.zero_grad(set_to_none=True)
        continue

    t_clip = time.perf_counter()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    synchronize()
    t_clip = time.perf_counter() - t_clip

    t_opt = time.perf_counter()
    optimizer.step()
    synchronize()
    t_opt = time.perf_counter() - t_opt

    t_zero = time.perf_counter()
    model.zero_grad(set_to_none=True)
    synchronize()
    t_zero = time.perf_counter() - t_zero

    dt = time.perf_counter() - t0
    census = _delta(snap0, _snapshot())

    if step >= args.warmup:
        records.append({"step": step, "dt": dt, "fwdbwd": t_fb,
                        "nan_chain": t_nan, "clip": t_clip,
                        "optimizer": t_opt, "zero_grad": t_zero,
                        "census": census})
        if master_process:
            mfu = 100 * num_flops_per_token * tbs / (dt * peak_flops
                                                     * ddp_world_size)
            print0(f"step {step:03d} | dt {dt*1000:8.1f}ms "
                   f"(fb {t_fb*1000:7.1f} / nan {t_nan*1000:6.1f} / "
                   f"clip {t_clip*1000:6.1f} / opt {t_opt*1000:7.1f} / "
                   f"zero {t_zero*1000:5.1f}) | mfu {mfu:5.2f}% | "
                   f"colls {sum(c['calls'] for c in census.values())}")

# ═══════════════════════════════════════════════════════════════════════
# 报告 (rank 0)
# ═══════════════════════════════════════════════════════════════════════
if master_process and records:
    def med(key):
        return statistics.median(r[key] for r in records)

    def census_med(op, key):
        return statistics.median(r["census"][op][key] for r in records)

    n_colls_med = statistics.median(
        sum(c["calls"] for c in r["census"].values()) for r in records)
    opt_launch = sum(census_med(op, "launch_s") for op in _DIST_OPS)
    opt_wait = sum(census_med(op, "wait_s") for op in _DIST_OPS)
    mfu = 100 * num_flops_per_token * tbs / (med("dt") * peak_flops
                                             * ddp_world_size)

    print0("\n═══ mfu probe 汇总 ═══")
    print0(f"shape: ws={ddp_world_size} db={db} GA={grad_accum} "
           f"tbs={tbs:,} | dtype={COMPUTE_DTYPE} | "
           f"steps={len(records)} (median)")
    print0(f"每步: dt {med('dt')*1000:.1f}ms = fwd/bwd "
           f"{med('fwdbwd')*1000:.1f} + nan链 {med('nan_chain')*1000:.1f} "
           f"+ clip {med('clip')*1000:.1f} + 优化器 {med('optimizer')*1000:.1f} "
           f"+ zero {med('zero_grad')*1000:.1f}")
    print0(f"MFU: {mfu:.2f}% (peak {peak_flops:.2e} × {ddp_world_size})")
    print0(f"优化器分解: host 提交 {opt_launch*1000:.1f}ms + "
           f"collective 等待 {opt_wait*1000:.1f}ms + "
           f"残余(计算+胶水) "
           f"{(med('optimizer') - opt_launch - opt_wait)*1000:.1f}ms")
    print0(f"collectives/步: 实测 {n_colls_med:.0f} vs 预测 "
           f"{predicted_colls + 1} (含 nan_flag)")
    for op in _DIST_OPS:
        c = {"calls": census_med(op, "calls"),
             "sync": census_med(op, "sync_calls"),
             "MB": census_med(op, "bytes") / 1e6,
             "launch_ms": census_med(op, "launch_s") * 1000,
             "wait_ms": census_med(op, "wait_s") * 1000}
        if c["calls"] > 0 or c["sync"] > 0:
            print0(f"  {op:24s} calls/步 {c['calls']:7.0f} (sync {c['sync']:.0f})"
                   f"  {c['MB']:8.1f}MB  launch {c['launch_ms']:7.1f}ms"
                   f"  wait {c['wait_ms']:7.1f}ms")

    out = {"env": {"world_size": ddp_world_size, "device": device_type,
                   "peak_flops": peak_flops, "dtype": str(COMPUTE_DTYPE)},
           "shape": {"device_batch": db, "max_seq_len": mss,
                     "grad_accum": grad_accum, "total_batch": tbs,
                     "flops_per_token": num_flops_per_token},
           "median_ms": {k: med(k) * 1000 for k in
                         ("dt", "fwdbwd", "nan_chain", "clip", "optimizer",
                          "zero_grad")},
           "mfu_pct": mfu,
           "optimizer_decomp_ms": {"host_launch": opt_launch * 1000,
                                   "collective_wait": opt_wait * 1000,
                                   "residual": (med("optimizer")
                                                - opt_launch - opt_wait) * 1000},
           "census_per_step_median": {
               op: {"calls": census_med(op, "calls"),
                    "sync_calls": census_med(op, "sync_calls"),
                    "MB": census_med(op, "bytes") / 1e6,
                    "launch_ms": census_med(op, "launch_s") * 1000,
                    "wait_ms": census_med(op, "wait_s") * 1000}
               for op in _DIST_OPS},
           "collectives_measured": n_colls_med,
           "collectives_predicted": predicted_colls + 1,
           "groups": group_rows,
           "steps": records}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print0(f"json → {args.out}")

    # 饼图 (可选; matplotlib 缺失静默降级)
    png = args.png or (os.path.splitext(args.out)[0] + ".png")
    if png.lower() not in ("no", "off", "none"):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            other = max(0.0, med("dt") - med("fwdbwd") - med("nan_chain")
                        - med("clip") - med("optimizer") - med("zero_grad"))
            labels = ["fwd/bwd", "NaN 链", "clip", "优化器·等待",
                      "优化器·host+计算", "zero_grad", "其他"]
            vals = [med("fwdbwd"), med("nan_chain"), med("clip"),
                    opt_wait, med("optimizer") - opt_wait, med("zero_grad"),
                    other]
            keep = [(l, v) for l, v in zip(labels, vals) if v > 0]
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.pie([v for _, v in keep],
                   labels=[f"{l}\n{v*1000:.0f}ms" for l, v in keep],
                   autopct="%1.0f%%", startangle=90)
            ax.set_title(f"step time pie — ws={ddp_world_size} db={db} "
                         f"GA={grad_accum} | MFU {mfu:.1f}%")
            fig.tight_layout()
            fig.savefig(png, dpi=150)
            print0(f"pie → {png}")
        except Exception as e:  # noqa: BLE001
            print0(f"(饼图跳过: {e!r})")

compute_cleanup()
