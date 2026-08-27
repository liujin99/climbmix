#!/usr/bin/env python3
# ═══════════════════════════════════════════════════════════════════════
#  check_ckpt_dtype.py — ckpt 张量 dtype 实证检查 + 静态显存预算推算
#
#  背景 (d28 Step-6 OOM 调查): 代码读出 nanochat-npu 的设计是
#  "bf16 计算 + fp32 主权重存储" (gpt.py Linear 注释; 无 set_default_dtype;
#  init_weights 只把 embeddings cast 成 bf16)。但 ckpt 是历史训练产物,
#  文件内部实际 dtype 未验证过 —— 这决定生产升 dbs>=2 的显存预算:
#    fp32 主权重: 静态(参数+梯度+优化器分片) ~17G → dbs=2 大概率靠换页
#    bf16 主权重: 静态 ~11G → dbs=2 纯 HBM 可行, 且快一倍
#
#  输出: 每个 tag 的分类 dtype/字节表 + ws=8 下每卡静态驻留估计 +
#  激活预算 + H1(fp32)/H2(bf16) 判定 + meta 是否记录 dtype 的确认。
#
#  用法 (服务器):
#    python3 scripts/diagnostics/check_ckpt_dtype.py            # 扫全部 tag
#    python3 scripts/diagnostics/check_ckpt_dtype.py --tags d28_speedrun d28
#    python3 scripts/diagnostics/check_ckpt_dtype.py --skip-optim   # 只看模型
#
#  mmap 优先 (秒级, 不真读入内存); 旧 torch 自动退回普通 load。
# ═══════════════════════════════════════════════════════════════════════
import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import torch

HBM_GIB = 29.49          # 910B4 单卡
CANN_CTX_GIB = 1.0       # CANN/HCCL 上下文粗估
WALL_GIB = 27.5          # 2026-08-27 dbs=4/dbs=8 观测的 OOM 水位 (allocated)


def load_lazy(path):
    try:
        return torch.load(path, map_location="cpu", mmap=True), True
    except Exception:
        return torch.load(path, map_location="cpu"), False


def walk_tensors(obj, prefix=""):
    if torch.is_tensor(obj):
        yield prefix, obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from walk_tensors(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from walk_tensors(v, f"{prefix}[{i}]")


def categorize(key):
    if key == "transformer.wte.weight":
        return "wte (embedding)"
    if key.startswith("value_embeds."):
        return "value_embeds (embedding)"
    if key == "lm_head.weight":
        return "lm_head (unembedding)"
    if key.startswith("transformer.h."):
        return "transformer matrices"
    return "scalars/lambdas"


def fmt_gib(n):
    return f"{n / 1024**3:.2f} GiB"


def check_tag(tag_dir, skip_optim=False):
    print(f"\n{'═' * 64}\n  {os.path.basename(tag_dir)}  ({tag_dir})\n{'═' * 64}")

    # ── meta: 是否记录 dtype ──
    meta_files = sorted(glob.glob(os.path.join(tag_dir, "meta_*.json")))
    dtype_hits = []
    for mf in meta_files:
        try:
            with open(mf) as f:
                meta = json.load(f)
            for k, v in meta.items():
                if "dtype" in k.lower():
                    dtype_hits.append(f"{os.path.basename(mf)}: {k}={v}")
        except Exception as e:
            print(f"  (meta 读取失败 {mf}: {e})")
    if dtype_hits:
        print("  meta dtype 记录:", "; ".join(dtype_hits))
    else:
        print("  meta: 无任何 *dtype* 键 — 证实 meta 不记录精度 (dtype 来自 NANOCHAT_DTYPE env)")

    # ── 模型文件 ──
    model_files = sorted(glob.glob(os.path.join(tag_dir, "model_*.pt")))
    if not model_files:
        print("  (无 model_*.pt, 跳过)")
        return
    mf = model_files[-1]
    data, mmaped = load_lazy(mf)
    data = {k.removeprefix("_orig_mod."): v for k, v in data.items()
            if torch.is_tensor(v)}
    print(f"  模型: {os.path.basename(mf)}  (mmap={mmaped})")

    stats = defaultdict(lambda: [0, 0])  # (category, dtype) -> [count, bytes]
    for key, t in data.items():
        cat = categorize(key)
        s = stats[(cat, str(t.dtype).replace("torch.", ""))]
        s[0] += 1
        s[1] += t.numel() * t.element_size()

    print(f"\n  {'类别':<26} {'dtype':<10} {'张量数':>6} {'大小':>12}")
    params_bytes = 0
    matrix_dtypes = set()
    for (cat, dtype), (n, b) in sorted(stats.items()):
        print(f"  {cat:<26} {dtype:<10} {n:>6} {fmt_gib(b):>12}")
        params_bytes += b
        if cat == "transformer matrices":
            matrix_dtypes.add(dtype)

    # ── 优化器分片 (rank0 代表; ws=8 时每卡只持有 1/8) ──
    optim_bytes = 0
    optim_dtypes = set()
    if not skip_optim:
        opt_files = sorted(glob.glob(os.path.join(tag_dir, "optim_*_rank0.pt")))
        if opt_files:
            of = opt_files[-1]
            odata, _ = load_lazy(of)
            for _key, t in walk_tensors(odata):
                optim_bytes += t.numel() * t.element_size()
                optim_dtypes.add(str(t.dtype).replace("torch.", ""))
            print(f"\n  优化器: {os.path.basename(of)}  dtypes={sorted(optim_dtypes)}"
                  f"  rank0={fmt_gib(optim_bytes)}")
        else:
            print("\n  优化器: 无 optim_*_rank0.pt")
    else:
        print("\n  优化器: --skip-optim")

    # ── 静态预算推算 (ws=8 每卡) ──
    grads_bytes = params_bytes          # 梯度 dtype 跟参数 (fp32 主权重设计)
    static = params_bytes + grads_bytes + optim_bytes  # 优化器已按 rank 分片
    act_budget = HBM_GIB - CANN_CTX_GIB - static / 1024**3
    observed_act = WALL_GIB - static / 1024**3  # 反推 dbs=4 撞墙时激活占量

    print(f"\n  ── ws=8 每卡静态驻留估计 ──")
    print(f"    参数 {fmt_gib(params_bytes)} + 梯度 {fmt_gib(grads_bytes)}"
          f" + 优化器分片 {fmt_gib(optim_bytes)} = {fmt_gib(static)}")
    print(f"    HBM {HBM_GIB} - CANN ~{CANN_CTX_GIB} - 静态 → 激活预算 ≈ {act_budget:.2f} GiB")
    if observed_act > 0:
        print(f"    反推 dbs=4 撞墙时激活 ≈ {observed_act:.2f} GiB"
              f" (观测 {WALL_GIB} - 静态)")
        if observed_act > act_budget:
            print(f"    → dbs=4 需 {observed_act:.1f}G > 预算 {act_budget:.1f}G,"
                  f" 与实测 OOM 自洽")

    # ── H1/H2 判定 + dbs 投影 ──
    print(f"\n  ── 判定 ──")
    if matrix_dtypes == {"float32"}:
        print("    ✓ H1 证实: 矩阵主权重为 fp32 (代码设计一致)")
    elif matrix_dtypes == {"bfloat16"}:
        print("    ✓ H2 证实: 矩阵主权重为 bf16 (与当前代码设计不同, 历史产物)")
    else:
        print(f"    ? 矩阵 dtype 混合: {matrix_dtypes} — 需人工解读")
    if observed_act > 0:
        # 标定: 2026-08-27 dbs=4 run 在 allocated≈27.5G 处撞墙 (差 66 MiB),
        # 即该次激活需求 ≈ observed_act。假设激活随 dbs 线性 → 每份 ≈ /4。
        per_dbs = observed_act / 4.0
        static_gib = static / 1024**3
        print(f"    dbs 投影 (峰值 allocated ≈ 静态 {static_gib:.1f}G + "
              f"{per_dbs:.2f}G × dbs; 实测墙 ≈ {WALL_GIB}G):")
        for dbs in (1, 2, 4):
            proj = static_gib + per_dbs * dbs
            if dbs == 4:
                note = "← 标定点 (实测撞墙, 差 66 MiB)"
            elif proj < WALL_GIB - 1.5:
                note = "纯 HBM 内, 留碎片余量 ✓"
            else:
                note = "贴近墙, 需 unified memory 换页"
            print(f"      dbs={dbs}: ≈ {proj:.1f} GiB   {note}")
        print("    (以 speedrun dbs=1 实测 'Peak memory usage' 作第二标定点复核)")


def main():
    p = argparse.ArgumentParser(description="ckpt dtype 实证检查 + 显存预算推算")
    p.add_argument("--base-dir",
                   default="/home/ma-user/work/nanochat_model_dir/base_checkpoints")
    p.add_argument("--tags", nargs="*", default=None,
                   help="只检查这些 tag (默认全部含 model_*.pt 的子目录)")
    p.add_argument("--skip-optim", action="store_true",
                   help="跳过优化器分片 (更快)")
    args = p.parse_args()

    if not os.path.isdir(args.base_dir):
        sys.exit(f"✗ 目录不存在: {args.base_dir}")

    tags = args.tags or sorted(
        d for d in os.listdir(args.base_dir)
        if glob.glob(os.path.join(args.base_dir, d, "model_*.pt"))
    )
    if not tags:
        sys.exit(f"✗ {args.base_dir} 下没有任何含 model_*.pt 的子目录")

    print(f"═══ ckpt dtype 检查: {args.base_dir} ═══")
    print(f"    tags: {', '.join(tags)}")
    for tag in tags:
        check_tag(os.path.join(args.base_dir, tag), skip_optim=args.skip_optim)

    print(f"\n{'═' * 64}")
    print("  完成。把每个 tag 的判定行和静态估计带回即可定生产 dbs。")


if __name__ == "__main__":
    main()
