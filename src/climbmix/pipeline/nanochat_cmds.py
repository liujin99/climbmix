"""Shared nanochat subprocess helpers — the single source of truth for BOTH
execution backends:

  - the local executor (ProxyRunner, ThreadPool + local torchrun)
  - the remote worker (scripts/remote_worker.py, inside remote job
    containers, driven by an ExpSpec whose commands were built HERE on the
    submit host)

Constraint: this module MUST stay importable with ZERO climbmix/3rd-party
dependencies (stdlib only) — the remote worker imports it standalone from the
assets bundle shipped to job containers. Every behavior change here changes
remote+local execution identically (and enters the stage fingerprints).

Contents:
  - build_mid_train_cmd / build_eval_cmd: exact torchrun argv
  - make_eval_base_dir: the private-base-dir symlink farm that makes parallel
    eval CSVs collision-free (base_eval writes step-only CSV names)
  - claim_eval_csv: move THIS experiment's CSV out of the private dir
  - parse_eval_results: CSV -> per-task/stem metrics
"""

import os
import shutil
from typing import Dict, List, Optional, Tuple


def build_mid_train_cmd(
    model_tag: str,
    mixture_data_dir: str,
    device_type: str,
    num_iterations: int,
    lr_scale: float,
    warmup: float,
    warmdown: float,
    nproc_per_node: Optional[int] = None,
    master_port: Optional[int] = None,
    npu_devices: int = 8,
) -> List[str]:
    """torchrun mid_train argv for proxy experiments."""
    nproc = nproc_per_node or npu_devices
    cmd = [
        "torchrun", "--standalone",
        f"--nproc_per_node={nproc}",
    ]
    if master_port is not None:
        cmd += ["--master_port", str(master_port)]
    cmd += [
        "-m", "scripts.mid_train", "--",
        "--run", model_tag,
        "--device-type", device_type,
        "--model-tag", model_tag,
        "--num-iterations", str(num_iterations),
        "--lr-scale", str(lr_scale),
        "--warmup-ratio", str(warmup),
        "--warmdown-ratio", str(warmdown),
        # Base checkpoints save optimizer state as PER-RANK SHARDS
        # (optim_<step>_rank<r>.pt; 8-rank pretrain -> lm_head/wte
        # moments are [vocab/8, n_embd]). A proxy run may use a different
        # world size, and torch's load_state_dict does NOT shape-check
        # state tensors: the mismatched shard is silently assigned and
        # explodes at the first AdamW lerp_ (aclnnInplaceLerp EZ1001
        # "32768 and 4096 cannot broadcast", speedrun 2026-08-26).
        # Fresh optimizer state is also the CLIMB-correct semantics: proxy
        # experiments are short fine-tunes compared ACROSS mixtures, so
        # every candidate must get identical (cold) optimizer state.
        # LR inheritance is unaffected: lrs come from the pretrain meta
        # (user_config), and the batch_ratio LR adjustment inside the
        # load_optimizer block is a no-op here (proxy inherits
        # total_batch_size from the same checkpoint).
        "--load-optimizer", "0",
        # flat = 零裁剪文档打包 (DeepSeek V3 式)。与 target 阶段
        # (speedrun/run_climbmix Step 6) 及 quadmix STEM 实验保持同一
        # 口径 —— "proxy 分数预测 target 表现" 的前提是数据打包方式一致。
        # bos_bestfit 会裁掉 ~35% token, 且两阶段混用会使预测迁移失真。
        "--loader", "flat",
        # mid_train 默认 sample_every=500 会在 step 500 及 last_step 触发
        # Engine.generate_batch(), 打碎 NPU 内存 → optimizer.step() OOM
        # (quadmix af525ee 用崩溃换来的修复, 直接移植)。
        "--sample-every", "-1",
        # Disable the IN-TRAINING benchmark eval (--core-metric-every,
        # default 500, fires unconditionally at last_step). The external
        # base_eval right after training scores the same benchmarks
        # anyway; the in-training copy measured ~2h10m per experiment on
        # the speedrun — pure duplication. Val bpb (--eval-every) stays
        # on as the training signal.
        "--core-metric-every", "-1",
        "--data-dir", mixture_data_dir,
    ]
    return cmd


def build_eval_cmd(
    model_tag: str,
    device_type: str,
    eval_benchmarks: str,
    eval_max_per_task: int,
    nproc_per_node: Optional[int] = None,
    master_port: Optional[int] = None,
    npu_devices: int = 8,
) -> List[str]:
    """torchrun base_eval argv (--eval core; the private base dir is set via
    the NANOCHAT_BASE_DIR env, see build_subprocess_env).

    Subsample cap for cheap proxy evals: base_eval shuffles each task with a
    FIXED seed (random.Random(1337)) before truncating, so every experiment
    scores the SAME subset — scores stay comparable across candidate
    mixtures. -1 (default) = full eval sets (production); small caps (e.g.
    100) keep proxy evals cheap (speedrun).
    """
    nproc = nproc_per_node or npu_devices
    cmd = [
        "torchrun", "--standalone",
        f"--nproc_per_node={nproc}",
    ]
    if master_port is not None:
        cmd += ["--master_port", str(master_port)]
    cmd += [
        "-m", "scripts.base_eval", "--",
        "--eval", "core",
        "--eval-benchmarks", eval_benchmarks,
        "--model-tag", model_tag,
        "--model-type", "mid",
        "--device-type", device_type,
    ]
    if eval_max_per_task and eval_max_per_task > 0:
        cmd += ["--max-per-task", str(eval_max_per_task)]
    return cmd


def build_subprocess_env(
    nanochat_dir: str,
    nanochat_base_dir: str,
    device_ids: Optional[List[int]] = None,
    master_port: Optional[int] = None,
    base_dir_override: Optional[str] = None,
    base_env: Optional[Dict[str, str]] = None,
    extra_env: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Environment for mid_train/base_eval subprocesses (local AND remote).

    Mirrors the original ProxyRunner._run_subprocess semantics: PYTHONPATH +
    NANOCHAT_BASE_DIR (with optional private-base-dir override for eval),
    NPU device pinning, master port. base_env defaults to os.environ (local);
    the remote worker passes the container environment.
    """
    env = dict(base_env if base_env is not None else os.environ)
    env["PYTHONPATH"] = nanochat_dir + ":" + env.get("PYTHONPATH", "")
    env["NANOCHAT_BASE_DIR"] = base_dir_override or nanochat_base_dir
    if device_ids is not None:
        # ASCEND_RT_VISIBLE_DEVICES is the torch_npu-documented pinning var
        # (logical npu:k = k-th entry of the mask). ASCEND_VISIBLE_DEVICES
        # alone may be ignored by the runtime, which would pile every exp
        # of a parallel batch onto physical device 0. Set both (the
        # embedding workers pin with the RT var for the same reason).
        ids = ",".join(str(d) for d in device_ids)
        env["ASCEND_RT_VISIBLE_DEVICES"] = ids
        env["ASCEND_VISIBLE_DEVICES"] = ids
        env["RANK_SIZE"] = str(len(device_ids))
    if master_port is not None:
        env["MASTER_PORT"] = str(master_port)
    if extra_env:
        env.update(extra_env)
    return env


def make_eval_base_dir(
    nanochat_base_dir: str,
    exp_dir: str,
    model_tag: str,
    log=print,
) -> str:
    """Private NANOCHAT_BASE_DIR for one experiment's eval subprocess.

    base_eval.py writes its results CSV to
    {base_dir}/base_eval/mid_model_{step:06d}.csv — a step-only name with NO
    model tag — and every proxy experiment finishes at the same final step.
    With a shared base dir, two concurrent evals overwrite each other's CSV
    and the wrong scores get attributed silently. A private base dir makes
    the collision physically impossible and lets evals run fully in parallel
    (this exact logic runs inside remote job containers too).

    Everything the eval READS is symlinked to the real shared data; the two
    things it WRITES ({base_eval}/ CSV, {report}/) are private real dirs
    inside exp_dir.
    """
    eval_base = os.path.join(exp_dir, "_eval_base")
    # Rebuild from scratch on every eval attempt: a previously crashed
    # eval may have left a partial CSV or download in the private dirs.
    shutil.rmtree(eval_base, ignore_errors=True)

    mid_src = os.path.join(nanochat_base_dir, "mid_checkpoints", model_tag)
    if not os.path.isdir(mid_src):
        raise FileNotFoundError(
            f"mid checkpoint for eval not found: {mid_src} — refusing to "
            f"evaluate a missing/stale model")
    os.makedirs(os.path.join(eval_base, "mid_checkpoints"))
    os.symlink(mid_src, os.path.join(eval_base, "mid_checkpoints", model_tag))

    tok_src = os.path.join(nanochat_base_dir, "tokenizer")
    if not os.path.isdir(tok_src):
        raise FileNotFoundError(
            f"tokenizer not found at {tok_src} — eval cannot run")
    os.symlink(tok_src, os.path.join(eval_base, "tokenizer"))

    for name in ("eval_bundle", "eval_stem"):
        src = os.path.join(nanochat_base_dir, name)
        if os.path.isdir(src):
            os.symlink(src, os.path.join(eval_base, name))
        else:
            log(f"  [Eval] WARNING: {src} not found — eval will download "
                f"{name} into the private dir (slow)")

    os.makedirs(os.path.join(eval_base, "base_eval"), exist_ok=True)
    os.makedirs(os.path.join(eval_base, "report"), exist_ok=True)
    return eval_base


def claim_eval_csv(
    exp_dir: str,
    model_tag: str,
    eval_base: str,
    log=print,
) -> Optional[str]:
    """Move the CSV written by THIS eval from its private base dir into
    exp_dir. The private dir was rebuilt empty immediately before the eval
    subprocess started, so any mid_model_*.csv in it is unambiguously ours —
    no lock and no mtime heuristics needed. Also gives resume/debug a
    per-experiment record at eval_{model_tag}.csv.
    """
    csv_dir = os.path.join(eval_base, "base_eval")
    try:
        names = [f for f in os.listdir(csv_dir)
                 if f.startswith("mid_model_") and f.endswith(".csv")]
    except OSError:
        names = []
    if not names:
        log(f"  [Eval] WARNING: base_eval exited 0 but wrote no CSV in "
            f"{csv_dir} — scores for {model_tag} will be NaN")
        return None
    newest = max(names, key=lambda f: os.path.getmtime(os.path.join(csv_dir, f)))
    src = os.path.join(csv_dir, newest)
    dst = os.path.join(exp_dir, f"eval_{model_tag}.csv")
    shutil.move(src, dst)
    log(f"  [Eval] Claimed {newest} -> {dst}")
    return dst


def parse_eval_results(
    csv_path: Optional[str],
    val_tasks: List[str],
    log=print,
) -> Tuple[Optional[Dict[str, float]], float, Optional[float],
           Optional[Dict[str, float]], float]:
    """Parse the eval CSV into (per_task, val_accuracy, stem_metric,
    per_task_nlls, stem_nll). csv_path=None yields per_task=None — the search
    scores this experiment NaN."""
    per_task: Optional[Dict[str, float]] = None
    per_task_nlls: Optional[Dict[str, float]] = None
    val_accuracy: float = 0.0
    stem_metric: Optional[float] = None
    stem_nll: float = 0.0

    if csv_path is not None:
        per_task = {}
        per_task_nlls = {}
        with open(csv_path) as f:
            for line in f:
                parts = [p.strip() for p in line.strip().split(",")]
                if len(parts) >= 3:
                    task_name = parts[0]
                    centered_val = parts[2]
                    nll_val = parts[3] if len(parts) >= 4 else "0.0"
                    if task_name == "STEM":
                        try:
                            stem_metric = float(centered_val)
                        except ValueError:
                            pass
                        try:
                            stem_nll = float(nll_val)
                        except ValueError:
                            stem_nll = 0.0
                        continue
                    if task_name == "CORE":
                        continue
                    try:
                        per_task[task_name] = float(centered_val)
                    except ValueError:
                        continue
                    try:
                        per_task_nlls[task_name] = float(nll_val)
                    except ValueError:
                        per_task_nlls[task_name] = 0.0

    if stem_metric is not None:
        val_accuracy = stem_metric
    elif per_task:
        task_subset = [per_task[t] for t in val_tasks if t in per_task]
        if task_subset:
            val_accuracy = sum(task_subset) / len(task_subset)

    if stem_nll == 0.0 and per_task_nlls:
        nll_subset = [per_task_nlls[t] for t in val_tasks if t in per_task_nlls]
        if nll_subset:
            stem_nll = sum(nll_subset) / len(nll_subset)

    log(f"  [Eval] stem_metric={stem_metric}, val_accuracy={val_accuracy:.4f}, stem_nll={stem_nll:.4f}")
    return per_task, val_accuracy, stem_metric, per_task_nlls, stem_nll
