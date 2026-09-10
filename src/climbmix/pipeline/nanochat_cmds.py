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
  - build_target_mid_train_cmd / build_target_eval_cmd: the d28 target-arm
    argv (shell-Step-6/7 form; parity with runs/lib/target_arm.sh is
    test-asserted — dispatch_target_arm.py builds the REMOTE arm jobs
    with these so remote and local arms run the same tokens)
  - retarget_torchrun_multinode / resolve_multinode_rendezvous: the
    multi-node worker halves — node_rank/master_addr resolve INSIDE the
    job (platform env or k8s StatefulSet DNS), so the worker swaps the
    torchrun launcher prefix at runtime while the spec keeps pinning the
    training argv after "-m"
  - make_eval_base_dir: the private-base-dir symlink farm that makes parallel
    eval CSVs collision-free (base_eval writes step-only CSV names)
  - claim_eval_csv: move THIS experiment's CSV out of the private dir
  - parse_eval_results: CSV -> per-task/stem metrics
"""

import os
import re
import shutil
import socket
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


def build_target_mid_train_cmd(
    run_name: str,
    model_tag: str,
    data_dir: str,
    num_iterations,
    lr_scale,
    warmup,
    warmdown,
    core_metric_every,
    device_batch_size,
    loader,
    load_optimizer: Optional[str] = None,
    nproc_per_node: Optional[int] = None,
    master_port: Optional[int] = None,
    npu_devices: int = 8,
) -> List[str]:
    """torchrun mid_train argv for the TARGET arms (d28 climb/random).

    Token-for-token identical to runs/lib/target_arm.sh (the local
    fallback path) — parity is asserted by
    scripts/diagnostics/test_prod2_runtime.py. Differences vs the proxy
    build_mid_train_cmd are deliberate and match the shell Step-6 form
    proven in prod1:
      - `--key=value` token style (the shell's form)
      - no --load-optimizer by default: the single-node arm runs the SAME
        8-rank world size as the pretrain, so the d28 optimizer shards
        load with matching shapes (the proxy path must NOT load them: its
        world size varies). load_optimizer="0" (multi-node arms, ws=32)
        appends --load-optimizer=0 — the 8-shard moments CANNOT load at
        ws!=8 (AdamW reduce_scatter shape assert, run 601cdb67), so the
        optimizer state is cold; the SHELL fallback emits the same flag
        from TARGET_LOAD_OPTIMIZER=0 (both arms must share the semantics
        or a remote-cold vs local-warm arm pair is an unfair verdict).
      - no --device-type (the proven target argv does not pass it)
      - --eval-every=-1 explicit (val bpb eval off; the external
        base_eval after training is the scorer)
    Values are stringified verbatim (callers pass the launch env's raw
    strings so "1.0" never becomes "1.0" vs "1" drift between arms).
    """
    nproc = nproc_per_node or npu_devices
    cmd = [
        "torchrun", "--standalone",
        f"--nproc_per_node={nproc}",
    ]
    if master_port is not None:
        cmd += ["--master_port", str(master_port)]
    cmd += [
        "-m", "scripts.mid_train", "--",
        f"--num-iterations={num_iterations}",
        f"--lr-scale={lr_scale}",
        f"--warmup-ratio={warmup}",
        f"--warmdown-ratio={warmdown}",
        f"--core-metric-every={core_metric_every}",
        f"--device-batch-size={device_batch_size}",
        f"--loader={loader}",
        "--sample-every=-1",
        "--eval-every=-1",
    ]
    if load_optimizer is not None:
        cmd.append(f"--load-optimizer={load_optimizer}")
    cmd += [
        f"--run={run_name}",
        f"--model-tag={model_tag}",
        f"--data-dir={data_dir}",
    ]
    return cmd


def build_target_eval_cmd(
    model_tag: str,
    eval_benchmarks: str,
    eval_max_per_task,
    device_batch_size,
    core_batch_size,
    model_type: str = "mid",
    nproc_per_node: Optional[int] = None,
    master_port: Optional[int] = None,
    npu_devices: int = 8,
) -> List[str]:
    """torchrun base_eval argv for the target arms (--eval core).

    Parity target: runs/lib/target_arm.sh target_arm_eval. model_type
    stays parameterized so the remote base anchor can evaluate the raw
    d28 base checkpoint with the argv form prod1 used locally
    (--model-type=base; the worker points mid_checkpoints/{tag} at the
    d28 asset mount via spec.ckpt_src, and make_eval_base_dir links
    base_checkpoints/{tag} — load_model("base") reads that dir — from
    the container's mounted asset).
    --max-per-task is ALWAYS emitted (including -1 = full sets) to match
    the shell form.
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
        "--eval=core",
        f"--eval-benchmarks={eval_benchmarks}",
        f"--max-per-task={eval_max_per_task}",
        f"--device-batch-size={device_batch_size}",
        f"--core-eval-batch-size={core_batch_size}",
        f"--model-tag={model_tag}",
        f"--model-type={model_type}",
    ]
    return cmd


# ── multi-node target arms (Phase 1: 4-node ws=32 d28 arms) ───────────────
# Phase 0 (2026-09-09, job 440f760e) proved the shape: 4 nodes x 8 ranks,
# 6.1 s/step vs 18.2 s single-node, 34 optimizer ckpt shards across nodes.
# The spec's mid_train_cmd is built on the SUBMIT host, where node_rank and
# master_addr are unknowable — they resolve INSIDE the job (platform env or
# the k8s StatefulSet DNS ModelArts jobs run as). The two helpers below are
# the worker-side halves of that contract; both live here so the whole
# fleet (local executor, dispatch, worker) shares one implementation.


def retarget_torchrun_multinode(
    cmd: List[str],
    node_count: int,
    node_rank: int,
    master_addr: str,
    master_port: int,
    log=print,
) -> List[str]:
    """Rewrite a single-node torchrun argv into its multi-node form.

    Swaps ONLY the launcher prefix (everything before "-m"): the module
    tail — the training argv the spec pins — stays byte-identical. The
    resulting prefix mirrors the Phase-0-proven form
    (scripts/probe/probe_common.sh rdzv_torchrun_argv):
      --nnodes=N --node_rank=R --master_addr=A --master_port=P
      --nproc_per_node=M

    Strict on purpose: a launcher-shape drift must raise here, not
    silently mangle the training argv.
    """
    if not cmd or cmd[0] != "torchrun":
        raise ValueError(
            f"retarget: cmd[0] is not torchrun: {cmd[:2]!r}")
    try:
        m_idx = cmd.index("-m")
    except ValueError:
        raise ValueError(
            f"retarget: no -m module boundary in launcher prefix "
            f"{cmd[:8]!r}")
    nproc = None
    for tok in cmd[1:m_idx]:
        if tok.startswith("--nproc_per_node="):
            nproc = tok.split("=", 1)[1]
            break
    if nproc is None:
        raise ValueError(
            f"retarget: no --nproc_per_node=N token before -m in "
            f"{cmd[:m_idx]!r}")
    if not 0 <= node_rank < node_count:
        raise ValueError(
            f"retarget: node_rank {node_rank} outside [0, {node_count})")
    prefix = [
        "torchrun",
        f"--nnodes={node_count}",
        f"--node_rank={node_rank}",
        f"--master_addr={master_addr}",
        f"--master_port={master_port}",
        f"--nproc_per_node={nproc}",
    ]
    log(f"[worker] torchrun prefix -> {' '.join(prefix[1:])}")
    return prefix + cmd[m_idx:]


def _hosts_file_lookup(name: str, hosts_file: str) -> Optional[str]:
    """Resolve a name to an IP from a hosts file (first match, comments
    skipped). Mirrors probe_common.sh rdzv_dns_lookup's awk: the name must
    appear as the 2nd or 3rd field."""
    try:
        with open(hosts_file) as f:
            for line in f:
                parts = line.split()
                if not parts or parts[0].startswith("#"):
                    continue
                if len(parts) >= 2 and name in parts[1:3]:
                    return parts[0]
    except OSError:
        pass
    return None


def resolve_multinode_rendezvous(
    node_count: int,
    master_port: int = 29500,
    env: Optional[Dict[str, str]] = None,
    hostname: Optional[str] = None,
    hosts_file: str = "/etc/hosts",
    log=print,
) -> Tuple[str, int, int]:
    """Resolve (master_addr, master_port, node_rank) inside a multi-node
    job container.

    Priority — identical to the Phase-0 probe chain
    (scripts/probe/probe_common.sh rdzv_resolve auto mode):
      1. platform env: MASTER_ADDR + MASTER_PORT + NODE_RANK (torch-elastic
         style), then the MA_/VC_ prefixed variants;
      2. k8s StatefulSet DNS: hostname is <svc>-worker-N and /etc/hosts
         carries the pod FQDN <host>.<svc>.<ns>.svc.cluster.local —
         master = <svc>-worker-0.<domain> (resolved to an IP via the hosts
         file, then DNS), rank = own worker number. ModelArts runs its
         multi-node jobs as exactly this StatefulSet (probe A verified).
    Anything else raises: a wrong-but-plausible triple hangs the whole
    job at the first HCCL collective (EI0015), so ambiguity must fail
    FAST and VISIBLY instead.
    """
    e = env if env is not None else os.environ
    for p in ("", "MA_", "VC_"):
        addr = e.get(p + "MASTER_ADDR", "")
        port = e.get(p + "MASTER_PORT", "")
        rank = e.get(p + "NODE_RANK", "")
        if addr and port and rank:
            r = int(rank)
            if not 0 <= r < node_count:
                raise RuntimeError(
                    f"rdzv: platform NODE_RANK={r} outside "
                    f"[0, {node_count})")
            log(f"[worker] rdzv: platform env ({p or 'plain'}MASTER_*) "
                f"-> {addr}:{port} rank={r}")
            return addr, int(port), r

    host = hostname or socket.gethostname()
    m = re.match(r"^(.*)-worker-([0-9]+)$", host)
    if not m:
        raise RuntimeError(
            f"rdzv: no platform MASTER_* env and hostname {host!r} is not "
            f"<svc>-worker-N — cannot derive this node's rank; refusing to "
            f"guess (a wrong triple hangs every rank at the first HCCL "
            f"collective)")
    svc, rank = m.group(1), int(m.group(2))
    if not 0 <= rank < node_count:
        raise RuntimeError(
            f"rdzv: sts-dns rank {rank} (from {host!r}) outside "
            f"[0, {node_count})")
    # pod FQDN: the hosts-file line whose 2nd (NF==2) or 3rd (NF>=3) field
    # is our short hostname — same awk as probe_common.sh rdzv_try_sts_dns
    fqdn = ""
    try:
        with open(hosts_file) as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 3 and parts[2] == host:
                    fqdn = parts[1]
                    break
                if len(parts) == 2 and parts[1] == host:
                    fqdn = parts[1]
                    break
    except OSError:
        pass
    if not fqdn or "." not in fqdn:
        raise RuntimeError(
            f"rdzv: sts-dns: no pod FQDN for {host!r} in {hosts_file}")
    domain = fqdn.split(".", 1)[1]
    master_name = f"{svc}-worker-0.{domain}"
    ip = _hosts_file_lookup(master_name, hosts_file)
    if ip is None:
        try:
            ip = socket.gethostbyname(master_name)
        except OSError as e:
            raise RuntimeError(
                f"rdzv: sts-dns: cannot resolve master {master_name}: {e}")
    log(f"[worker] rdzv: sts-dns -> master {master_name} ({ip}):"
        f"{master_port} rank={rank}")
    return ip, master_port, rank


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
    # print0 has no flush=True; with stdout redirected to a log file the
    # 8KB block buffer delays disk content by ~35 step lines, so streamed
    # copies of mid_train.log end mid-run and lose the tail on hard kills.
    env["PYTHONUNBUFFERED"] = "1"
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
    subdir: str = "_eval_base",
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

    subdir: the private dir's name under exp_dir. Multi-node workers pass
    a per-node suffix (_eval_base_node{r}) — exp_dir can sit on the
    cross-node shared output mount, and this function rmtree's + rebuilds
    its target, so concurrent nodes must not share one path.
    """
    eval_base = os.path.join(exp_dir, subdir)
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

    # --model-type=base (the remote d28 anchor) loads
    # {base_dir}/base_checkpoints/{tag} — nanochat's load_model maps
    # source "base" to that dir — so the private dir needs the link too
    # (2026-09-09: the remote anchor exited 1 because only mid_checkpoints
    # was linked; model-type=mid evals never read it and never noticed).
    # Conditional: search exps (model-type=mid) have no such dir.
    base_src = os.path.join(nanochat_base_dir, "base_checkpoints", model_tag)
    if os.path.isdir(base_src):
        os.makedirs(os.path.join(eval_base, "base_checkpoints"))
        os.symlink(base_src, os.path.join(eval_base, "base_checkpoints",
                                          model_tag))

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
    eval_rc: Optional[int] = None,
    log=print,
) -> Optional[str]:
    """Move the CSV written by THIS eval from its private base dir into
    exp_dir. The private dir was rebuilt empty immediately before the eval
    subprocess started, so any mid_model_*.csv in it is unambiguously ours —
    no lock and no mtime heuristics needed. Also gives resume/debug a
    per-experiment record at eval_{model_tag}.csv. base_model_*.csv
    (base-model evals, e.g. the remote d28 anchor) is claimed the same way.
    """
    csv_dir = os.path.join(eval_base, "base_eval")
    try:
        names = [f for f in os.listdir(csv_dir)
                 if ((f.startswith("mid_model_") or f.startswith("base_model_"))
                     and f.endswith(".csv"))]
    except OSError:
        names = []
    if not names:
        # rc included when known: 2026-09-09 the canned "exited 0" text
        # contradicted the FAILED (exit code 1) line above it and sent
        # the diagnosis down the wrong path.
        rc_txt = f" (eval rc={eval_rc})" if eval_rc is not None else ""
        log(f"  [Eval] WARNING: base_eval wrote no CSV{rc_txt} in "
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
