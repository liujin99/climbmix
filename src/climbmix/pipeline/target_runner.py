"""
Target runner for CLIMB — trains the final target model (e.g. d28) using
the optimal mixture weights found during proxy search.

Steps:
  1. Select STEM docs by optimal mixture weights
  2. Mix 70% STEM + 30% ClimbMix general data (adaptive 3-50 shards, reverse download)
  3. Call nanochat mid_train.py (d28 annealing from base checkpoint)
  4. Call nanochat base_eval.py (STEM benchmark evaluation)
  5. Parse evaluation results

Key design:
  - Uses optimal mixture weights from pipeline Stage 4
  - Same 70/30 STEM + ClimbMix mixing as proxy experiments
  - General data cached and reused from proxy phase
  - Evaluation: --eval-benchmarks=stem → stem_metric
"""

import os
import sys
import json
import shutil
import importlib
import subprocess
import time
import numpy as np
import numpy.typing as npt
from typing import Dict, List, Optional, Any, Tuple

from climbmix.core.types import CLIMBConfig, MixtureWeights
from climbmix.sampling.data_selector import select_data_by_mixture


class TargetRunner:

    def __init__(
        self,
        config: CLIMBConfig,
        cluster_labels: Optional[npt.NDArray[np.int64]] = None,
        token_counts: Optional[npt.NDArray[np.int64]] = None,
        metadata_manager: Optional[Any] = None,
    ):
        self.config = config
        self.nanochat_dir = config.nanochat_dir
        self.nanochat_base_dir = config.nanochat_base_dir
        self.target_depth = config.target.depth
        self.target_num_iterations = config.target.training_iterations
        self.target_lr_scale = config.target.lr_scale
        self.target_warmup = config.target.warmup
        self.target_warmdown = config.target.warmdown
        self.target_phase1_ckpt = config.target.phase1_checkpoint_path
        self.val_tasks = config.val_tasks
        self.device_type = config.device.device_type
        self.npu_devices = config.device.npu_devices
        self.general_data_dir = config.general_data_dir
        self.stem_ratio = config.stem_ratio
        self.eval_benchmarks = config.eval_benchmarks

        self.cluster_labels = cluster_labels
        self.token_counts = token_counts
        self.metadata_manager = metadata_manager

    def run(
        self,
        optimal_weights: MixtureWeights,
        selected_indices: npt.NDArray[np.int64],
        output_dir: str,
    ) -> Dict[str, Any]:
        """Run target training with optimal mixture.

        Args:
            optimal_weights: Optimal mixture weights from pipeline.
            selected_indices: Pre-selected doc indices from pipeline Stage 4.
            output_dir: Output directory for this run.

        Returns:
            Dict with training and evaluation results.
        """
        target_dir = os.path.join(output_dir, "target_run")
        os.makedirs(target_dir, exist_ok=True)

        model_tag = f"climbmix_target_d{self.target_depth}"

        self._symlink_base_checkpoint(model_tag)

        mixture_data_dir = os.path.join(target_dir, "mixture_data")

        t_start = time.time()
        print(f"\n  [Target] Starting target training (d{self.target_depth}, tag={model_tag})")

        print(f"  [Target] Preparing mixture data "
              f"({self.stem_ratio*100:.0f}% STEM + {(1-self.stem_ratio)*100:.0f}% general)...")
        self._prepare_mixture_data(selected_indices, mixture_data_dir)

        mid_cmd = self._build_mid_train_cmd(model_tag, mixture_data_dir)
        print(f"  [Target] mid_train: {' '.join(mid_cmd)}")
        mid_rc = self._run_subprocess(mid_cmd, target_dir, "mid_train")
        if mid_rc != 0:
            # Fail-fast: target training is the single most expensive stage;
            # a silent rc!=0 here would leave the run "completing" without a
            # trained model. Established semantics: target failure kills the
            # run loudly (resume restarts the whole target stage).
            raise RuntimeError(
                f"target mid_train failed (rc={mid_rc}), "
                f"see {os.path.join(target_dir, 'mid_train.log')}")

        eval_cmd = self._build_eval_cmd(model_tag)
        print(f"  [Target] base_eval: {' '.join(eval_cmd)}")
        eval_start_time = time.time()
        eval_rc = self._run_subprocess(eval_cmd, target_dir, "eval")
        eval_end_time = time.time()
        if eval_rc != 0:
            raise RuntimeError(
                f"target base_eval failed (rc={eval_rc}), "
                f"see {os.path.join(target_dir, 'eval.log')}")

        self._copy_mid_checkpoint(model_tag, target_dir)
        per_task, stem_metric, per_task_nlls, stem_nll = \
            self._parse_eval_results(model_tag, target_dir,
                                     eval_start=eval_start_time,
                                     eval_end=eval_end_time)

        elapsed = time.time() - t_start

        result = {
            "model_tag": model_tag,
            "target_depth": self.target_depth,
            "target_scaling_M": self.config.target.scaling_M,
            "target_num_iterations": self.target_num_iterations,
            "lr_scale": self.target_lr_scale,
            "warmup": self.target_warmup,
            "warmdown": self.target_warmdown,
            "stem_ratio": self.stem_ratio,
            "eval_benchmarks": self.eval_benchmarks,
            "elapsed_seconds": elapsed,
            "mid_train_rc": mid_rc,
            "eval_rc": eval_rc,
        }
        if per_task is not None:
            result["per_task_accuracies"] = per_task
        if stem_metric is not None:
            result["stem_metric"] = stem_metric
        if per_task_nlls is not None:
            result["per_task_nlls"] = per_task_nlls
            result["stem_nll"] = stem_nll

        result_path = os.path.join(target_dir, "target_result.json")
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)

        print(f"  [Target] Done in {elapsed:.1f}s, stem_metric={stem_metric}\n")

        return result

    def _symlink_base_checkpoint(self, model_tag: str):
        base_dir = self.nanochat_base_dir
        base_src = self.target_phase1_ckpt or os.path.join(base_dir, "base_checkpoints", f"d{self.target_depth}")
        base_dst = os.path.join(base_dir, "base_checkpoints", model_tag)

        if not os.path.exists(base_dst):
            if os.path.isdir(base_src):
                os.symlink(base_src, base_dst)
                print(f"  [Symlink] {base_dst} -> {base_src}")
            else:
                print(f"  [WARNING] Base checkpoint not found: {base_src}")

    def _copy_mid_checkpoint(self, model_tag: str, target_dir: str):
        mid_src_dir = os.path.join(self.nanochat_base_dir, "mid_checkpoints", model_tag)
        mid_dst_dir = os.path.join(target_dir, "mid_checkpoint")

        if os.path.isdir(mid_src_dir):
            if os.path.exists(mid_dst_dir):
                shutil.rmtree(mid_dst_dir)
            shutil.copytree(mid_src_dir, mid_dst_dir)
            print(f"  [Copy] mid checkpoint -> {mid_dst_dir}")

    def _load_mix_module(self):
        """Load scripts/mix_general_data.py as a module (reuses download + mix logic)."""
        scripts_dir = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "scripts")
        )
        script_path = os.path.join(scripts_dir, "mix_general_data.py")
        if not os.path.isfile(script_path):
            raise FileNotFoundError(f"mix_general_data.py not found at {script_path}")
        if self.nanochat_dir not in sys.path:
            sys.path.insert(0, self.nanochat_dir)
        spec = importlib.util.spec_from_file_location("mix_general_data", script_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def _prepare_mixture_data(
        self,
        selected_indices: npt.NDArray[np.int64],
        mixture_data_dir: str,
    ):
        if self.metadata_manager is None:
            print("  [Target] WARNING: No metadata_manager, cannot prepare mixture data")
            return

        os.makedirs(mixture_data_dir, exist_ok=True)

        stem_texts = self.metadata_manager.read_texts(selected_indices)
        print(f"  [Target] Selected {len(stem_texts)} STEM docs by optimal weights")

        # Write STEM texts to temp parquet shards (mix_general_data expects shard_*.parquet).
        # Convention (same as prepare_shards.py / mix_general_data.py): the LAST
        # shard_*.parquet is the val split; train shards must NOT contain val docs.
        stem_temp_dir = os.path.join(mixture_data_dir, "_stem_temp")
        os.makedirs(stem_temp_dir, exist_ok=True)

        import pyarrow as pa
        import pyarrow.parquet as pq

        nproc = self.npu_devices
        batch_per_file = 10000
        n_stem = len(stem_texts)

        # Hold out a real val split (tail of the data): ~1% of docs, capped at 256,
        # at least 2 docs per NPU so every rank owns >= 2 val row groups (rg_size=1).
        n_val = min(256, max(2 * nproc, n_stem // 100))
        if n_stem >= 2:
            n_val = max(1, min(n_val, n_stem - 1))
        else:
            n_val = min(n_val, n_stem)
        train_texts = stem_texts[:n_stem - n_val]
        val_texts = stem_texts[n_stem - n_val:]

        n_train = len(train_texts)
        if n_train < 2 * nproc:
            raise ValueError(
                f"Target run: only {n_train} train docs < 2*nproc ({2 * nproc}). "
                f"The DDP dataloader assigns row groups round-robin per rank; some ranks "
                f"would starve and hang before the first all_reduce. "
                f"Raise --target-tokens."
            )
        n_shards = max(1, (n_train + batch_per_file - 1) // batch_per_file)
        # Absorb a tiny remainder into the previous shard: a last shard with
        # < 2*nproc docs cannot provide one row group per rank -> DDP starvation.
        while n_shards > 1 and n_train - (n_shards - 1) * batch_per_file < 2 * nproc:
            n_shards -= 1
        # Row groups sized from the ACTUAL doc count of the last (smallest) shard,
        # guaranteeing every shard has >= nproc*2 row groups (DDP round-robin safety).
        last_shard_docs = n_train - (n_shards - 1) * batch_per_file
        rg_size = max(1, last_shard_docs // (nproc * 2))

        for i in range(n_shards):
            start = i * batch_per_file
            end = min(start + batch_per_file, n_train)
            shard_table = pa.table({"text": train_texts[start:end]})
            shard_path = os.path.join(stem_temp_dir, f"shard_{i:05d}.parquet")
            pq.write_table(shard_table, shard_path, row_group_size=rg_size)

        val_path = os.path.join(stem_temp_dir, f"shard_{n_shards:05d}.parquet")
        pq.write_table(pa.table({"text": val_texts}), val_path, row_group_size=1)

        print(f"  [Target] Wrote {n_train} train docs ({n_shards} shards, "
              f"rg_size={rg_size}) + {n_val} val docs")

        # Mix with ClimbMix general data via mix_general_data.py module
        if self.stem_ratio < 1.0 and self.general_data_dir:
            try:
                mix_mod = self._load_mix_module()
            except FileNotFoundError as e:
                raise FileNotFoundError(
                    f"Target run requires stem_ratio={self.stem_ratio} mixing, "
                    f"but mix_general_data.py could not be loaded: {e}"
                ) from e

            stem_train_files = sorted(
                os.path.join(stem_temp_dir, f)
                for f in os.listdir(stem_temp_dir)
                if f.startswith("shard_") and f.endswith(".parquet")
            )[:-1]

            stem_docs = mix_mod.count_stem_docs(stem_train_files)
            needed_shards = mix_mod.calc_climbmix_count(stem_docs, self.stem_ratio)

            print(f"  [Target] STEM: {stem_docs:,} docs -> need {needed_shards} ClimbMix shards "
                  f"({self.stem_ratio*100:.0f}% STEM + {(1-self.stem_ratio)*100:.0f}% general)")

            climb_files = mix_mod.download_climbmix(
                self.general_data_dir, needed_shards, self.npu_devices
            )

            detected_batch = mix_mod.detect_shard_size(stem_train_files)
            num_output_files = len(stem_train_files)
            mix_mod.mix_data(
                stem_temp_dir, climb_files, mixture_data_dir,
                num_output_files, detected_batch, num_npu=nproc,
                stem_ratio=self.stem_ratio,
            )
            print(f"  [Target] Mixed {stem_docs:,} STEM + "
                  f"~{int(stem_docs * (1-self.stem_ratio) / self.stem_ratio):,} general")
        else:
            self._copy_stem_only(stem_temp_dir, mixture_data_dir)
            print(f"  [Target] No general data, using {n_stem} STEM docs only")

        shutil.rmtree(stem_temp_dir, ignore_errors=True)
        print(f"  [Target] Data ready at {mixture_data_dir}")

    @staticmethod
    def _copy_stem_only(stem_temp_dir: str, mixture_data_dir: str):
        for f in os.listdir(stem_temp_dir):
            shutil.copy2(
                os.path.join(stem_temp_dir, f),
                os.path.join(mixture_data_dir, f),
            )

    def _build_mid_train_cmd(
        self,
        model_tag: str,
        mixture_data_dir: str,
    ) -> List[str]:
        cmd = [
            "torchrun", "--standalone",
            f"--nproc_per_node={self.npu_devices}",
            "-m", "scripts.mid_train", "--",
            "--run", model_tag,
            "--device-type", self.device_type,
            "--model-tag", model_tag,
            "--num-iterations", str(self.target_num_iterations),
            "--lr-scale", str(self.target_lr_scale),
            "--warmup-ratio", str(self.target_warmup),
            "--warmdown-ratio", str(self.target_warmdown),
            # d28 mid_train 实证安全配置, 与 speedrun/run_climbmix Step 6 及
            # proxy_runner 同口径:
            #   dbs=1 — d28 fp32 主权重+梯度 ~16G 静态 (ws=8), 不传则继承
            #   预训练 dbs (32), 第一个 forward 即撞 ~27.5G 墙; dbs>=4 实测
            #   必 OOM。dbs 只切 micro-batch, total batch 不变。
            "--device-batch-size", "1",
            # flat = 零裁剪打包, 与 proxy 阶段同口径 (proxy 分数预测 target
            # 表现的前提); bos_bestfit 会裁掉 ~35% token。
            "--loader", "flat",
            # 默认 sample_every=500 在 last_step 触发 generate_batch(),
            # 打碎 NPU 内存 → optimizer.step() OOM (quadmix af525ee)。
            "--sample-every", "-1",
            # 训练内 benchmark eval 与紧随其后的外部 base_eval 重复
            # (speedrun 实测 ~2h10m/次), 关掉。
            "--core-metric-every", "-1",
            # 默认 eval_every=100 在 step 0 必跑 1280 个 val forward,
            # 碎片化 allocator → optimizer Phase-1 的 2G stacked_grads
            # 连续分配失败 (2026-08-28 dbs=1 实测 22.24G alloc OOM;
            # quadmix 显式 -1, 同堆叠路径 390 步无 OOM)。d28 峰值
            # ~24.2G vs 天花板 ~24.5G, 余量 <0.3G, allocator 必须干净。
            "--eval-every", "-1",
            "--data-dir", mixture_data_dir,
        ]
        return cmd

    def _build_eval_cmd(
        self,
        model_tag: str,
    ) -> List[str]:
        cmd = [
            "torchrun", "--standalone",
            f"--nproc_per_node={self.npu_devices}",
            "-m", "scripts.base_eval", "--",
            "--eval", "core",
            "--eval-benchmarks", self.eval_benchmarks,
            "--model-tag", model_tag,
            "--model-type", "mid",
            "--device-type", self.device_type,
        ]
        return cmd

    def _run_subprocess(
        self,
        cmd: List[str],
        run_dir: str,
        stage_name: str,
    ) -> int:
        log_path = os.path.join(run_dir, f"{stage_name}.log")
        env = os.environ.copy()
        env["PYTHONPATH"] = self.nanochat_dir + ":" + env.get("PYTHONPATH", "")
        env["NANOCHAT_BASE_DIR"] = self.nanochat_base_dir

        with open(log_path, "w") as log_f:
            proc = subprocess.Popen(
                cmd,
                cwd=self.nanochat_dir,
                env=env,
                stdout=log_f,
                stderr=subprocess.STDOUT,
            )
            proc.wait()

        if proc.returncode != 0:
            print(f"  [{stage_name}] FAILED (exit code {proc.returncode}), see {log_path}")
        else:
            print(f"  [{stage_name}] Completed (exit code 0)")

        return proc.returncode

    @staticmethod
    def _locate_eval_csv(
        csv_dir: str,
        model_tag: str,
        eval_start: Optional[float] = None,
        eval_end: Optional[float] = None,
    ) -> Optional[str]:
        """Locate the base_eval CSV produced by this run (3-tier strategy).

        1. mid_model_* files whose filename contains model_tag (newest by mtime)
        2. mid_model_* files modified within our eval window [start-60s, end+60s]
           (newest by mtime)
        3. globally newest mid_model_* file (last-resort fallback)

        Only considers nanochat's own ``mid_model_*`` outputs, so archive copies
        named ``{model_tag}.csv`` never shadow a fresh result.
        """
        try:
            names = [f for f in os.listdir(csv_dir)
                     if f.endswith(".csv") and f.startswith("mid_model_")]
        except OSError:
            return None
        if not names:
            return None

        paths = [os.path.join(csv_dir, f) for f in names]
        mtimes = {p: os.path.getmtime(p) for p in paths}

        tagged = [p for p in paths if model_tag in os.path.basename(p)]
        if tagged:
            return max(tagged, key=lambda p: mtimes[p])

        if eval_start is not None:
            slack = 60.0
            end = eval_end if eval_end is not None else float("inf")
            window = [p for p in paths
                      if mtimes[p] >= eval_start - slack and mtimes[p] <= end + slack]
            if window:
                return max(window, key=lambda p: mtimes[p])

        return max(paths, key=lambda p: mtimes[p])

    def _parse_eval_results(
        self,
        model_tag: str,
        run_dir: str,
        eval_start: Optional[float] = None,
        eval_end: Optional[float] = None,
    ) -> Tuple[Optional[Dict[str, float]], Optional[float], Optional[Dict[str, float]], float]:
        per_task: Optional[Dict[str, float]] = None
        stem_metric: Optional[float] = None
        per_task_nlls: Optional[Dict[str, float]] = None
        stem_nll: float = 0.0

        csv_dir = os.path.join(self.nanochat_base_dir, "base_eval")

        csv_path = self._locate_eval_csv(csv_dir, model_tag, eval_start, eval_end)
        if csv_path is not None:
            local_copy = os.path.join(run_dir, f"eval_{model_tag}.csv")
            shutil.copy2(csv_path, local_copy)
            print(f"  [Target Eval] Using {os.path.basename(csv_path)}")

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

        print(f"  [Target Eval] stem_metric={stem_metric}, stem_nll={stem_nll:.4f}")
        return per_task, stem_metric, per_task_nlls, stem_nll
