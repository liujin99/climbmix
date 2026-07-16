"""
Proxy runner for CLIMB — method A: subprocess calls nanochat mid_train.py + base_eval.py.

Each proxy experiment:
  1. Prepare mixture-weighted data: sample docs by cluster weights → write parquet
  2. Call nanochat mid_train.py (annealing from base checkpoint, unique model-tag)
  3. Call nanochat base_eval.py (CORE metric evaluation)
  4. Parse evaluation results → ProxyResult

Key design:
  - Each experiment gets a unique model-tag (e.g. "climbmix_exp_0000")
  - Each experiment gets a unique data-dir with mixture-weighted parquet shard
  - Annealing semantics: lr_scale=1.0, warmup=0.0, warmdown=0.9
  - Fixed training via --num-iterations (not ratio-based)
  - Validation: high-signal CORE task subset → average centered accuracy
"""

import os
import json
import subprocess
import time
import numpy as np
import pandas as pd
import numpy.typing as npt
from typing import Dict, List, Optional, Any

from climbmix.core.types import MixtureConfig, MixtureWeights, ProxyResult, CLIMBConfig
from climbmix.sampling.data_selector import select_data_by_mixture


class ProxyRunner:

    def __init__(
        self,
        config: CLIMBConfig,
        cluster_labels: Optional[npt.NDArray[np.int64]] = None,
        token_counts: Optional[npt.NDArray[np.int64]] = None,
        metadata_manager: Optional[Any] = None,
    ):
        self.config = config
        self.nanochat_dir = config.nanochat_dir
        self._validate_nanochat()
        self.proxy_depth = config.proxy.depth
        self.proxy_num_iterations = config.proxy.training_iterations
        self.proxy_lr_scale = config.proxy.lr_scale
        self.proxy_warmup = config.proxy.warmup
        self.proxy_warmdown = config.proxy.warmdown
        self.proxy_phase1_ckpt = config.proxy.phase1_checkpoint_path
        self.validation_metric = config.proxy.validation_metric
        self.val_tasks = config.val_tasks
        self.device_type = config.device.device_type
        self.npu_devices = config.device.npu_devices

        self.cluster_labels = cluster_labels
        self.token_counts = token_counts
        self.metadata_manager = metadata_manager

    def _validate_nanochat(self):
        nc_dir = self.nanochat_dir
        if not os.path.isdir(nc_dir):
            raise FileNotFoundError(f"nanochat-npu not found at: {nc_dir}")
        required_scripts = ["scripts/mid_train.py", "scripts/base_eval.py"]
        for script in required_scripts:
            path = os.path.join(nc_dir, script)
            if not os.path.isfile(path):
                raise FileNotFoundError(f"nanochat script missing: {path}")
        required_modules = ["nanochat/gpt.py", "nanochat/checkpoint_manager.py", "nanochat/dataloader.py"]
        for module in required_modules:
            path = os.path.join(nc_dir, module)
            if not os.path.isfile(path):
                raise FileNotFoundError(f"nanochat module missing: {path}")
        print(f"  [ProxyRunner] nanochat-npu validated at {nc_dir}")

    def run_experiment(
        self,
        mixture_config: MixtureConfig,
        experiment_id: int = 0,
        data_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
    ) -> ProxyResult:
        output_dir = output_dir or self.config.output_dir
        exp_dir = os.path.join(output_dir, f"exp_{experiment_id:04d}")
        os.makedirs(exp_dir, exist_ok=True)

        model_tag = f"climbmix_{experiment_id:04d}"

        self._symlink_base_checkpoint(model_tag)

        mixture_data_dir = os.path.join(exp_dir, "mixture_data")

        t_start = time.time()
        print(f"\n  [Exp {experiment_id}] Starting proxy experiment (d{self.proxy_depth}, tag={model_tag})")

        # Step 1: Prepare mixture-weighted data
        print(f"  [Exp {experiment_id}] Preparing mixture-weighted data...")
        self._prepare_mixture_data(mixture_config, experiment_id, mixture_data_dir)

        # Step 2: mid_train.py (annealing)
        mid_cmd = self._build_mid_train_cmd(model_tag, mixture_data_dir)
        print(f"  [Exp {experiment_id}] mid_train: {' '.join(mid_cmd)}")
        mid_rc = self._run_subprocess(mid_cmd, exp_dir, "mid_train")

        # Step 3: base_eval.py (CORE evaluation)
        eval_cmd = self._build_eval_cmd(model_tag)
        print(f"  [Exp {experiment_id}] base_eval: {' '.join(eval_cmd)}")
        eval_rc = self._run_subprocess(eval_cmd, exp_dir, "eval")

        # Step 4: Copy mid checkpoint to exp_dir + parse results
        self._copy_mid_checkpoint(model_tag, exp_dir)
        per_task, val_accuracy, core_metric = self._parse_eval_results(model_tag, exp_dir)

        elapsed = time.time() - t_start

        meta = {
            "experiment_id": experiment_id,
            "model_tag": model_tag,
            "proxy_depth": self.proxy_depth,
            "proxy_scaling_M": self.config.proxy.scaling_M,
            "proxy_num_iterations": self.proxy_num_iterations,
            "lr_scale": self.proxy_lr_scale,
            "warmup": self.proxy_warmup,
            "warmdown": self.proxy_warmdown,
            "mixture_weights": mixture_config.mixture_weights.to_dict(),
            "validation_metric": self.validation_metric,
            "val_tasks": self.val_tasks,
            "elapsed_seconds": elapsed,
            "mid_train_rc": mid_rc,
            "eval_rc": eval_rc,
        }
        if per_task is not None:
            meta["per_task_accuracies"] = per_task
            meta["val_accuracy"] = val_accuracy
        if core_metric is not None:
            meta["core_metric"] = core_metric

        meta_path = os.path.join(exp_dir, "meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        print(f"  [Exp {experiment_id}] Done in {elapsed:.1f}s, accuracy={val_accuracy:.4f}\n")

        return ProxyResult(
            mixture_config=mixture_config,
            validation_loss=0.0,
            validation_accuracy=val_accuracy,
            per_task_accuracies=per_task,
            metadata=meta,
        )

    def run_batch(
        self,
        configs: List[MixtureConfig],
        data_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
    ) -> List[ProxyResult]:
        results = []
        for i, config in enumerate(configs):
            r = self.run_experiment(config, experiment_id=i, data_dir=data_dir, output_dir=output_dir)
            results.append(r)
        return results

    def _symlink_base_checkpoint(self, model_tag: str):
        base_dir = os.path.expanduser("~/.cache/nanochat")
        base_src = self.proxy_phase1_ckpt or os.path.join(base_dir, "base_checkpoints", f"d{self.proxy_depth}")
        base_dst = os.path.join(base_dir, "base_checkpoints", model_tag)

        if not os.path.exists(base_dst):
            if os.path.isdir(base_src):
                os.symlink(base_src, base_dst)
                print(f"  [Symlink] {base_dst} -> {base_src}")
            else:
                print(f"  [WARNING] Base checkpoint not found: {base_src}")

    def _copy_mid_checkpoint(self, model_tag: str, exp_dir: str):
        base_dir = os.path.expanduser("~/.cache/nanochat")
        mid_src_dir = os.path.join(base_dir, "mid_checkpoints", model_tag)
        mid_dst_dir = os.path.join(exp_dir, "mid_checkpoint")

        if os.path.isdir(mid_src_dir):
            import shutil
            if os.path.exists(mid_dst_dir):
                shutil.rmtree(mid_dst_dir)
            shutil.copytree(mid_src_dir, mid_dst_dir)
            print(f"  [Copy] mid checkpoint -> {mid_dst_dir}")

    def _prepare_mixture_data(
        self,
        mixture_config: MixtureConfig,
        experiment_id: int,
        mixture_data_dir: str,
    ):
        if self.cluster_labels is None or self.metadata_manager is None:
            print(f"  [Exp {experiment_id}] WARNING: No cluster labels or metadata_manager, "
                  f"using raw data_dir directly")
            return

        os.makedirs(mixture_data_dir, exist_ok=True)

        selected_indices, _ = select_data_by_mixture(
            self.cluster_labels,
            mixture_config.mixture_weights,
            self.token_counts,
            seed=experiment_id + 42,
        )

        print(f"  [Exp {experiment_id}] Selected {len(selected_indices)} docs by mixture weights")

        texts = self.metadata_manager.read_texts(selected_indices)

        df = pd.DataFrame({"text": texts})
        shard_path = os.path.join(mixture_data_dir, "shard_00000.parquet")
        df.to_parquet(shard_path, index=False)

        val_df = pd.DataFrame({"text": texts[:max(1, len(texts) // 100)]})
        val_path = os.path.join(mixture_data_dir, "shard_00001.parquet")
        val_df.to_parquet(val_path, index=False)

        print(f"  [Exp {experiment_id}] Wrote {len(texts)} docs to {mixture_data_dir}")

    def _build_mid_train_cmd(
        self,
        model_tag: str,
        mixture_data_dir: str,
    ) -> List[str]:
        cmd = [
            "torchrun", "--standalone",
            f"--nproc_per_node={self.npu_devices}",
            "-m", "scripts.mid_train",
            "--run", model_tag,
            "--device-type", self.device_type,
            "--model-tag", model_tag,
            "--num-iterations", str(self.proxy_num_iterations),
            "--lr-scale", str(self.proxy_lr_scale),
            "--warmup-ratio", str(self.proxy_warmup),
            "--warmdown-ratio", str(self.proxy_warmdown),
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
            "-m", "scripts.base_eval",
            "--eval", "core",
            "--model-tag", model_tag,
            "--model-type", "mid",
            "--max-per-task", "500",
            "--device-type", self.device_type,
        ]
        return cmd

    def _run_subprocess(
        self,
        cmd: List[str],
        exp_dir: str,
        stage_name: str,
    ) -> int:
        log_path = os.path.join(exp_dir, f"{stage_name}.log")
        env = os.environ.copy()
        env["PYTHONPATH"] = self.nanochat_dir + ":" + env.get("PYTHONPATH", "")
        env["NANOCHAT_BASE_DIR"] = env.get("NANOCHAT_BASE_DIR", os.path.expanduser("~/.cache/nanochat"))

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

    def _parse_eval_results(
        self,
        model_tag: str,
        exp_dir: str,
    ) -> tuple:
        per_task: Optional[Dict[str, float]] = None
        val_accuracy: float = 0.0
        core_metric: Optional[float] = None

        base_dir = os.path.expanduser("~/.cache/nanochat")
        csv_dir = os.path.join(base_dir, "base_eval")

        if os.path.isdir(csv_dir):
            csv_files = sorted(os.listdir(csv_dir))
            csv_files = [f for f in csv_files if f.endswith(".csv") and f.startswith("mid_model_")]
            if csv_files:
                latest_csv = csv_files[-1]
                latest_csv_path = os.path.join(csv_dir, latest_csv)

                tagged_csv = f"{model_tag}.csv"
                tagged_path = os.path.join(csv_dir, tagged_csv)
                if os.path.exists(latest_csv_path):
                    import shutil
                    shutil.copy2(latest_csv_path, tagged_path)
                    local_copy = os.path.join(exp_dir, f"eval_{tagged_csv}")
                    shutil.copy2(latest_csv_path, local_copy)

                per_task = {}
                with open(tagged_path if os.path.exists(tagged_path) else latest_csv_path) as f:
                    for line in f:
                        parts = [p.strip() for p in line.strip().split(",")]
                        if len(parts) >= 3:
                            task_name = parts[0]
                            centered_val = parts[2]
                            if task_name == "CORE":
                                try:
                                    core_metric = float(centered_val)
                                except ValueError:
                                    pass
                                continue
                            try:
                                per_task[task_name] = float(centered_val)
                            except ValueError:
                                continue

        if per_task:
            task_subset = [per_task[t] for t in self.val_tasks if t in per_task]
            if task_subset:
                val_accuracy = sum(task_subset) / len(task_subset)

        if core_metric is None and per_task:
            task_subset = [per_task[t] for t in self.val_tasks if t in per_task]
            if task_subset:
                core_metric = sum(task_subset) / len(task_subset)

        print(f"  [Eval] val_accuracy={val_accuracy:.4f}, core_metric={core_metric}")
        return per_task, val_accuracy, core_metric
