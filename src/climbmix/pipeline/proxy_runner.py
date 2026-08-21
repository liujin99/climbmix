"""
Proxy runner for CLIMB — subprocess calls nanochat mid_train.py + base_eval.py.

Each proxy experiment:
  1. Prepare mixture-weighted data: 70% STEM (by cluster weights) + 30% ClimbMix
     (adaptive 3-50 shards, reverse download from shard 6542, not full 400B)
  2. Call nanochat mid_train.py (annealing from base checkpoint, unique model-tag)
  3. Call nanochat base_eval.py (STEM benchmark evaluation)
  4. Parse evaluation results → ProxyResult

Key design:
  - Each experiment gets a unique model-tag (e.g. "climbmix_exp_0000")
  - Each experiment gets a unique data-dir with mixture-weighted parquet shard
  - Annealing semantics: lr_scale=1.0, warmup=0.0, warmdown=0.9
  - Fixed training via --num-iterations (not ratio-based)
  - Validation: STEM benchmarks → stem_metric (centered accuracy)
  - General data: ClimbMix shards (adaptive 3-50, reverse order from 6542)
    to avoid overlap with pretrain data (shards 0-999); count auto-calculated
    from STEM doc count, not full 400B dataset
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
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

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
        self.nanochat_base_dir = config.nanochat_base_dir
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
        self.general_data_dir = config.general_data_dir
        self.stem_ratio = config.stem_ratio
        self.eval_benchmarks = config.eval_benchmarks
        self.npu_per_exp = getattr(config, 'npu_per_exp', 0)

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
        device_ids: Optional[List[int]] = None,
        master_port: Optional[int] = None,
        nproc_per_node: Optional[int] = None,
    ) -> ProxyResult:
        output_dir = output_dir or self.config.output_dir
        exp_dir = os.path.join(output_dir, f"exp_{experiment_id:04d}")
        os.makedirs(exp_dir, exist_ok=True)

        model_tag = f"climbmix_{experiment_id:04d}"

        self._symlink_base_checkpoint(model_tag)

        mixture_data_dir = os.path.join(exp_dir, "mixture_data")

        t_start = time.time()
        group_tag = f"d{device_ids[0]}-{device_ids[-1]}" if device_ids else "all"
        print(f"\n  [Exp {experiment_id}] Starting proxy experiment (d{self.proxy_depth}, tag={model_tag}, npu={group_tag})")

        print(f"  [Exp {experiment_id}] Preparing mixture-weighted data "
              f"({self.stem_ratio*100:.0f}% STEM + {(1-self.stem_ratio)*100:.0f}% general)...")
        self._prepare_mixture_data(mixture_config, experiment_id, mixture_data_dir)

        mid_cmd = self._build_mid_train_cmd(model_tag, mixture_data_dir,
                                            nproc_per_node=nproc_per_node,
                                            master_port=master_port)
        print(f"  [Exp {experiment_id}] mid_train: {' '.join(mid_cmd)}")
        mid_rc = self._run_subprocess(mid_cmd, exp_dir, "mid_train",
                                      device_ids=device_ids, master_port=master_port)

        eval_cmd = self._build_eval_cmd(model_tag,
                                        nproc_per_node=nproc_per_node,
                                        master_port=master_port)
        print(f"  [Exp {experiment_id}] base_eval: {' '.join(eval_cmd)}")
        eval_rc = self._run_subprocess(eval_cmd, exp_dir, "eval",
                                       device_ids=device_ids, master_port=master_port)

        self._copy_mid_checkpoint(model_tag, exp_dir)
        per_task, val_accuracy, stem_metric, per_task_nlls, stem_nll = self._parse_eval_results(model_tag, exp_dir)

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
            "stem_ratio": self.stem_ratio,
            "eval_benchmarks": self.eval_benchmarks,
            "elapsed_seconds": elapsed,
            "mid_train_rc": mid_rc,
            "eval_rc": eval_rc,
        }
        if per_task is not None:
            meta["per_task_accuracies"] = per_task
            meta["val_accuracy"] = val_accuracy
        if stem_metric is not None:
            meta["stem_metric"] = stem_metric
        if per_task_nlls is not None:
            meta["per_task_nlls"] = per_task_nlls
            meta["stem_nll"] = stem_nll

        meta_path = os.path.join(exp_dir, "meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        print(f"  [Exp {experiment_id}] Done in {elapsed:.1f}s, stem_metric={val_accuracy:.4f}, stem_nll={stem_nll:.4f}\n")

        return ProxyResult(
            mixture_config=mixture_config,
            validation_loss=0.0,
            validation_accuracy=val_accuracy,
            validation_nll=stem_nll,
            per_task_accuracies=per_task,
            per_task_nlls=per_task_nlls,
            metadata=meta,
        )

    def run_batch(
        self,
        configs: List[MixtureConfig],
        data_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
    ) -> List[ProxyResult]:
        if self.npu_per_exp == 0 or self.npu_per_exp >= self.npu_devices:
            results = []
            for i, config in enumerate(configs):
                r = self.run_experiment(config, experiment_id=i, data_dir=data_dir, output_dir=output_dir)
                results.append(r)
            return results

        devices_per_exp = self.npu_per_exp
        n_parallel = self.npu_devices // devices_per_exp
        assert self.npu_devices % devices_per_exp == 0, (
            f"npu_devices ({self.npu_devices}) must be divisible by npu_per_exp ({devices_per_exp})"
        )

        print(f"\n  [ProxyRunner] Parallel mode: {n_parallel} experiments x {devices_per_exp} NPUs = {self.npu_devices} total")

        results: List[Optional[ProxyResult]] = [None] * len(configs)

        for batch_start in range(0, len(configs), n_parallel):
            batch_end = min(batch_start + n_parallel, len(configs))
            batch_size = batch_end - batch_start
            print(f"\n  [ProxyRunner] Batch {batch_start // n_parallel + 1}: "
                  f"experiments {batch_start}..{batch_end - 1} ({batch_size} parallel)")

            with ThreadPoolExecutor(max_workers=batch_size) as pool:
                futures = {}
                for i in range(batch_size):
                    exp_id = batch_start + i
                    group_id = i
                    dev_start = group_id * devices_per_exp
                    devices = list(range(dev_start, dev_start + devices_per_exp))
                    port = 29500 + group_id

                    future = pool.submit(
                        self.run_experiment,
                        configs[exp_id],
                        experiment_id=exp_id,
                        data_dir=data_dir,
                        output_dir=output_dir,
                        device_ids=devices,
                        master_port=port,
                        nproc_per_node=devices_per_exp,
                    )
                    futures[future] = exp_id

                for future in as_completed(futures):
                    exp_id = futures[future]
                    try:
                        results[exp_id] = future.result()
                    except Exception as e:
                        print(f"  [Exp {exp_id}] FAILED: {e}")
                        results[exp_id] = ProxyResult(
                            mixture_config=configs[exp_id],
                            validation_loss=float("inf"),
                            validation_accuracy=0.0,
                            validation_nll=float("inf"),
                            per_task_accuracies={},
                            per_task_nlls={},
                            metadata={"experiment_id": exp_id, "error": str(e)},
                        )

        return results

    def _symlink_base_checkpoint(self, model_tag: str):
        base_dir = self.nanochat_base_dir
        base_src = self.proxy_phase1_ckpt or os.path.join(base_dir, "base_checkpoints", f"d{self.proxy_depth}")
        base_dst = os.path.join(base_dir, "base_checkpoints", model_tag)

        if not os.path.exists(base_dst):
            if os.path.isdir(base_src):
                os.symlink(base_src, base_dst)
                print(f"  [Symlink] {base_dst} -> {base_src}")
            else:
                print(f"  [WARNING] Base checkpoint not found: {base_src}")

    def _copy_mid_checkpoint(self, model_tag: str, exp_dir: str):
        mid_src_dir = os.path.join(self.nanochat_base_dir, "mid_checkpoints", model_tag)
        mid_dst_dir = os.path.join(exp_dir, "mid_checkpoint")

        if os.path.isdir(mid_src_dir):
            import shutil
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
        print(f"  [Exp {experiment_id}] Selected {len(selected_indices)} STEM docs by mixture weights")

        stem_texts = self.metadata_manager.read_texts(selected_indices)

        # Write STEM texts to temp parquet shards (mix_general_data expects shard_*.parquet)
        stem_temp_dir = os.path.join(mixture_data_dir, "_stem_temp")
        os.makedirs(stem_temp_dir, exist_ok=True)

        import pyarrow as pa
        import pyarrow.parquet as pq

        batch_per_file = 10000
        n_stem = len(stem_texts)
        n_shards = max(1, (n_stem + batch_per_file - 1) // batch_per_file)

        for i in range(n_shards):
            start = i * batch_per_file
            end = min(start + batch_per_file, n_stem)
            shard_table = pa.table({"text": stem_texts[start:end]})
            shard_path = os.path.join(stem_temp_dir, f"shard_{i:05d}.parquet")
            pq.write_table(shard_table, shard_path, row_group_size=1024)

        val_table = pa.table({"text": stem_texts[:max(1, n_stem // 100)]})
        val_path = os.path.join(stem_temp_dir, f"shard_{n_shards:05d}.parquet")
        pq.write_table(val_table, val_path, row_group_size=1)

        print(f"  [Exp {experiment_id}] Wrote {n_stem} STEM docs to {n_shards} temp shards")

        # Mix with ClimbMix general data via mix_general_data.py module
        if self.stem_ratio < 1.0 and self.general_data_dir:
            try:
                mix_mod = self._load_mix_module()
            except FileNotFoundError as e:
                print(f"  [Exp {experiment_id}] WARNING: {e}, using STEM only")
                self._copy_stem_only(stem_temp_dir, mixture_data_dir)
                shutil.rmtree(stem_temp_dir, ignore_errors=True)
                return

            stem_train_files = sorted(
                os.path.join(stem_temp_dir, f)
                for f in os.listdir(stem_temp_dir)
                if f.startswith("shard_") and f.endswith(".parquet")
            )[:-1]

            stem_docs = mix_mod.count_stem_docs(stem_train_files)
            needed_shards = mix_mod.calc_climbmix_count(stem_docs, self.stem_ratio)

            print(f"  [Exp {experiment_id}] STEM: {stem_docs:,} docs -> need {needed_shards} ClimbMix shards "
                  f"({self.stem_ratio*100:.0f}% STEM + {(1-self.stem_ratio)*100:.0f}% general)")

            climb_files = mix_mod.download_climbmix(
                self.general_data_dir, needed_shards, self.npu_devices
            )

            if not climb_files:
                print(f"  [Exp {experiment_id}] WARNING: No ClimbMix data available, using STEM only")
                self._copy_stem_only(stem_temp_dir, mixture_data_dir)
            else:
                detected_batch = mix_mod.detect_shard_size(stem_train_files)
                num_output_files = len(stem_train_files)
                mix_mod.mix_data(
                    stem_temp_dir, climb_files, mixture_data_dir,
                    num_output_files, detected_batch,
                )
                print(f"  [Exp {experiment_id}] Mixed {stem_docs:,} STEM + "
                      f"~{int(stem_docs * (1-self.stem_ratio) / self.stem_ratio):,} general")
        else:
            self._copy_stem_only(stem_temp_dir, mixture_data_dir)
            print(f"  [Exp {experiment_id}] No general data, using {n_stem} STEM docs only")

        shutil.rmtree(stem_temp_dir, ignore_errors=True)
        print(f"  [Exp {experiment_id}] Data ready at {mixture_data_dir}")

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
        nproc_per_node: Optional[int] = None,
        master_port: Optional[int] = None,
    ) -> List[str]:
        nproc = nproc_per_node or self.npu_devices
        cmd = [
            "torchrun", "--standalone",
            f"--nproc_per_node={nproc}",
        ]
        if master_port is not None:
            cmd += ["--master_port", str(master_port)]
        cmd += [
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
        nproc_per_node: Optional[int] = None,
        master_port: Optional[int] = None,
    ) -> List[str]:
        nproc = nproc_per_node or self.npu_devices
        cmd = [
            "torchrun", "--standalone",
            f"--nproc_per_node={nproc}",
        ]
        if master_port is not None:
            cmd += ["--master_port", str(master_port)]
        cmd += [
            "-m", "scripts.base_eval",
            "--eval", "core",
            "--eval-benchmarks", self.eval_benchmarks,
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
        device_ids: Optional[List[int]] = None,
        master_port: Optional[int] = None,
    ) -> int:
        log_path = os.path.join(exp_dir, f"{stage_name}.log")
        env = os.environ.copy()
        env["PYTHONPATH"] = self.nanochat_dir + ":" + env.get("PYTHONPATH", "")
        env["NANOCHAT_BASE_DIR"] = self.nanochat_base_dir

        if device_ids is not None:
            env["ASCEND_VISIBLE_DEVICES"] = ",".join(str(d) for d in device_ids)
            env["RANK_SIZE"] = str(len(device_ids))
        if master_port is not None:
            env["MASTER_PORT"] = str(master_port)

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
        per_task_nlls: Optional[Dict[str, float]] = None
        val_accuracy: float = 0.0
        stem_metric: Optional[float] = None
        stem_nll: float = 0.0

        csv_dir = os.path.join(self.nanochat_base_dir, "base_eval")

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
                per_task_nlls = {}
                with open(tagged_path if os.path.exists(tagged_path) else latest_csv_path) as f:
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
            task_subset = [per_task[t] for t in self.val_tasks if t in per_task]
            if task_subset:
                val_accuracy = sum(task_subset) / len(task_subset)

        if stem_nll == 0.0 and per_task_nlls:
            nll_subset = [per_task_nlls[t] for t in self.val_tasks if t in per_task_nlls]
            if nll_subset:
                stem_nll = sum(nll_subset) / len(nll_subset)

        print(f"  [Eval] stem_metric={stem_metric}, val_accuracy={val_accuracy:.4f}, stem_nll={stem_nll:.4f}")
        return per_task, val_accuracy, stem_metric, per_task_nlls, stem_nll
