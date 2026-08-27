"""
Proxy runner for CLIMB — subprocess calls nanochat mid_train.py + base_eval.py.

Each proxy experiment:
  1. Prepare mixture-weighted data: 70% STEM (by cluster weights) + 30% ClimbMix
     (adaptive 3-50 shards, reverse download from shard 6542, not full 400B)
  2. Call nanochat mid_train.py (annealing from base checkpoint, unique model-tag)
  3. Call nanochat base_eval.py (STEM benchmark evaluation)
  4. Parse evaluation results → ProxyResult

Key design:
  - Each experiment gets a unique model-tag (e.g. "climbmix_main_0000");
    the {experiment_name} part (CLIMBConfig.experiment_name) keeps parallel
    runs with different names from overwriting each other's checkpoints
  - Each experiment gets a unique data-dir with mixture-weighted parquet shard
  - Resume: run_experiment reuses a completed experiment (exp_XXXX/meta.json
    with rc=0/0 and matching weights) instead of retraining; experiment ids
    are GLOBAL (== config ids, passed via run_batch(experiment_id_base)) so
    exp dirs never collide across iterations
  - Annealing semantics: lr_scale=1.0, warmup=0.0, warmdown=0.9
  - Fixed training via --num-iterations (not ratio-based)
  - Validation: STEM benchmarks → stem_metric (centered accuracy)
  - General data: ClimbMix shards (adaptive 3-50, reverse order from 6542)
    to avoid overlap with pretrain data (shards 0-999); count auto-calculated
    from STEM doc count, not full 400B dataset
"""

import os
import sys
import glob
import json
import shutil
import hashlib
import importlib
import subprocess
import threading
import time
import numpy as np
import numpy.typing as npt
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from climbmix.core.types import MixtureConfig, MixtureWeights, ProxyResult, CLIMBConfig
from climbmix.sampling.data_selector import select_data_by_mixture


class ProxyRunner:
    # Parallel eval safety: base_eval.py writes its results CSV into
    # {NANOCHAT_BASE_DIR}/base_eval/mid_model_{step:06d}.csv — a step-only
    # name with NO model tag (base_eval.py:537), and every proxy experiment
    # finishes at the same final step. With a shared base dir, two
    # concurrent evals overwrite each other's CSV and the wrong scores get
    # attributed silently (this used to force evals to be serialized behind
    # a global lock). Fix: each eval subprocess gets a PRIVATE base dir
    # (symlink farm — see _make_eval_base_dir) so its CSV path is physically
    # per-experiment; evals run fully in parallel, one per device group.

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
        self.eval_max_per_task = getattr(config, 'eval_max_per_task', -1)
        self.npu_per_exp = getattr(config, 'npu_per_exp', 0)
        self.proxy_target_tokens = config.proxy.target_tokens
        self.experiment_name = getattr(config, 'experiment_name', 'main') or "main"

        self.cluster_labels = cluster_labels
        self.token_counts = token_counts
        self.metadata_manager = metadata_manager
        # download_climbmix (called from _prepare_mixture_data in worker threads)
        # resumes into a fixed .tmp path; two threads downloading the same shard
        # would corrupt each other's partial writes.
        self._download_lock = threading.Lock()

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

    @staticmethod
    def _load_completed_result(meta_path: str, mixture_config: MixtureConfig) -> Optional[ProxyResult]:
        """Reconstruct a ProxyResult from a completed experiment's meta.json.

        Returns None (=> the experiment must be re-run) unless the meta file
        parses, BOTH mid_train and eval exited 0, and the recorded mixture
        weights match the requested ones (np.allclose). The weight check
        guards against reusing a result that belongs to a different config.
        """
        if not os.path.exists(meta_path):
            return None
        try:
            with open(meta_path) as f:
                meta = json.load(f)
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            return None
        if not isinstance(meta, dict):
            return None
        if meta.get("mid_train_rc") != 0 or meta.get("eval_rc") != 0:
            return None
        w = meta.get("weights")
        if not isinstance(w, list):
            return None
        try:
            w_arr = np.array(w, dtype=np.float64)
        except (TypeError, ValueError):
            return None
        if w_arr.shape != mixture_config.mixture_weights.weights.shape or \
                not np.allclose(w_arr, mixture_config.mixture_weights.weights, atol=1e-9):
            return None
        return ProxyResult(
            mixture_config=mixture_config,
            validation_loss=0.0,
            validation_accuracy=float(meta.get("val_accuracy", 0.0)),
            validation_nll=float(meta.get("stem_nll", 0.0)),
            per_task_accuracies=meta.get("per_task_accuracies"),
            per_task_nlls=meta.get("per_task_nlls"),
            metadata=meta,
        )

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
        meta_path = os.path.join(exp_dir, "meta.json")

        model_tag = f"climbmix_{self.experiment_name}_{experiment_id:04d}"
        t_start = time.time()

        # Resume level 1: completed experiment (meta.json with rc=0/0 and
        # matching weights) -> reuse scores without running any subprocess.
        reused = self._load_completed_result(meta_path, mixture_config)
        if reused is not None:
            print(f"\n  [Exp {experiment_id}] Reusing completed experiment "
                  f"(tag={model_tag}, weights match, rc=0/0)")
            # The mixture data of a finished experiment is no longer needed.
            shutil.rmtree(os.path.join(exp_dir, "mixture_data"), ignore_errors=True)
            return reused

        # Resume level 2 (.mid_train_ok marker): training of THIS exact config
        # already succeeded but eval did not complete (crash / kill / ^C).
        # Skip the expensive training and re-run eval only — without this, a
        # kill during a slow eval burns the full training again (speedrun
        # 2026-08-27: ^C during iteration-2's first eval would have wasted
        # all three ~1h trainings).
        if self._load_mid_train_marker(exp_dir, mixture_config, model_tag):
            print(f"\n  [Exp {experiment_id}] mid_train already complete "
                  f"(marker valid, tag={model_tag}) — re-running eval only")
            mid_rc = 0  # marker validity == training succeeded
        else:
            # Fresh (re)run: clear any partial state left by a previous crashed
            # attempt — a half-written mixture_data dir would feed the dataloader
            # stale shards alongside the new ones.
            if os.path.isdir(exp_dir):
                shutil.rmtree(exp_dir)
            os.makedirs(exp_dir, exist_ok=True)

            self._symlink_base_checkpoint(model_tag)

            mixture_data_dir = os.path.join(exp_dir, "mixture_data")

            group_tag = f"d{device_ids[0]}-{device_ids[-1]}" if device_ids else "all"
            print(f"\n  [Exp {experiment_id}] Starting proxy experiment (d{self.proxy_depth}, tag={model_tag}, npu={group_tag})")

            print(f"  [Exp {experiment_id}] Preparing mixture-weighted data "
                  f"({self.stem_ratio*100:.0f}% STEM + {(1-self.stem_ratio)*100:.0f}% general)...")
            self._prepare_mixture_data(mixture_config, experiment_id, mixture_data_dir,
                                       nproc_per_node=nproc_per_node)

            mid_cmd = self._build_mid_train_cmd(model_tag, mixture_data_dir,
                                                nproc_per_node=nproc_per_node,
                                                master_port=master_port)
            print(f"  [Exp {experiment_id}] mid_train: {' '.join(mid_cmd)}")
            mid_rc = self._run_subprocess(mid_cmd, exp_dir, "mid_train",
                                          device_ids=device_ids, master_port=master_port)
            if mid_rc != 0:
                # Fail-fast: a failed proxy training must never enter the search
                # as a fake score (0.0 looks like a real measurement and silently
                # poisons the predictor — observed in speedrun 2026-08-26 17:35).
                # run_batch converts this raise into an inf result with the error
                # recorded; resume re-runs the experiment.
                raise RuntimeError(
                    f"mid_train failed (rc={mid_rc}) for experiment {experiment_id}, "
                    f"see {os.path.join(exp_dir, 'mid_train.log')}")
            # Training succeeded: persist the marker so a later crash before
            # meta.json (i.e. during eval) resumes at eval, not at training.
            self._write_mid_train_marker(exp_dir, mixture_config, model_tag)

        # Eval in a PRIVATE base dir (see _make_eval_base_dir): the CSV base_eval
        # writes is addressed only by step, so a shared dir would let parallel
        # evals overwrite each other. Private dir -> no lock, full parallelism.
        eval_base = self._make_eval_base_dir(exp_dir, model_tag)
        eval_cmd = self._build_eval_cmd(model_tag,
                                        nproc_per_node=nproc_per_node,
                                        master_port=master_port)
        print(f"  [Exp {experiment_id}] base_eval: {' '.join(eval_cmd)}")
        eval_rc = self._run_subprocess(eval_cmd, exp_dir, "eval",
                                       device_ids=device_ids, master_port=master_port,
                                       base_dir_override=eval_base)
        if eval_rc != 0:
            raise RuntimeError(
                f"base_eval failed (rc={eval_rc}) for experiment {experiment_id}, "
                f"see {os.path.join(exp_dir, 'eval.log')}")

        csv_path = self._claim_eval_csv(exp_dir, model_tag, eval_base)

        self._copy_mid_checkpoint(model_tag, exp_dir)
        per_task, val_accuracy, stem_metric, per_task_nlls, stem_nll = \
            self._parse_eval_results(csv_path)

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
            # Plain weight list: exact-match key for experiment resume
            # (see _load_completed_result).
            "weights": mixture_config.mixture_weights.weights.tolist(),
            "validation_metric": self.validation_metric,
            "val_tasks": self.val_tasks,
            "stem_ratio": self.stem_ratio,
            "eval_benchmarks": self.eval_benchmarks,
            "eval_max_per_task": self.eval_max_per_task,
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
        experiment_id_base: int = 0,
    ) -> List[ProxyResult]:
        """Run a batch of proxy experiments.

        experiment_id_base: global offset for experiment ids (and thus exp dirs
        and model tags). The bootstrapper passes the current accumulated-config
        count so ids stay globally unique across iterations — exp_0000 of
        iteration 2 never collides with exp_0000 of iteration 1.
        """
        if self.npu_per_exp == 0 or self.npu_per_exp >= self.npu_devices:
            results = []
            for i, config in enumerate(configs):
                r = self.run_experiment(config, experiment_id=experiment_id_base + i,
                                        data_dir=data_dir, output_dir=output_dir)
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
                  f"experiments {experiment_id_base + batch_start}.."
                  f"{experiment_id_base + batch_end - 1} ({batch_size} parallel)")

            with ThreadPoolExecutor(max_workers=batch_size) as pool:
                futures = {}
                for i in range(batch_size):
                    local_idx = batch_start + i
                    exp_id = experiment_id_base + local_idx
                    group_id = i
                    dev_start = group_id * devices_per_exp
                    devices = list(range(dev_start, dev_start + devices_per_exp))
                    port = 29500 + group_id

                    future = pool.submit(
                        self.run_experiment,
                        configs[local_idx],
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
                    local_idx = exp_id - experiment_id_base
                    try:
                        results[local_idx] = future.result()
                    except Exception as e:
                        print(f"  [Exp {exp_id}] FAILED: {e}")
                        results[local_idx] = ProxyResult(
                            mixture_config=configs[local_idx],
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

    def _effective_nproc(self, nproc_per_node: Optional[int] = None) -> int:
        return nproc_per_node or self.npu_devices

    def _prepare_mixture_data(
        self,
        mixture_config: MixtureConfig,
        experiment_id: int,
        mixture_data_dir: str,
        nproc_per_node: Optional[int] = None,
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
            target_tokens=self.proxy_target_tokens,
            seed=experiment_id + 42,
        )
        print(f"  [Exp {experiment_id}] Selected {len(selected_indices)} STEM docs by mixture weights")

        stem_texts = self.metadata_manager.read_texts(selected_indices)

        # Write STEM texts to temp parquet shards (mix_general_data expects shard_*.parquet).
        # Convention (same as prepare_shards.py / mix_general_data.py): the LAST
        # shard_*.parquet is the val split; train shards must NOT contain val docs.
        stem_temp_dir = os.path.join(mixture_data_dir, "_stem_temp")
        os.makedirs(stem_temp_dir, exist_ok=True)

        import pyarrow as pa
        import pyarrow.parquet as pq

        nproc = self._effective_nproc(nproc_per_node)
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
                f"Exp {experiment_id}: only {n_train} train docs < 2*nproc ({2 * nproc}). "
                f"The DDP dataloader assigns row groups round-robin per rank; some ranks "
                f"would starve and hang before the first all_reduce. "
                f"Raise --proxy-target-tokens (currently {self.proxy_target_tokens} tokens)."
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

        print(f"  [Exp {experiment_id}] Wrote {n_train} train docs ({n_shards} shards, "
              f"rg_size={rg_size}) + {n_val} val docs")

        # Mix with ClimbMix general data via mix_general_data.py module
        if self.stem_ratio < 1.0 and self.general_data_dir:
            try:
                mix_mod = self._load_mix_module()
            except FileNotFoundError as e:
                raise FileNotFoundError(
                    f"Exp {experiment_id}: stem_ratio={self.stem_ratio} requires "
                    f"mix_general_data.py, but it could not be loaded: {e}"
                ) from e

            stem_train_files = sorted(
                os.path.join(stem_temp_dir, f)
                for f in os.listdir(stem_temp_dir)
                if f.startswith("shard_") and f.endswith(".parquet")
            )[:-1]

            stem_docs = mix_mod.count_stem_docs(stem_train_files)
            needed_shards = mix_mod.calc_climbmix_count(stem_docs, self.stem_ratio)

            print(f"  [Exp {experiment_id}] STEM: {stem_docs:,} docs -> need {needed_shards} ClimbMix shards "
                  f"({self.stem_ratio*100:.0f}% STEM + {(1-self.stem_ratio)*100:.0f}% general)")

            # Parallel experiments in the same process share general_data_dir;
            # serialize downloads (first thread downloads, the rest hit the
            # exists+validate cache in download_single_file).
            with self._download_lock:
                climb_files = mix_mod.download_climbmix(
                    self.general_data_dir, needed_shards, self.npu_devices
                )

            detected_batch = mix_mod.detect_shard_size(stem_train_files)
            num_output_files = len(stem_train_files)
            mix_mod.mix_data(
                stem_temp_dir, climb_files, mixture_data_dir,
                num_output_files, detected_batch, num_npu=nproc,
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
            "-m", "scripts.mid_train", "--",
            "--run", model_tag,
            "--device-type", self.device_type,
            "--model-tag", model_tag,
            "--num-iterations", str(self.proxy_num_iterations),
            "--lr-scale", str(self.proxy_lr_scale),
            "--warmup-ratio", str(self.proxy_warmup),
            "--warmdown-ratio", str(self.proxy_warmdown),
            # Base checkpoints save optimizer state as PER-RANK SHARDS
            # (optim_<step>_rank<r>.pt; 8-rank pretrain -> lm_head/wte
            # moments are [vocab/8, n_embd]). This proxy runs at a different
            # world size (npu_per_exp), and torch's load_state_dict does NOT
            # shape-check state tensors: the mismatched shard is silently
            # assigned and explodes at the first AdamW lerp_ (aclnnInplaceLerp
            # EZ1001 "32768 and 4096 cannot broadcast", speedrun 2026-08-26).
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
            # the speedrun (gsm8k_cot ~50min + math_cot_500 ~49min + 26 more
            # tasks) — pure duplication. Val bpb (--eval-every) stays on as
            # the training signal.
            "--core-metric-every", "-1",
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
            "-m", "scripts.base_eval", "--",
            "--eval", "core",
            "--eval-benchmarks", self.eval_benchmarks,
            "--model-tag", model_tag,
            "--model-type", "mid",
            "--device-type", self.device_type,
        ]
        # Subsample cap for cheap proxy evals: base_eval shuffles each task
        # with a FIXED seed (random.Random(1337)) before truncating
        # (base_eval.py:356-359), so every experiment scores the SAME subset
        # — scores stay comparable across candidate mixtures. -1 (default) =
        # full eval sets (production); the speedrun passes a small cap.
        if self.eval_max_per_task and self.eval_max_per_task > 0:
            cmd += ["--max-per-task", str(self.eval_max_per_task)]
        return cmd

    def _run_subprocess(
        self,
        cmd: List[str],
        exp_dir: str,
        stage_name: str,
        device_ids: Optional[List[int]] = None,
        master_port: Optional[int] = None,
        base_dir_override: Optional[str] = None,
    ) -> int:
        log_path = os.path.join(exp_dir, f"{stage_name}.log")
        env = os.environ.copy()
        env["PYTHONPATH"] = self.nanochat_dir + ":" + env.get("PYTHONPATH", "")
        # base_dir_override: private NANOCHAT_BASE_DIR for eval subprocesses
        # (per-experiment symlink farm) — see _make_eval_base_dir.
        env["NANOCHAT_BASE_DIR"] = base_dir_override or self.nanochat_base_dir

        if device_ids is not None:
            # ASCEND_RT_VISIBLE_DEVICES is the torch_npu-documented pinning var
            # (logical npu:k = k-th entry of the mask). ASCEND_VISIBLE_DEVICES
            # alone may be ignored by the runtime, which would pile every exp
            # of a parallel batch onto physical device 0. Set both (the
            # embedding workers pin with the RT var for the same reason).
            env["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(str(d) for d in device_ids)
            env["ASCEND_VISIBLE_DEVICES"] = ",".join(str(d) for d in device_ids)
            env["RANK_SIZE"] = str(len(device_ids))
        if master_port is not None:
            env["MASTER_PORT"] = str(master_port)

        print(f"  [{stage_name}] started (log: {log_path})")
        with open(log_path, "w") as log_f:
            proc = subprocess.Popen(
                cmd,
                cwd=self.nanochat_dir,
                env=env,
                stdout=log_f,
                stderr=subprocess.STDOUT,
            )
            # Heartbeat: subprocess output goes to the log file, so a long
            # stage otherwise prints NOTHING to the console for its entire
            # duration — a 3h eval looks exactly like a hung pipeline
            # (speedrun 2026-08-26). Print progress every 5 minutes instead.
            t0 = time.time()
            while True:
                try:
                    proc.wait(timeout=300)
                    break
                except subprocess.TimeoutExpired:
                    elapsed_min = (time.time() - t0) / 60
                    print(f"  [{stage_name}] running {elapsed_min:.0f}m | "
                          f"{self._tail_last_line(log_path)}")

        if proc.returncode != 0:
            print(f"  [{stage_name}] FAILED (exit code {proc.returncode}), see {log_path}")
        else:
            print(f"  [{stage_name}] Completed (exit code 0)")

        return proc.returncode

    @staticmethod
    def _tail_last_line(log_path: str, max_chars: int = 120) -> str:
        """Last non-empty line of a growing log file, truncated for console."""
        try:
            with open(log_path, "rb") as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                f.seek(max(0, size - 8192))
                chunk = f.read().decode("utf-8", errors="replace")
            lines = [l.strip() for l in chunk.splitlines() if l.strip()]
            if not lines:
                return "(no output yet)"
            last = lines[-1]
            return last[:max_chars] + ("..." if len(last) > max_chars else "")
        except OSError:
            return "(log not readable)"

    # ── mid_train resume marker (skip retraining when only eval is missing) ──

    @staticmethod
    def _weights_sha256(mixture_config: MixtureConfig) -> str:
        arr = np.ascontiguousarray(
            mixture_config.mixture_weights.weights, dtype=np.float64)
        return hashlib.sha256(arr.tobytes()).hexdigest()

    @staticmethod
    def _mid_train_marker_path(exp_dir: str) -> str:
        return os.path.join(exp_dir, ".mid_train_ok")

    def _load_mid_train_marker(
        self,
        exp_dir: str,
        mixture_config: MixtureConfig,
        model_tag: str,
    ) -> bool:
        """True iff a previous mid_train of THIS config (same weights hash,
        same model tag) succeeded and its checkpoint still exists. Any
        mismatch/corruption -> False (full retrain): fail-safe, never
        fail-wrong."""
        path = self._mid_train_marker_path(exp_dir)
        if not os.path.isfile(path):
            return False
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            return False
        if not isinstance(data, dict):
            return False
        if data.get("weights_sha256") != self._weights_sha256(mixture_config):
            return False
        if data.get("model_tag") != model_tag:
            return False
        ckpt_dir = os.path.join(self.nanochat_base_dir, "mid_checkpoints", model_tag)
        if not glob.glob(os.path.join(ckpt_dir, "model_*.pt")):
            return False
        return True

    def _write_mid_train_marker(
        self,
        exp_dir: str,
        mixture_config: MixtureConfig,
        model_tag: str,
    ) -> None:
        payload = {
            "weights_sha256": self._weights_sha256(mixture_config),
            "model_tag": model_tag,
            "num_iterations": self.proxy_num_iterations,
        }
        with open(self._mid_train_marker_path(exp_dir), "w") as f:
            json.dump(payload, f, indent=2)

    # ── private eval base dir (parallel-safe CSV attribution) ──

    def _make_eval_base_dir(self, exp_dir: str, model_tag: str) -> str:
        """Private NANOCHAT_BASE_DIR for this experiment's eval subprocess.

        base_eval.py writes its results CSV to
        {base_dir}/base_eval/mid_model_{step:06d}.csv — a step-only name with
        NO model tag (base_eval.py:537) — and every proxy experiment finishes
        at the same final step. With the shared base dir, two concurrent
        evals overwrite each other's CSV and the wrong scores get attributed
        silently (which used to force eval serialization behind a global
        lock). A private base dir makes the collision physically impossible
        and lets evals run fully in parallel.

        Everything the eval READS is symlinked to the real shared data; the
        two things it WRITES ({base_eval}/ CSV, {report}/) are private real
        dirs inside exp_dir.
        """
        eval_base = os.path.join(exp_dir, "_eval_base")
        # Rebuild from scratch on every eval attempt: a previously crashed
        # eval may have left a partial CSV or download in the private dirs.
        shutil.rmtree(eval_base, ignore_errors=True)

        # Model checkpoint — mid_train just succeeded (or the marker verified
        # it), so it must exist; refuse to eval anything else (fail-fast).
        mid_src = os.path.join(self.nanochat_base_dir, "mid_checkpoints", model_tag)
        if not os.path.isdir(mid_src):
            raise FileNotFoundError(
                f"mid checkpoint for eval not found: {mid_src} — refusing to "
                f"evaluate a missing/stale model")
        os.makedirs(os.path.join(eval_base, "mid_checkpoints"))
        os.symlink(mid_src, os.path.join(eval_base, "mid_checkpoints", model_tag))

        # Tokenizer — required by build_model (get_tokenizer / token_bytes).
        tok_src = os.path.join(self.nanochat_base_dir, "tokenizer")
        if not os.path.isdir(tok_src):
            raise FileNotFoundError(
                f"tokenizer not found at {tok_src} — eval cannot run")
        os.symlink(tok_src, os.path.join(eval_base, "tokenizer"))

        # Eval datasets: symlink when present (normal case — the shell
        # pre-flight downloads them once). When absent, skip the link and let
        # base_eval download its own copy into the private dir (correct but
        # slow); warn so the operator can pre-download once instead.
        for name in ("eval_bundle", "eval_stem"):
            src = os.path.join(self.nanochat_base_dir, name)
            if os.path.isdir(src):
                os.symlink(src, os.path.join(eval_base, name))
            else:
                print(f"  [Eval] WARNING: {src} not found — eval will download "
                      f"{name} into the private dir (slow)")

        # Private writable output dirs (CSV + report); never shared.
        os.makedirs(os.path.join(eval_base, "base_eval"), exist_ok=True)
        os.makedirs(os.path.join(eval_base, "report"), exist_ok=True)
        return eval_base

    def _claim_eval_csv(self, exp_dir: str, model_tag: str, eval_base: str) -> Optional[str]:
        """Move the CSV written by THIS eval from its private base dir into
        exp_dir.

        The private dir was rebuilt empty immediately before the eval
        subprocess started, so any mid_model_*.csv in it is unambiguously
        ours — no lock and no mtime heuristics needed. Moving it out also
        gives resume/debug a per-experiment record at eval_{model_tag}.csv.
        """
        csv_dir = os.path.join(eval_base, "base_eval")
        try:
            names = [f for f in os.listdir(csv_dir)
                     if f.startswith("mid_model_") and f.endswith(".csv")]
        except OSError:
            names = []
        if not names:
            print(f"  [Eval] WARNING: base_eval exited 0 but wrote no CSV in "
                  f"{csv_dir} — scores for {model_tag} will be NaN")
            return None
        newest = max(names,
                     key=lambda f: os.path.getmtime(os.path.join(csv_dir, f)))
        src = os.path.join(csv_dir, newest)
        dst = os.path.join(exp_dir, f"eval_{model_tag}.csv")
        shutil.move(src, dst)
        print(f"  [Eval] Claimed {newest} -> {dst}")
        return dst

    def _parse_eval_results(self, csv_path: Optional[str]) -> tuple:
        """Parse the eval CSV previously claimed into exp_dir by
        _claim_eval_csv. csv_path=None (eval produced no readable output)
        yields per_task=None — the search scores this experiment NaN."""
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
            task_subset = [per_task[t] for t in self.val_tasks if t in per_task]
            if task_subset:
                val_accuracy = sum(task_subset) / len(task_subset)

        if stem_nll == 0.0 and per_task_nlls:
            nll_subset = [per_task_nlls[t] for t in self.val_tasks if t in per_task_nlls]
            if nll_subset:
                stem_nll = sum(nll_subset) / len(nll_subset)

        print(f"  [Eval] stem_metric={stem_metric}, val_accuracy={val_accuracy:.4f}, stem_nll={stem_nll:.4f}")
        return per_task, val_accuracy, stem_metric, per_task_nlls, stem_nll
