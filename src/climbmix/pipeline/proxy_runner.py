"""
Proxy runner for CLIMB — trains proxy models on mixture-selected data.

Simpler than quadmix's runner because CLIMB:
  - Selects data by cluster mixture weights (not quality ranking + sigmoid)
  - Uses larger proxy models (350M instead of 1M)
  - Evaluates on downstream benchmark tasks
"""

import os
import math
import time
import json
import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
from typing import List, Optional, Dict, Any

from climbmix.core.types import MixtureConfig, MixtureWeights, ProxyResult, CLIMBConfig
from climbmix.core.proxy_model import ProxyModel, ModelArchitectureConfig
from climbmix.sampling.data_selector import select_data_by_mixture
from climbmix.npu.device import DeviceManager, DeviceType
from climbmix.pipeline.loss_utils import chunked_loss_from_hidden


class ProxyRunner:
    """
    Trains proxy models on data selected by mixture weights,
    evaluates on downstream tasks.
    """

    def __init__(
        self,
        config: CLIMBConfig,
        cluster_labels: npt.NDArray[np.int64],
        texts: Optional[List[str]] = None,
        token_ids: Optional[torch.Tensor] = None,
        token_counts: Optional[npt.NDArray[np.int64]] = None,
        val_data_path: Optional[str] = None,
        val_data: Optional[Dict] = None,
        output_dir: str = "./proxy_validation",
    ):
        self.config = config
        self.cluster_labels = cluster_labels
        self.texts = texts
        self._token_ids = token_ids
        self.token_counts = token_counts
        self.output_dir = output_dir
        self.num_clusters = len(np.unique(cluster_labels[cluster_labels >= 0]))

        self.model_variant = config.proxy.model_size
        self.device_type = config.device.device_type
        self.global_batch_size = config.proxy.batch_size
        self.micro_batch_size = config.proxy.micro_batch_size
        self.max_step = config.proxy.training_steps
        self.warmup_fraction = config.proxy.warmup_fraction
        self.learning_rate = config.proxy.learning_rate
        self.weight_decay = config.proxy.weight_decay
        self.grad_clip = config.proxy.grad_clip

        self.model_config = ModelArchitectureConfig.from_name(self.model_variant)
        self.block_size = self.model_config.block_size
        self.gradient_accumulation_steps = max(1, self.global_batch_size // self.micro_batch_size)

        if token_counts is None and texts is not None:
            self.token_counts = np.array(
                [max(1, len(t) // 4) for t in texts], dtype=np.int64
            )

        # Load validation data
        if val_data is not None:
            self._val_token_ids = val_data["token_ids"]
            self._val_loss_mask = val_data.get("loss_mask", None)
            self._val_task_labels = val_data.get("task_labels", None)
        elif val_data_path is not None:
            val = torch.load(val_data_path, map_location="cpu", weights_only=False)
            self._val_token_ids = val["token_ids"]
            self._val_loss_mask = val.get("loss_mask", None)
            self._val_task_labels = val.get("task_labels", None)
        else:
            self._val_token_ids = None
            self._val_loss_mask = None
            self._val_task_labels = None

        # Tokenizer
        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        except ImportError:
            self.tokenizer = None

    def _select_documents(
        self,
        mixture_config: MixtureConfig,
        experiment_id: int = 0,
    ) -> np.ndarray:
        """Select documents based on mixture weights."""
        selected, _ = select_data_by_mixture(
            self.cluster_labels,
            mixture_config.mixture_weights,
            self.token_counts,
            seed=experiment_id + 42,
        )
        return selected

    def _tokenize_selected(
        self,
        selected_indices: np.ndarray,
    ) -> torch.Tensor:
        """Tokenize selected documents."""
        if self._token_ids is not None:
            return self._token_ids[selected_indices]

        if self.texts is None or self.tokenizer is None:
            n = len(selected_indices)
            return torch.randint(0, self.model_config.vocab_size, (n, self.block_size))

        selected_texts = [self.texts[i] for i in selected_indices]
        enc = self.tokenizer(
            selected_texts,
            max_length=self.block_size,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return enc["input_ids"]

    def run_experiment(
        self,
        mixture_config: MixtureConfig,
        experiment_id: int = 0,
    ) -> ProxyResult:
        """Train one proxy model on mixture-selected data."""

        device_mgr = DeviceManager(
            device_type=DeviceType(self.device_type),
            npu_device_id=self.config.device.npu_device_id,
        )
        device = device_mgr.get_device()

        # Select documents
        selected_indices = self._select_documents(mixture_config, experiment_id)
        print(f"  [Exp {experiment_id}] Selected {len(selected_indices)} docs "
              f"with mixture weights")

        # Tokenize
        train_tokens = self._tokenize_selected(selected_indices)

        # Create model
        model = ProxyModel(config=self.model_config).to(device)
        if device.type == "npu":
            model = model.to(torch.bfloat16)

        non_emb = model.count_params(non_embedding_only=True)
        print(f"  [Exp {experiment_id}] Model: {model.count_params():,} total, "
              f"{non_emb:,} non-emb, variant={self.model_variant}")

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=self.weight_decay,
        )

        # Prepare training data
        pad_id = self.tokenizer.pad_token_id if self.tokenizer else 0
        eos_id = self.tokenizer.eos_token_id if self.tokenizer else 0

        real_mask = train_tokens != pad_id
        non_empty = real_mask.any(dim=1)
        flat_train_parts = []
        eos_buf = torch.tensor([eos_id], dtype=train_tokens.dtype)
        for doc in train_tokens[non_empty]:
            flat_train_parts.append(doc[doc != pad_id])
            flat_train_parts.append(eos_buf)
        flat_train = torch.cat(flat_train_parts)
        del train_tokens, flat_train_parts, real_mask, non_empty

        num_steps = self.max_step
        grad_acc = self.gradient_accumulation_steps
        max_iters = num_steps * grad_acc
        total_blocks = max(1, flat_train.size(0) - self.block_size)

        warmup_steps = max(1, int(num_steps * self.warmup_fraction))

        # Training loop
        model.train()
        epoch_rng = np.random.default_rng(experiment_id + 42)
        perm = epoch_rng.permutation(total_blocks)
        epoch_pos = 0
        step_ct = 0
        loss_accum = torch.tensor(0.0, device=device)
        t_start = time.time()

        print(f"  [Exp {experiment_id}] Training {num_steps} steps "
              f"(grad_acc={grad_acc}, warmup={warmup_steps})")

        for iter_ct in range(max_iters):
            micro_in_step = iter_ct % grad_acc
            if micro_in_step == 0:
                remaining = total_blocks - epoch_pos
                if remaining < grad_acc * self.micro_batch_size:
                    perm = epoch_rng.permutation(total_blocks)
                    epoch_pos = 0

                accum_bs = self.micro_batch_size * grad_acc
                start_pos = epoch_pos
                end_pos = min(start_pos + accum_bs, total_blocks)
                block_starts = perm[start_pos:end_pos]
                epoch_pos = end_pos

                idx_cpu = block_starts.unsqueeze(1) + torch.arange(self.block_size + 1)
                batch_cpu = flat_train[idx_cpu]
                batch = batch_cpu.to(device)

            mb_start = micro_in_step * self.micro_batch_size
            mb_end = mb_start + self.micro_batch_size
            micro_batch = batch[mb_start:mb_end]
            inp = micro_batch[:, :self.block_size].contiguous()
            tgt = micro_batch[:, 1:self.block_size + 1].contiguous()

            hidden = model(inp, return_hidden=True)
            loss = chunked_loss_from_hidden(model, hidden, tgt, chunk_size=2048)
            (loss / grad_acc).backward()

            is_acc = (iter_ct + 1) % grad_acc != 0
            if not is_acc:
                lr = self._lr_schedule(iter_ct, max_iters)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step_ct += 1

                if step_ct % max(1, num_steps // 5) == 0:
                    avg = (loss_accum / max(1, iter_ct + 1)).item()
                    elapsed = time.time() - t_start
                    print(f"    [Exp {experiment_id}] Step {step_ct}/{num_steps}, "
                          f"loss={avg:.4f}, lr={lr:.2e}")

            loss_accum += loss.detach()

        # Validation
        val_loss, per_task_losses = self._run_validation(model, device)

        avg_train = (loss_accum / max_iters).item() if max_iters > 0 else 0

        os.makedirs(self.output_dir, exist_ok=True)
        exp_dir = os.path.join(self.output_dir, f"exp_{experiment_id:04d}")
        os.makedirs(exp_dir, exist_ok=True)

        meta = {
            "experiment_id": experiment_id,
            "variant": self.model_variant,
            "train_loss": avg_train,
            "val_loss": val_loss,
            "mixture_weights": mixture_config.mixture_weights.to_dict(),
            "num_selected_docs": len(selected_indices),
            "training_config": {
                "global_batch_size": self.global_batch_size,
                "micro_batch_size": self.micro_batch_size,
                "learning_rate": self.learning_rate,
                "block_size": self.block_size,
                "device_type": self.device_type,
            },
        }
        if per_task_losses is not None:
            meta["per_task_losses"] = per_task_losses

        with open(os.path.join(exp_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        return ProxyResult(
            mixture_config=mixture_config,
            validation_loss=val_loss,
            per_task_losses=per_task_losses,
            metadata=meta,
        )

    def _run_validation(self, model, device):
        """Run validation on downstream tasks."""
        if self._val_token_ids is None:
            return 3.5, None

        model.eval()
        val_tokens = self._val_token_ids.to(device)
        total_loss = 0.0
        n_batches = 0
        per_task: Dict[str, float] = {}

        val_batch_size = min(16, len(val_tokens))

        with torch.no_grad():
            for start in range(0, len(val_tokens), val_batch_size):
                batch = val_tokens[start:start + val_batch_size]
                inp = batch[:, :self.block_size]
                tgt = batch[:, 1:self.block_size + 1]

                if self._val_loss_mask is not None:
                    mask = self._val_loss_mask[start:start + val_batch_size].to(device)
                    logits = model(inp)
                    per_token = F.cross_entropy(
                        logits.reshape(-1, model.config.vocab_size),
                        tgt.reshape(-1),
                        reduction="none",
                    ).view(tgt.shape)
                    loss = float((per_token * mask).sum() / mask.sum().clamp(min=1))
                else:
                    hidden = model(inp, return_hidden=True)
                    loss = float(chunked_loss_from_hidden(model, hidden, tgt))

                total_loss += loss
                n_batches += 1

        avg_val = total_loss / max(1, n_batches)

        if self._val_task_labels is not None:
            unique_tasks = sorted(set(self._val_task_labels))
            for task in unique_tasks:
                task_indices = [i for i, t in enumerate(self._val_task_labels) if t == task]
                task_tokens = self._val_token_ids[task_indices].to(device)
                task_loss = 0.0
                with torch.no_grad():
                    for start in range(0, len(task_tokens), val_batch_size):
                        batch = task_tokens[start:start + val_batch_size]
                        inp = batch[:, :self.block_size]
                        tgt = batch[:, 1:self.block_size + 1]
                        hidden = model(inp, return_hidden=True)
                        task_loss += float(chunked_loss_from_hidden(model, hidden, tgt))
                per_task[task] = task_loss / max(1, len(task_tokens) // val_batch_size + 1)

        model.train()
        return avg_val, per_task if per_task else None

    def _lr_schedule(self, iter_ct, max_iters):
        warmup_iters = int(self.warmup_fraction * max_iters)
        if iter_ct < warmup_iters:
            return self.learning_rate * iter_ct / warmup_iters
        decay = 1.0 - (iter_ct - warmup_iters) / (max_iters - warmup_iters)
        return self.learning_rate * max(decay, 0.01)

    def run_batch(
        self,
        configs: List[MixtureConfig],
        cluster_labels: Optional[np.ndarray] = None,
        cluster_token_counts: Optional[np.ndarray] = None,
    ) -> List[ProxyResult]:
        """Run proxy experiments for a batch of mixture configs."""
        results = []
        for i, config in enumerate(configs):
            r = self.run_experiment(config, experiment_id=i)
            results.append(r)

            if self.device_type == "npu":
                import gc
                gc.collect()
                torch.npu.empty_cache()
            elif self.device_type == "cuda":
                import gc
                gc.collect()
                torch.cuda.empty_cache()

        return results
