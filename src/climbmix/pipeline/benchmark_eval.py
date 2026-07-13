"""
Benchmark evaluation using lm-evaluation-harness.

Paper (Section 2.2): evaluates proxy models on PIQA, ARC-Easy,
HellaSwag (0-shot) using lm-evaluation-harness to compute
benchmark accuracy. This replaces loss-based validation.

Why accuracy instead of loss (paper's rationale):
  1. Loss measures token-level prediction quality, but doesn't
     directly reflect downstream task performance
  2. Different data mixtures can achieve similar loss values but
     very different benchmark accuracies
  3. Accuracy on benchmarks directly measures the capability that
     matters for real-world applications
  4. Loss can be dominated by easy/common tokens, masking differences
     in higher-order reasoning that benchmarks capture
"""

import os
import json
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple

from climbmix.core.proxy_model import ProxyModel, ModelArchitectureConfig


def evaluate_with_lm_eval(
    model: ProxyModel,
    model_config: ModelArchitectureConfig,
    tasks: List[str],
    device: str = "cpu",
    batch_size: int = 8,
    output_dir: Optional[str] = None,
) -> Dict[str, float]:
    """
    Evaluate a proxy model on benchmark tasks using lm-evaluation-harness.

    Args:
        model: The proxy model to evaluate.
        model_config: Model architecture config.
        tasks: List of benchmark task names (e.g. ["piqa", "arc_easy", "hellaswag"]).
        device: Device string ("cpu", "cuda", "npu").
        batch_size: Batch size for evaluation.
        output_dir: Directory to save lm-eval results.

    Returns:
        Dict mapping task_name -> accuracy (0-1).
    """
    try:
        import lm_eval
        from lm_eval.models.hf_model import HFLM
    except ImportError:
        print("[BenchmarkEval] lm-eval not installed, falling back to loss-based validation")
        return {}

    import tempfile
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    tokenizer.pad_token = tokenizer.eos_token

    save_dir = output_dir or tempfile.mkdtemp(prefix="climbmix_eval_")
    os.makedirs(save_dir, exist_ok=True)

    checkpoint_path = os.path.join(save_dir, "proxy_checkpoint.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": {
            "n_layer": model_config.n_layer,
            "n_head": model_config.n_head,
            "n_embd": model_config.n_embd,
            "vocab_size": model_config.vocab_size,
            "block_size": model_config.block_size,
            "intermediate_size": model_config.intermediate_size,
        },
    }, checkpoint_path)

    lm = HFLM(
        pretrained=checkpoint_path,
        tokenizer=tokenizer,
        batch_size=batch_size,
        device=device,
    )

    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=tasks,
        num_fewshot=0,
        batch_size=batch_size,
    )

    task_accuracies: Dict[str, float] = {}
    if "results" in results:
        for task_name in tasks:
            task_result = results["results"].get(task_name, {})
            acc_key = f"{task_name},acc"
            acc_norm_key = f"{task_name},acc_norm"
            if acc_norm_key in task_result:
                task_accuracies[task_name] = float(task_result[acc_norm_key])
            elif acc_key in task_result:
                task_accuracies[task_name] = float(task_result[acc_key])

    if output_dir:
        results_path = os.path.join(output_dir, "lm_eval_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

    return task_accuracies


def compute_average_accuracy(task_accuracies: Dict[str, float]) -> float:
    """Compute average accuracy across all evaluated tasks."""
    if not task_accuracies:
        return 0.0
    return float(np.mean(list(task_accuracies.values())))
