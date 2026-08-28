"""ExpSpec — the JSON contract between the submit host and the remote worker.

Design: the spec carries the FULLY-BUILT torchrun commands (constructed on the
submit host by the shared builders in climbmix.pipeline.nanochat_cmds) plus
every semantic field needed for audit. Building the commands on the submit
host guarantees the remote job runs EXACTLY the argv the local executor would
run — even if the code shipped to the container were to drift, the commands
themselves are pinned by the spec (and the worker refuses mismatched
spec_version values).

The worker (scripts/remote_worker.py) is standalone: it receives this JSON,
downloads the mixture shards + (on eval-only resume) the mid checkpoint from
OBS, runs the commands, and uploads result.json + logs + CSV back.
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional

SPEC_VERSION = 1


@dataclass
class ExpSpec:
    # Contract version. The worker refuses other versions (fail-fast instead
    # of silently misinterpreting fields).
    spec_version: int = SPEC_VERSION
    experiment_id: int = 0
    experiment_name: str = "main"
    model_tag: str = ""
    # Plain weight list — audit trail only (the mixture data itself is
    # already baked into the uploaded shards).
    weights: List[float] = field(default_factory=list)

    # ── container-side paths (fixed conventions, see RemoteConfig) ──
    nanochat_dir: str = "/home/ma-user/work/nanochat-npu"
    base_dir: str = "/home/ma-user/work/nanochat_base"
    work_dir: str = "/home/ma-user/work/climbmix_exp"
    # Container path of the d20 base checkpoint (symlink source for
    # base_checkpoints/{model_tag}).
    base_ckpt_src: str = ""

    # ── OBS references ──
    mixture_data_uri: str = ""   # obs://.../exps/exp_XXXX/mixture_data
    result_uri: str = ""         # obs://.../exps/exp_XXXX/result

    # ── commands (built by nanochat_cmds on the submit host) ──
    mid_train_cmd: List[str] = field(default_factory=list)
    eval_cmd: List[str] = field(default_factory=list)

    # ── execution mode ──
    # True = skip training, download the mid checkpoint from
    # {result_uri}/mid_checkpoint and run eval only (remote analog of the
    # .mid_train_ok resume marker).
    eval_only: bool = False
    # Upload {base_dir}/mid_checkpoints/{model_tag} to {result_uri}/
    # mid_checkpoint after successful training (enables eval-only resume and
    # post-hoc debugging; costs one model-weights upload per experiment).
    upload_checkpoint: bool = True

    # NPU pinning inside the job container (jobs get dedicated cards; the
    # pinning just mirrors the local executor's env semantics).
    visible_devices: List[int] = field(default_factory=lambda: [0])

    # Extra env for the train/eval subprocesses (HF_ENDPOINT, ...). The
    # worker also inherits the container's own environment.
    env: Dict[str, str] = field(default_factory=dict)

    def remote_mixture_data_dir(self) -> str:
        """Container path of the mixture shards (the --data-dir value baked
        into mid_train_cmd)."""
        import os
        return os.path.join(self.work_dir, "mixture_data")

    def to_dict(self) -> Dict:
        return {
            "spec_version": self.spec_version,
            "experiment_id": self.experiment_id,
            "experiment_name": self.experiment_name,
            "model_tag": self.model_tag,
            "weights": list(self.weights),
            "nanochat_dir": self.nanochat_dir,
            "base_dir": self.base_dir,
            "work_dir": self.work_dir,
            "base_ckpt_src": self.base_ckpt_src,
            "mixture_data_uri": self.mixture_data_uri,
            "result_uri": self.result_uri,
            "mid_train_cmd": list(self.mid_train_cmd),
            "eval_cmd": list(self.eval_cmd),
            "eval_only": self.eval_only,
            "upload_checkpoint": self.upload_checkpoint,
            "visible_devices": list(self.visible_devices),
            "env": dict(self.env),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @staticmethod
    def from_dict(d: Dict) -> "ExpSpec":
        v = d.get("spec_version")
        if v != SPEC_VERSION:
            raise ValueError(
                f"ExpSpec: unsupported spec_version {v!r} (expected "
                f"{SPEC_VERSION}) — the worker and the submit host run "
                f"different code versions; re-stage the assets bundle")
        known = {f for f in ExpSpec.__dataclass_fields__}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"ExpSpec: unknown fields {sorted(unknown)}")
        return ExpSpec(
            spec_version=d["spec_version"],
            experiment_id=int(d["experiment_id"]),
            experiment_name=str(d["experiment_name"]),
            model_tag=str(d["model_tag"]),
            weights=[float(x) for x in d.get("weights", [])],
            nanochat_dir=str(d["nanochat_dir"]),
            base_dir=str(d["base_dir"]),
            work_dir=str(d["work_dir"]),
            base_ckpt_src=str(d.get("base_ckpt_src", "")),
            mixture_data_uri=str(d["mixture_data_uri"]),
            result_uri=str(d["result_uri"]),
            mid_train_cmd=[str(x) for x in d.get("mid_train_cmd", [])],
            eval_cmd=[str(x) for x in d.get("eval_cmd", [])],
            eval_only=bool(d.get("eval_only", False)),
            upload_checkpoint=bool(d.get("upload_checkpoint", True)),
            visible_devices=[int(x) for x in d.get("visible_devices", [0])],
            env={str(k): str(v) for k, v in d.get("env", {}).items()},
        )

    @staticmethod
    def from_json(s: str) -> "ExpSpec":
        return ExpSpec.from_dict(json.loads(s))
