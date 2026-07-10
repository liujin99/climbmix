"""
Device manager adapted from quadmix (CPU/CUDA/NPU support).
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional


class DeviceType(Enum):
    CPU = "cpu"
    CUDA = "cuda"
    NPU = "npu"


@dataclass
class NPUDeviceConfig:
    device_id: int = 0
    cann_version: str = "8.0.RC2"
    mixed_precision: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {"device_type": "npu", "device_id": self.device_id}


class DeviceManager:
    def __init__(self, device_type: DeviceType = DeviceType.CPU, npu_device_id: int = 0):
        self.device_type = device_type
        self.npu_device_id = npu_device_id
        self._device = self._init_device()

    def _init_device(self) -> Any:
        import torch
        if self.device_type == DeviceType.CPU:
            return torch.device("cpu")
        elif self.device_type == DeviceType.CUDA:
            if torch.cuda.is_available():
                return torch.device("cuda:0")
            self.device_type = DeviceType.CPU
            return torch.device("cpu")
        elif self.device_type == DeviceType.NPU:
            import torch_npu
            npu_count = torch.npu.device_count()
            if npu_count > 0:
                dev_id = min(self.npu_device_id, npu_count - 1)
                torch.npu.set_device(dev_id)
                return torch.device(f"npu:{dev_id}")
            raise RuntimeError("No NPU devices found")
        raise ValueError(f"Unsupported device: {self.device_type}")

    def get_device(self) -> Any:
        return self._device
