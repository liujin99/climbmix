"""
ConcurrencyConfig — centralized parallelism configuration for ClimbMix.

Auto-calculates optimal outer-workers x inner-BLAS-threads ~ cpu_count,
preventing thread explosion on large servers (192 vCPUs).

Usage:
  from climbmix.utils.concurrency import ConcurrencyConfig
  cfg = ConcurrencyConfig()
  n_jobs = min(n_tasks, cfg.max_io_workers)
"""

import ctypes
import os
from dataclasses import dataclass


_BLAS_LIB_NAMES = [
    'libopenblas.so', 'libopenblas.so.0',
    'libscipy_openblas64.so',
    'libmkl_rt.so', 'libmkl_rt.so.1', 'libmkl_rt.so.2',
]


def _probe_blas_lib():
    for lib_name in _BLAS_LIB_NAMES:
        try:
            lib = ctypes.CDLL(lib_name)
            return lib
        except OSError:
            continue
    return None


def set_blas_threads(n: int) -> bool:
    lib = _probe_blas_lib()
    if lib is None:
        return False
    if hasattr(lib, 'openblas_set_num_threads'):
        lib.openblas_set_num_threads(n)
        return True
    if hasattr(lib, 'MKL_Set_Num_Threads'):
        lib.MKL_Set_Num_Threads(n)
        return True
    return False


@dataclass(frozen=True)
class ConcurrencyConfig:
    cpu_count: int = 0
    max_compute_workers: int = 0
    max_io_workers: int = 0

    def __post_init__(self):
        if self.cpu_count == 0:
            object.__setattr__(self, 'cpu_count', os.cpu_count() or 4)
        if self.max_compute_workers == 0:
            object.__setattr__(self, 'max_compute_workers',
                               min(self.cpu_count, max(16, self.cpu_count // 4)))
        if self.max_io_workers == 0:
            object.__setattr__(self, 'max_io_workers',
                               min(self.cpu_count, max(32, self.cpu_count // 4)))

    def blas_threads_for(self, n_workers: int) -> int:
        return max(1, round(self.cpu_count / max(1, n_workers)))

    def apply_env_vars(self):
        import sys
        bt = self.blas_threads_for(self.max_compute_workers)
        if "numpy" in sys.modules:
            import logging
            logging.getLogger(__name__).warning(
                "numpy already imported -- OPENBLAS/OMP/MKL thread settings "
                "will have no effect. Call apply_env_vars() before importing numpy."
            )
        os.environ.setdefault("OPENBLAS_NUM_THREADS", str(bt))
        os.environ.setdefault("OMP_NUM_THREADS", str(bt))
        os.environ.setdefault("MKL_NUM_THREADS", str(bt))
        os.environ.setdefault("NUMEXPR_NUM_THREADS", str(bt))
        os.environ.setdefault("RAYON_NUM_THREADS", "4")
