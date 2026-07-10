"""
Performance timer utility (adapted from quadmix).
"""

import time
from typing import Dict, List


class PerfTimer:
    _timings: Dict[str, List[float]] = {}
    _enabled: bool = True

    @classmethod
    def section(cls, name: str, prefix: str = "") -> "PerfTimerSection":
        key = f"{prefix}.{name}" if prefix else name
        return PerfTimerSection(key)

    @classmethod
    def report(cls) -> str:
        lines = ["PerfTimer Report:"]
        for key, times in sorted(cls._timings.items()):
            total = sum(times)
            avg = total / len(times)
            lines.append(f"  {key}: total={total:.1f}s, avg={avg:.1f}s, n={len(times)}")
        return "\n".join(lines)


class PerfTimerSection:
    def __init__(self, key: str):
        self.key = key
        self._start = None

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        elapsed = time.perf_counter() - self._start
        PerfTimer._timings.setdefault(self.key, []).append(elapsed)
