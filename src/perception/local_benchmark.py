"""Time the locked live detector on this machine.

The live detector stays classical unless AEGIS_DETECTOR=trained. This measurement
is not a Graviton result and not a COOL result.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.perception.trained_detector import use_trained_detector
from src.perception.video_processing import analyze


def measure_classical(frame: NDArray[np.uint8], repeats: int = 3) -> dict[str, Any]:
    if use_trained_detector():
        raise RuntimeError(
            "The live detector is locked to classical. "
            "Unset AEGIS_DETECTOR before measuring the live path."
        )
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    samples: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        analyze(frame)
        samples.append((time.perf_counter() - started) * 1000)
    ordered = sorted(samples)
    p95_index = max(0, int(round(len(ordered) * 0.95)) - 1)
    return {
        "detector": "classical",
        "locked": True,
        "host": "local",
        "graviton_measured": False,
        "cool_measured": False,
        "deployment_verified": False,
        "frames": repeats,
        "p95_ms": ordered[p95_index],
    }
