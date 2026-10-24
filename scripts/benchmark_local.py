"""Measure the locked classical detector locally.

Prints a local timing only. Graviton and COOL stay unmeasured, and nothing here
sets deployment_verified.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.perception.local_benchmark import measure_classical  # noqa: E402


def main() -> None:
    frame = np.zeros((180, 320, 3), dtype=np.uint8)
    frame[:] = (150, 145, 80)
    print(json.dumps(measure_classical(frame, repeats=3), indent=2))


if __name__ == "__main__":
    main()
