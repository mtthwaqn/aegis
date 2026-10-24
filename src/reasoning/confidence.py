from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum


class ConfidenceStatus(StrEnum):
    HIGH_CONFIDENCE = "HIGH_CONFIDENCE"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"


@dataclass(frozen=True, slots=True)
class ConfidenceAssessment:
    score: float
    status: ConfidenceStatus

    @property
    def reliable(self) -> bool:
        return self.status is ConfidenceStatus.HIGH_CONFIDENCE


def assess_confidence(
    confidence: float,
    threshold: float = 0.65,
    is_occluded: bool = False,
) -> ConfidenceAssessment:
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise TypeError("confidence must be a number")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise TypeError("threshold must be a number")
    score = float(confidence)
    cutoff = float(threshold)
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("confidence must be finite and between 0 and 1")
    if not math.isfinite(cutoff) or not 0 < cutoff <= 1:
        raise ValueError("threshold must be finite and greater than 0 and at most 1")
    if not isinstance(is_occluded, bool):
        raise TypeError("is_occluded must be a boolean")
    status = (
        ConfidenceStatus.HIGH_CONFIDENCE
        if score >= cutoff and not is_occluded
        else ConfidenceStatus.LOW_CONFIDENCE
    )
    return ConfidenceAssessment(score=score, status=status)
