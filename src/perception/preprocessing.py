"""Frame cleanup before segmentation."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray


def preprocess(frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Denoise, then stretch local contrast on the lightness channel."""
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Expected a BGR frame")
    denoised = cv2.bilateralFilter(frame, 5, 30, 30)
    lightness, a, b = cv2.split(cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB))
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    enhanced = cv2.merge((clahe.apply(lightness), a, b))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


def low_visibility(frame: NDArray[np.uint8]) -> bool:
    """True when the frame is too flat to trust a detection, such as haze."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var()) < 15.0
