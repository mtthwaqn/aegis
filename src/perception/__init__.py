"""Vision pipeline: preprocess, segment, detect, annotate."""

from src.perception.video_processing import PerceptionResult, analyze

__all__ = ["PerceptionResult", "analyze"]
