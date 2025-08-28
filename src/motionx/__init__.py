"""
motionx package

This package provides a simple interface for extracting motion from video streams.
It builds on top of OpenCV and exposes high-level classes and functions.

The primary API is :class:`motionx.core.MotionExtractor` along with parameter
dataclasses for different algorithms:
- :class:`motionx.core.DiffParams`
- :class:`motionx.core.Mog2Params`
- :class:`motionx.core.WeightedParams`

See the README for details.
"""

from .core import MotionExtractor, DiffParams, Mog2Params, WeightedParams

__all__ = [
    "MotionExtractor",
    "DiffParams",
    "Mog2Params",
    "WeightedParams",
]
