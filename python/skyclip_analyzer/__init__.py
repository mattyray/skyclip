"""
SkyClip Analyzer - Python sidecar for visual analysis and content-aware editing.

This module provides:
- Motion analysis via OpenCV optical flow
- Scene change detection
- Dominant color extraction
- Object detection via YOLO
- Semantic embeddings via CLIP
- Content-aware edit suggestions
"""

from .motion import MotionAnalyzer
from .scene import SceneAnalyzer
from .color import ColorAnalyzer
from .editor import EditSuggestionEngine

__version__ = "0.1.0"
__all__ = ["MotionAnalyzer", "SceneAnalyzer", "ColorAnalyzer", "EditSuggestionEngine"]
