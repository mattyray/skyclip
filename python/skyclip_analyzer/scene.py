"""
Scene change detection and analysis.

Detects:
- Hard cuts (abrupt scene changes)
- Gradual transitions (dissolves, fades)
- Scene boundaries for better segment splitting
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
from enum import Enum


class TransitionType(Enum):
    """Types of scene transitions."""
    HARD_CUT = "cut"
    DISSOLVE = "dissolve"
    FADE_TO_BLACK = "fade_black"
    FADE_FROM_BLACK = "fade_from_black"
    NONE = "none"


@dataclass
class SceneChange:
    """Detected scene change."""
    frame_number: int
    timestamp_ms: int
    transition_type: TransitionType
    confidence: float  # 0-1


@dataclass
class SceneAnalysis:
    """Complete scene analysis for a video."""
    scene_changes: List[SceneChange]
    avg_scene_duration_ms: float
    is_single_shot: bool


class SceneAnalyzer:
    """Analyzes scenes and detects transitions in video."""

    def __init__(self,
                 cut_threshold: float = 30.0,
                 fade_threshold: float = 10.0,
                 min_scene_frames: int = 15):
        """
        Initialize scene analyzer.

        Args:
            cut_threshold: Histogram difference threshold for hard cuts
            fade_threshold: Brightness change threshold for fades
            min_scene_frames: Minimum frames between scene changes
        """
        self.cut_threshold = cut_threshold
        self.fade_threshold = fade_threshold
        self.min_scene_frames = min_scene_frames

    def analyze_video(self, video_path: str, start_ms: int = 0, end_ms: Optional[int] = None) -> SceneAnalysis:
        """
        Analyze a video for scene changes.

        Args:
            video_path: Path to video file
            start_ms: Start time in milliseconds
            end_ms: End time in milliseconds

        Returns:
            SceneAnalysis with detected scene changes
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        start_frame = int((start_ms / 1000) * fps)
        end_frame = int((end_ms / 1000) * fps) if end_ms else total_frames

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        scene_changes = []
        prev_hist = None
        prev_brightness = None
        last_change_frame = start_frame - self.min_scene_frames
        frame_num = start_frame

        while frame_num < end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            # Calculate histogram for scene change detection
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
            cv2.normalize(hist, hist)

            # Calculate brightness for fade detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = float(np.mean(gray))

            if prev_hist is not None and (frame_num - last_change_frame) >= self.min_scene_frames:
                # Compare histograms using correlation
                correlation = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                diff = (1 - correlation) * 100  # Convert to percentage difference

                # Check for hard cut
                if diff > self.cut_threshold:
                    timestamp_ms = int((frame_num / fps) * 1000)
                    confidence = min(1.0, diff / 50.0)

                    scene_changes.append(SceneChange(
                        frame_number=frame_num,
                        timestamp_ms=timestamp_ms,
                        transition_type=TransitionType.HARD_CUT,
                        confidence=confidence
                    ))
                    last_change_frame = frame_num

                # Check for fade to/from black
                elif prev_brightness is not None:
                    brightness_change = brightness - prev_brightness

                    if brightness < 20 and prev_brightness > 50:
                        # Fade to black
                        timestamp_ms = int((frame_num / fps) * 1000)
                        scene_changes.append(SceneChange(
                            frame_number=frame_num,
                            timestamp_ms=timestamp_ms,
                            transition_type=TransitionType.FADE_TO_BLACK,
                            confidence=0.8
                        ))
                        last_change_frame = frame_num

                    elif brightness > 50 and prev_brightness < 20:
                        # Fade from black
                        timestamp_ms = int((frame_num / fps) * 1000)
                        scene_changes.append(SceneChange(
                            frame_number=frame_num,
                            timestamp_ms=timestamp_ms,
                            transition_type=TransitionType.FADE_FROM_BLACK,
                            confidence=0.8
                        ))
                        last_change_frame = frame_num

            prev_hist = hist.copy()
            prev_brightness = brightness
            frame_num += 1

        cap.release()

        # Calculate average scene duration
        duration_ms = ((end_frame - start_frame) / fps) * 1000
        num_scenes = len(scene_changes) + 1
        avg_scene_duration = duration_ms / num_scenes

        return SceneAnalysis(
            scene_changes=scene_changes,
            avg_scene_duration_ms=avg_scene_duration,
            is_single_shot=len(scene_changes) == 0
        )

    def detect_transition_type(self, video_path: str, frame_number: int, window: int = 30) -> TransitionType:
        """
        Analyze frames around a potential transition to determine its type.

        Args:
            video_path: Path to video
            frame_number: Frame where transition was detected
            window: Number of frames to analyze around the transition

        Returns:
            TransitionType detected
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return TransitionType.HARD_CUT

        start = max(0, frame_number - window)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

        brightnesses = []
        histograms = []

        for i in range(window * 2):
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightnesses.append(np.mean(gray))

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0], None, [50], [0, 180])
            cv2.normalize(hist, hist)
            histograms.append(hist)

        cap.release()

        if len(brightnesses) < window:
            return TransitionType.HARD_CUT

        # Check for dissolve (gradual histogram change)
        mid = len(histograms) // 2
        if mid > 0 and mid < len(histograms) - 1:
            before_diff = cv2.compareHist(histograms[0], histograms[mid], cv2.HISTCMP_CORREL)
            after_diff = cv2.compareHist(histograms[mid], histograms[-1], cv2.HISTCMP_CORREL)

            # Dissolve shows gradual change on both sides
            if before_diff < 0.9 and after_diff < 0.9:
                return TransitionType.DISSOLVE

        # Check for fades
        min_brightness = min(brightnesses)
        if min_brightness < 20:
            min_idx = brightnesses.index(min_brightness)
            if min_idx < len(brightnesses) // 2:
                return TransitionType.FADE_FROM_BLACK
            else:
                return TransitionType.FADE_TO_BLACK

        return TransitionType.HARD_CUT

    def find_best_cut_point(self, video_path: str, start_frame: int, end_frame: int) -> int:
        """
        Find the best frame to cut within a range (avoiding mid-transition).

        Args:
            video_path: Path to video
            start_frame: Start of range
            end_frame: End of range

        Returns:
            Best frame number to cut on
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return start_frame

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frame_scores = []
        prev_hist = None
        frame_num = start_frame

        while frame_num < end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
            cv2.normalize(hist, hist)

            if prev_hist is not None:
                # Lower correlation = more stable (good cut point)
                correlation = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                stability = correlation  # Higher = more stable
                frame_scores.append((frame_num, stability))

            prev_hist = hist
            frame_num += 1

        cap.release()

        if not frame_scores:
            return start_frame

        # Return most stable frame
        best = max(frame_scores, key=lambda x: x[1])
        return best[0]
