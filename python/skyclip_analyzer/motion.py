"""
Motion analysis using OpenCV optical flow.

Detects frame-to-frame motion magnitude and direction to identify:
- Action peaks (high movement moments)
- Camera movement vs subject movement
- Motion direction for smart cut matching
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class MotionFrame:
    """Motion data for a single frame."""
    frame_number: int
    magnitude: float  # 0-1 normalized motion magnitude
    direction: float  # degrees, 0=right, 90=up, 180=left, 270=down
    is_camera_motion: bool  # True if motion is uniform (camera), False if localized (subject)


@dataclass
class MotionAnalysis:
    """Complete motion analysis for a video segment."""
    frames: List[MotionFrame]
    avg_magnitude: float
    peak_magnitude: float
    peak_frame: int
    dominant_direction: float
    motion_consistency: float  # 0-1, how consistent is the motion direction


class MotionAnalyzer:
    """Analyzes motion in video using optical flow."""

    def __init__(self, sample_rate: int = 2):
        """
        Initialize the motion analyzer.

        Args:
            sample_rate: Analyze every Nth frame (2 = every other frame)
        """
        self.sample_rate = sample_rate
        # Farneback optical flow parameters
        self.flow_params = dict(
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )

    def analyze_video(self, video_path: str, start_ms: int = 0, end_ms: Optional[int] = None) -> MotionAnalysis:
        """
        Analyze motion in a video file or segment.

        Args:
            video_path: Path to the video file
            start_ms: Start time in milliseconds
            end_ms: End time in milliseconds (None = end of video)

        Returns:
            MotionAnalysis with per-frame motion data
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calculate frame range
        start_frame = int((start_ms / 1000) * fps)
        end_frame = int((end_ms / 1000) * fps) if end_ms else total_frames

        # Seek to start
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        motion_frames = []
        prev_gray = None
        frame_num = start_frame

        while frame_num < end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if prev_gray is not None and (frame_num - start_frame) % self.sample_rate == 0:
                # Calculate optical flow
                flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, **self.flow_params)

                # Extract magnitude and angle
                mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

                # Normalize magnitude (empirically, values rarely exceed 20 pixels/frame)
                normalized_mag = np.clip(mag / 20.0, 0, 1)
                avg_magnitude = float(np.mean(normalized_mag))

                # Calculate dominant direction (in degrees)
                # Weight by magnitude so strong motion counts more
                weighted_angles = ang * normalized_mag
                avg_angle = float(np.degrees(np.arctan2(
                    np.sum(np.sin(ang) * normalized_mag),
                    np.sum(np.cos(ang) * normalized_mag)
                )))
                if avg_angle < 0:
                    avg_angle += 360

                # Detect if motion is camera (uniform) or subject (localized)
                # Camera motion has low std dev of motion vectors
                motion_std = float(np.std(normalized_mag))
                is_camera_motion = motion_std < 0.15 and avg_magnitude > 0.05

                motion_frames.append(MotionFrame(
                    frame_number=frame_num,
                    magnitude=avg_magnitude,
                    direction=avg_angle,
                    is_camera_motion=is_camera_motion
                ))

            prev_gray = gray
            frame_num += 1

        cap.release()

        if not motion_frames:
            return MotionAnalysis(
                frames=[],
                avg_magnitude=0.0,
                peak_magnitude=0.0,
                peak_frame=start_frame,
                dominant_direction=0.0,
                motion_consistency=0.0
            )

        # Calculate aggregate stats
        magnitudes = [f.magnitude for f in motion_frames]
        directions = [f.direction for f in motion_frames]

        avg_magnitude = float(np.mean(magnitudes))
        peak_idx = int(np.argmax(magnitudes))
        peak_magnitude = magnitudes[peak_idx]
        peak_frame = motion_frames[peak_idx].frame_number

        # Dominant direction (circular mean)
        sin_sum = sum(np.sin(np.radians(d)) for d in directions)
        cos_sum = sum(np.cos(np.radians(d)) for d in directions)
        dominant_direction = float(np.degrees(np.arctan2(sin_sum, cos_sum)))
        if dominant_direction < 0:
            dominant_direction += 360

        # Motion consistency (how aligned are the directions)
        # Uses circular variance: 1 - R, where R is resultant length
        r = np.sqrt(sin_sum**2 + cos_sum**2) / len(directions)
        motion_consistency = float(r)

        return MotionAnalysis(
            frames=motion_frames,
            avg_magnitude=avg_magnitude,
            peak_magnitude=peak_magnitude,
            peak_frame=peak_frame,
            dominant_direction=dominant_direction,
            motion_consistency=motion_consistency
        )

    def find_action_peaks(self, analysis: MotionAnalysis, threshold: float = 0.3) -> List[int]:
        """
        Find frame numbers where motion peaks occur.

        Args:
            analysis: MotionAnalysis from analyze_video
            threshold: Minimum magnitude to count as a peak (0-1)

        Returns:
            List of frame numbers at motion peaks
        """
        if not analysis.frames:
            return []

        peaks = []
        magnitudes = [f.magnitude for f in analysis.frames]

        for i in range(1, len(magnitudes) - 1):
            if magnitudes[i] > threshold:
                # Local maximum
                if magnitudes[i] > magnitudes[i-1] and magnitudes[i] > magnitudes[i+1]:
                    peaks.append(analysis.frames[i].frame_number)

        return peaks

    def suggest_cut_frame(self, analysis: MotionAnalysis, prefer_motion: bool = True) -> int:
        """
        Suggest the best frame to cut on within a segment.

        Args:
            analysis: MotionAnalysis from analyze_video
            prefer_motion: If True, cut during motion; if False, cut during still

        Returns:
            Frame number to cut on
        """
        if not analysis.frames:
            return 0

        if prefer_motion:
            # Find frame with motion closest to average (smooth transition)
            target_mag = analysis.avg_magnitude
            best_frame = min(analysis.frames, key=lambda f: abs(f.magnitude - target_mag))
            return best_frame.frame_number
        else:
            # Find stillest frame
            best_frame = min(analysis.frames, key=lambda f: f.magnitude)
            return best_frame.frame_number
