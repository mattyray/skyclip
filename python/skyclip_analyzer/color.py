"""
Dominant color extraction for content-aware transitions.

Extracts:
- Dominant colors per segment
- Color palette for matching similar scenes
- Brightness/contrast metrics
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from sklearn.cluster import KMeans


@dataclass
class ColorInfo:
    """Color information for a frame or segment."""
    dominant_colors: List[Tuple[int, int, int]]  # RGB tuples, sorted by dominance
    color_weights: List[float]  # Percentage of each color
    avg_brightness: float  # 0-255
    avg_saturation: float  # 0-255
    is_low_light: bool
    is_golden_hour: bool  # Warm orange/yellow tones


@dataclass
class ColorAnalysis:
    """Complete color analysis for a video segment."""
    segment_colors: ColorInfo
    frame_colors: List[Tuple[int, ColorInfo]]  # (frame_number, ColorInfo)
    color_consistency: float  # 0-1, how consistent are colors across segment


class ColorAnalyzer:
    """Analyzes colors in video for smart transitions."""

    def __init__(self, n_colors: int = 5, sample_rate: int = 10):
        """
        Initialize color analyzer.

        Args:
            n_colors: Number of dominant colors to extract
            sample_rate: Analyze every Nth frame
        """
        self.n_colors = n_colors
        self.sample_rate = sample_rate

    def analyze_frame(self, frame: np.ndarray) -> ColorInfo:
        """
        Extract color information from a single frame.

        Args:
            frame: BGR image array

        Returns:
            ColorInfo with dominant colors and metrics
        """
        # Resize for faster processing
        small = cv2.resize(frame, (100, 56))

        # Convert to RGB for color extraction
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        # Reshape to list of pixels
        pixels = rgb.reshape(-1, 3).astype(np.float32)

        # K-means clustering for dominant colors
        kmeans = KMeans(n_clusters=self.n_colors, n_init=10, random_state=42)
        kmeans.fit(pixels)

        # Get colors and their weights
        colors = kmeans.cluster_centers_.astype(int)
        labels = kmeans.labels_
        weights = np.bincount(labels) / len(labels)

        # Sort by weight (most dominant first)
        sorted_indices = np.argsort(weights)[::-1]
        colors = [tuple(colors[i]) for i in sorted_indices]
        weights = [float(weights[i]) for i in sorted_indices]

        # Calculate brightness and saturation
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        avg_brightness = float(np.mean(hsv[:, :, 2]))
        avg_saturation = float(np.mean(hsv[:, :, 1]))

        # Detect low light
        is_low_light = avg_brightness < 60

        # Detect golden hour (warm tones with high saturation)
        # Golden hour has orange/yellow dominant colors (hue 10-40)
        hue_channel = hsv[:, :, 0]
        warm_mask = ((hue_channel >= 5) & (hue_channel <= 25))
        warm_ratio = np.sum(warm_mask) / warm_mask.size
        is_golden_hour = warm_ratio > 0.3 and avg_saturation > 100

        return ColorInfo(
            dominant_colors=colors,
            color_weights=weights,
            avg_brightness=avg_brightness,
            avg_saturation=avg_saturation,
            is_low_light=is_low_light,
            is_golden_hour=is_golden_hour
        )

    def analyze_video(self, video_path: str, start_ms: int = 0, end_ms: Optional[int] = None) -> ColorAnalysis:
        """
        Analyze colors throughout a video segment.

        Args:
            video_path: Path to video file
            start_ms: Start time in milliseconds
            end_ms: End time in milliseconds

        Returns:
            ColorAnalysis with per-frame and aggregate color data
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        start_frame = int((start_ms / 1000) * fps)
        end_frame = int((end_ms / 1000) * fps) if end_ms else total_frames

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frame_colors = []
        all_pixels = []
        frame_num = start_frame

        while frame_num < end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            if (frame_num - start_frame) % self.sample_rate == 0:
                color_info = self.analyze_frame(frame)
                frame_colors.append((frame_num, color_info))

                # Collect pixels for aggregate analysis
                small = cv2.resize(frame, (50, 28))
                rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                all_pixels.extend(rgb.reshape(-1, 3).tolist())

            frame_num += 1

        cap.release()

        # Aggregate color analysis
        if all_pixels:
            pixels_array = np.array(all_pixels, dtype=np.float32)
            kmeans = KMeans(n_clusters=self.n_colors, n_init=10, random_state=42)
            kmeans.fit(pixels_array)

            colors = kmeans.cluster_centers_.astype(int)
            weights = np.bincount(kmeans.labels_) / len(kmeans.labels_)

            sorted_indices = np.argsort(weights)[::-1]
            segment_colors = [tuple(colors[i]) for i in sorted_indices]
            segment_weights = [float(weights[i]) for i in sorted_indices]

            # Average brightness/saturation from frame samples
            avg_brightness = np.mean([fc[1].avg_brightness for fc in frame_colors])
            avg_saturation = np.mean([fc[1].avg_saturation for fc in frame_colors])
            is_low_light = avg_brightness < 60
            is_golden_hour = any(fc[1].is_golden_hour for fc in frame_colors)

            segment_color_info = ColorInfo(
                dominant_colors=segment_colors,
                color_weights=segment_weights,
                avg_brightness=float(avg_brightness),
                avg_saturation=float(avg_saturation),
                is_low_light=is_low_light,
                is_golden_hour=is_golden_hour
            )
        else:
            segment_color_info = ColorInfo(
                dominant_colors=[(128, 128, 128)],
                color_weights=[1.0],
                avg_brightness=128.0,
                avg_saturation=0.0,
                is_low_light=False,
                is_golden_hour=False
            )

        # Calculate color consistency
        if len(frame_colors) > 1:
            # Compare each frame's dominant color to segment dominant
            segment_dominant = np.array(segment_color_info.dominant_colors[0])
            distances = []
            for _, fc in frame_colors:
                frame_dominant = np.array(fc.dominant_colors[0])
                dist = np.linalg.norm(segment_dominant - frame_dominant) / 441.67  # Max RGB distance
                distances.append(dist)
            color_consistency = 1.0 - np.mean(distances)
        else:
            color_consistency = 1.0

        return ColorAnalysis(
            segment_colors=segment_color_info,
            frame_colors=frame_colors,
            color_consistency=float(color_consistency)
        )

    def color_similarity(self, color1: ColorInfo, color2: ColorInfo) -> float:
        """
        Calculate similarity between two color profiles.

        Args:
            color1: First ColorInfo
            color2: Second ColorInfo

        Returns:
            Similarity score 0-1 (1 = identical)
        """
        # Compare dominant colors weighted by their importance
        total_dist = 0.0
        total_weight = 0.0

        for i, (c1, w1) in enumerate(zip(color1.dominant_colors, color1.color_weights)):
            if i < len(color2.dominant_colors):
                c2 = color2.dominant_colors[i]
                w2 = color2.color_weights[i]

                dist = np.linalg.norm(np.array(c1) - np.array(c2)) / 441.67
                weight = (w1 + w2) / 2
                total_dist += dist * weight
                total_weight += weight

        if total_weight == 0:
            return 0.5

        similarity = 1.0 - (total_dist / total_weight)

        # Also factor in brightness similarity
        brightness_diff = abs(color1.avg_brightness - color2.avg_brightness) / 255
        brightness_sim = 1.0 - brightness_diff

        # Weighted combination
        return 0.7 * similarity + 0.3 * brightness_sim

    def suggest_transition_for_colors(self, from_colors: ColorInfo, to_colors: ColorInfo) -> str:
        """
        Suggest best transition type based on color similarity.

        Args:
            from_colors: Color profile of outgoing clip
            to_colors: Color profile of incoming clip

        Returns:
            Suggested transition type: "cut", "dissolve", "dip_black"
        """
        similarity = self.color_similarity(from_colors, to_colors)

        # Very similar colors -> hard cut (seamless)
        if similarity > 0.8:
            return "cut"

        # Moderately similar -> dissolve
        if similarity > 0.5:
            return "dissolve"

        # Very different OR one is low light -> dip to black
        if from_colors.is_low_light or to_colors.is_low_light:
            return "dip_black"

        if similarity < 0.3:
            return "dip_black"

        return "dissolve"
