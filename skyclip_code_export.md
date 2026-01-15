# SkyClip Source Code

## python/skyclip_analyzer/__init__.py

```python
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
```

## python/skyclip_analyzer/cli.py

```python
#!/usr/bin/env python3
"""
CLI interface for SkyClip Analyzer.

This is the entry point called by the Rust Tauri backend.
Commands are received as JSON on stdin, results returned as JSON on stdout.
"""

import sys
import json
import traceback
from typing import Dict, Any


def analyze_clip_command(args: Dict[str, Any]) -> Dict[str, Any]:
    """Run full analysis on a single clip."""
    from .motion import MotionAnalyzer
    from .scene import SceneAnalyzer
    from .color import ColorAnalyzer

    video_path = args["video_path"]
    start_ms = args.get("start_ms", 0)
    end_ms = args.get("end_ms")
    include_objects = args.get("include_objects", False)
    include_semantic = args.get("include_semantic", False)

    result = {}

    # Motion analysis
    try:
        motion = MotionAnalyzer()
        motion_result = motion.analyze_video(video_path, start_ms, end_ms)
        result["motion"] = {
            "avg_magnitude": motion_result.avg_magnitude,
            "peak_magnitude": motion_result.peak_magnitude,
            "peak_frame": motion_result.peak_frame,
            "dominant_direction": motion_result.dominant_direction,
            "motion_consistency": motion_result.motion_consistency,
            "action_peaks": motion.find_action_peaks(motion_result)
        }
    except Exception as e:
        result["motion_error"] = str(e)

    # Scene analysis
    try:
        scene = SceneAnalyzer()
        scene_result = scene.analyze_video(video_path, start_ms, end_ms)
        result["scene"] = {
            "scene_changes": [
                {
                    "frame": sc.frame_number,
                    "timestamp_ms": sc.timestamp_ms,
                    "type": sc.transition_type.value,
                    "confidence": sc.confidence
                }
                for sc in scene_result.scene_changes
            ],
            "avg_scene_duration_ms": scene_result.avg_scene_duration_ms,
            "is_single_shot": scene_result.is_single_shot
        }
    except Exception as e:
        result["scene_error"] = str(e)

    # Color analysis
    try:
        color = ColorAnalyzer()
        color_result = color.analyze_video(video_path, start_ms, end_ms)
        result["color"] = {
            "dominant_colors": color_result.segment_colors.dominant_colors,
            "color_weights": color_result.segment_colors.color_weights,
            "avg_brightness": color_result.segment_colors.avg_brightness,
            "avg_saturation": color_result.segment_colors.avg_saturation,
            "is_low_light": color_result.segment_colors.is_low_light,
            "is_golden_hour": color_result.segment_colors.is_golden_hour,
            "color_consistency": color_result.color_consistency
        }
    except Exception as e:
        result["color_error"] = str(e)

    # Object detection (optional, heavy)
    if include_objects:
        try:
            from .objects import ObjectDetector
            detector = ObjectDetector()
            obj_result = detector.analyze_video(video_path, start_ms, end_ms)
            result["objects"] = {
                "primary_subject": obj_result.primary_subject.value if obj_result.primary_subject else None,
                "subject_entry_direction": obj_result.subject_entry_direction,
                "subject_exit_direction": obj_result.subject_exit_direction,
                "avg_subjects_per_frame": obj_result.avg_subjects_per_frame,
                "has_consistent_subject": obj_result.has_consistent_subject
            }
        except Exception as e:
            result["objects_error"] = str(e)

    # Semantic analysis (optional, heavy)
    if include_semantic:
        try:
            from .semantic import SemanticAnalyzer
            semantic = SemanticAnalyzer()
            sem_result = semantic.analyze_video(video_path, start_ms, end_ms)
            result["semantic"] = {
                "scene_type": sem_result.scene_type,
                "top_descriptions": sem_result.top_descriptions[:3],
                "embedding_size": len(sem_result.embedding)
            }
        except Exception as e:
            result["semantic_error"] = str(e)

    return result


def generate_edit_sequence_command(args: Dict[str, Any]) -> Dict[str, Any]:
    """Generate edit sequence for multiple clips."""
    from .editor import analyze_and_generate

    clips = args["clips"]  # List of {clip_id, video_path, start_ms, end_ms}
    style = args.get("style", "cinematic")
    reorder = args.get("reorder", True)
    full_analysis = args.get("full_analysis", False)

    video_paths = [
        (c["clip_id"], c["video_path"], c["start_ms"], c["end_ms"])
        for c in clips
    ]

    result = analyze_and_generate(video_paths, style, reorder, full_analysis)
    return result


def suggest_transition_command(args: Dict[str, Any]) -> Dict[str, Any]:
    """Suggest transition between two specific clips."""
    from .editor import EditSuggestionEngine, EditStyle

    engine = EditSuggestionEngine(enable_yolo=False, enable_clip=False)
    style = EditStyle(args.get("style", "cinematic"))

    clip_a = args["clip_a"]
    clip_b = args["clip_b"]

    analysis_a = engine.analyze_clip(
        clip_a["clip_id"], clip_a["video_path"],
        clip_a["start_ms"], clip_a["end_ms"],
        full_analysis=False
    )
    analysis_b = engine.analyze_clip(
        clip_b["clip_id"], clip_b["video_path"],
        clip_b["start_ms"], clip_b["end_ms"],
        full_analysis=False
    )

    trans_type, duration, confidence, reasoning = engine.suggest_transition(
        analysis_a, analysis_b, style
    )

    return {
        "transition_type": trans_type,
        "transition_duration_ms": duration,
        "confidence": confidence,
        "reasoning": reasoning
    }


def search_by_text_command(args: Dict[str, Any]) -> Dict[str, Any]:
    """Search clips by text query using CLIP embeddings."""
    from .semantic import SemanticAnalyzer
    import numpy as np

    query = args["query"]
    embeddings_data = args["embeddings"]  # List of {clip_id, embedding: list}

    semantic = SemanticAnalyzer()

    # Convert embeddings back to numpy
    embeddings = [np.array(e["embedding"]) for e in embeddings_data]
    clip_ids = [e["clip_id"] for e in embeddings_data]

    results = semantic.search_by_text(query, embeddings)

    return {
        "results": [
            {"clip_id": clip_ids[idx], "similarity": sim}
            for idx, sim in results
        ]
    }


# Command dispatcher
COMMANDS = {
    "analyze_clip": analyze_clip_command,
    "generate_edit_sequence": generate_edit_sequence_command,
    "suggest_transition": suggest_transition_command,
    "search_by_text": search_by_text_command,
}


def main():
    """Main entry point - reads JSON commands from stdin."""
    # Read input
    if len(sys.argv) > 1:
        # Command passed as argument (single shot mode)
        try:
            request = json.loads(sys.argv[1])
        except json.JSONDecodeError as e:
            print(json.dumps({"error": f"Invalid JSON: {e}"}))
            sys.exit(1)
    else:
        # Read from stdin
        try:
            request = json.load(sys.stdin)
        except json.JSONDecodeError as e:
            print(json.dumps({"error": f"Invalid JSON on stdin: {e}"}))
            sys.exit(1)

    command = request.get("command")
    args = request.get("args", {})

    if command not in COMMANDS:
        print(json.dumps({"error": f"Unknown command: {command}"}))
        sys.exit(1)

    try:
        result = COMMANDS[command](args)
        print(json.dumps({"success": True, "result": result}))
    except Exception as e:
        print(json.dumps({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }))
        sys.exit(1)


if __name__ == "__main__":
    main()
```

## python/skyclip_analyzer/color.py

```python
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
```

## python/skyclip_analyzer/editor.py

```python
"""
Content-aware edit suggestion engine.

Combines all analysis to make intelligent editing decisions:
- Transition types based on visual similarity
- Cut points based on motion
- Clip reordering for better flow
- Pacing based on content type
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from enum import Enum

from .motion import MotionAnalyzer, MotionAnalysis
from .scene import SceneAnalyzer, SceneAnalysis, TransitionType
from .color import ColorAnalyzer, ColorAnalysis, ColorInfo
from .objects import ObjectDetector, ObjectAnalysis, ObjectCategory


class EditStyle(Enum):
    """Editing style presets."""
    CINEMATIC = "cinematic"  # Slow, smooth, long takes
    ACTION = "action"  # Fast cuts, high energy
    SOCIAL = "social"  # Short, punchy, attention-grabbing


@dataclass
class ClipAnalysis:
    """Complete analysis of a single clip."""
    clip_id: str
    video_path: str
    start_ms: int
    end_ms: int
    motion: Optional[MotionAnalysis] = None
    scene: Optional[SceneAnalysis] = None
    color: Optional[ColorAnalysis] = None
    objects: Optional[ObjectAnalysis] = None
    semantic_embedding: Optional[np.ndarray] = None
    scene_type: Optional[str] = None


@dataclass
class EditDecision:
    """A single edit decision for a clip in the sequence."""
    clip_id: str
    sequence_order: int
    adjusted_start_ms: int
    adjusted_end_ms: int
    transition_type: str  # "cut", "dissolve", "dip_black", "wipe"
    transition_duration_ms: int
    confidence: float
    reasoning: str


@dataclass
class EditSequence:
    """Complete edit sequence for a highlight reel."""
    decisions: List[EditDecision]
    total_duration_ms: int
    style: EditStyle
    was_reordered: bool


class EditSuggestionEngine:
    """Generates intelligent edit suggestions based on clip analysis."""

    def __init__(self, enable_yolo: bool = True, enable_clip: bool = True):
        """
        Initialize the edit suggestion engine.

        Args:
            enable_yolo: Whether to use YOLO for object detection
            enable_clip: Whether to use CLIP for semantic analysis
        """
        self.motion_analyzer = MotionAnalyzer()
        self.scene_analyzer = SceneAnalyzer()
        self.color_analyzer = ColorAnalyzer()
        self.object_detector = ObjectDetector() if enable_yolo else None
        self.enable_clip = enable_clip

        # Style-specific parameters
        self.style_params = {
            EditStyle.CINEMATIC: {
                "min_clip_duration_ms": 4000,
                "max_clip_duration_ms": 15000,
                "prefer_dissolves": True,
                "transition_duration_ms": 1000,
                "cut_on_motion": False,  # Prefer cutting on still moments
            },
            EditStyle.ACTION: {
                "min_clip_duration_ms": 1000,
                "max_clip_duration_ms": 5000,
                "prefer_dissolves": False,
                "transition_duration_ms": 200,
                "cut_on_motion": True,
            },
            EditStyle.SOCIAL: {
                "min_clip_duration_ms": 1500,
                "max_clip_duration_ms": 4000,
                "prefer_dissolves": False,
                "transition_duration_ms": 300,
                "cut_on_motion": True,
            },
        }

    def analyze_clip(self, clip_id: str, video_path: str, start_ms: int, end_ms: int,
                     full_analysis: bool = True) -> ClipAnalysis:
        """
        Perform complete analysis of a clip.

        Args:
            clip_id: Unique identifier for the clip
            video_path: Path to video file
            start_ms: Start time in milliseconds
            end_ms: End time in milliseconds
            full_analysis: If True, run all analyzers; if False, just motion/color

        Returns:
            ClipAnalysis with all analysis results
        """
        analysis = ClipAnalysis(
            clip_id=clip_id,
            video_path=video_path,
            start_ms=start_ms,
            end_ms=end_ms
        )

        # Always run motion and color analysis
        analysis.motion = self.motion_analyzer.analyze_video(video_path, start_ms, end_ms)
        analysis.color = self.color_analyzer.analyze_video(video_path, start_ms, end_ms)

        if full_analysis:
            # Scene analysis
            analysis.scene = self.scene_analyzer.analyze_video(video_path, start_ms, end_ms)

            # Object detection (if enabled)
            if self.object_detector:
                try:
                    analysis.objects = self.object_detector.analyze_video(video_path, start_ms, end_ms)
                except Exception as e:
                    print(f"Object detection failed: {e}")

            # Semantic analysis (if enabled)
            if self.enable_clip:
                try:
                    from .semantic import SemanticAnalyzer
                    semantic = SemanticAnalyzer()
                    info = semantic.analyze_video(video_path, start_ms, end_ms)
                    analysis.semantic_embedding = info.embedding
                    analysis.scene_type = info.scene_type
                except Exception as e:
                    print(f"Semantic analysis failed: {e}")

        return analysis

    def suggest_transition(self, from_clip: ClipAnalysis, to_clip: ClipAnalysis,
                           style: EditStyle) -> Tuple[str, int, float, str]:
        """
        Suggest best transition between two clips.

        Args:
            from_clip: Analysis of outgoing clip
            to_clip: Analysis of incoming clip
            style: Editing style

        Returns:
            Tuple of (transition_type, duration_ms, confidence, reasoning)
        """
        params = self.style_params[style]
        reasons = []

        # Start with style default
        base_transition = "dissolve" if params["prefer_dissolves"] else "cut"
        duration = params["transition_duration_ms"]
        confidence = 0.7

        # Color-based decision
        if from_clip.color and to_clip.color:
            color_similarity = self.color_analyzer.color_similarity(
                from_clip.color.segment_colors,
                to_clip.color.segment_colors
            )

            if color_similarity > 0.8:
                base_transition = "cut"
                confidence = 0.9
                reasons.append("similar colors allow hard cut")
            elif color_similarity < 0.4:
                if from_clip.color.segment_colors.is_low_light or to_clip.color.segment_colors.is_low_light:
                    base_transition = "dip_black"
                    duration = 500
                    reasons.append("different colors with low light suggests dip to black")
                else:
                    base_transition = "dissolve"
                    duration = min(1000, params["transition_duration_ms"] + 300)
                    reasons.append("very different colors need dissolve")

        # Motion-based decision
        if from_clip.motion and to_clip.motion:
            from_dir = from_clip.motion.dominant_direction
            to_dir = to_clip.motion.dominant_direction

            # If motion directions are similar, hard cut works well
            dir_diff = abs(from_dir - to_dir)
            if dir_diff > 180:
                dir_diff = 360 - dir_diff

            if dir_diff < 45 and from_clip.motion.avg_magnitude > 0.1:
                base_transition = "cut"
                confidence = max(confidence, 0.85)
                reasons.append("matching motion directions favor cut")

        # Subject continuity
        if from_clip.objects and to_clip.objects:
            from_exit = from_clip.objects.subject_exit_direction
            to_entry = to_clip.objects.subject_entry_direction

            # Good match: exit right, enter left (or vice versa)
            if from_exit == "right" and to_entry == "left":
                base_transition = "cut"
                confidence = 0.95
                reasons.append("subject exit/entry match for seamless cut")
            elif from_exit == "left" and to_entry == "right":
                base_transition = "cut"
                confidence = 0.95
                reasons.append("subject exit/entry match for seamless cut")

        # Semantic similarity (if available)
        if from_clip.semantic_embedding is not None and to_clip.semantic_embedding is not None:
            from .semantic import SemanticAnalyzer
            semantic = SemanticAnalyzer()
            similarity = semantic.compute_similarity(
                from_clip.semantic_embedding,
                to_clip.semantic_embedding
            )
            if similarity > 0.7:
                if base_transition == "dissolve":
                    duration = max(300, duration - 200)
                reasons.append(f"similar content (semantic similarity: {similarity:.2f})")

        reasoning = "; ".join(reasons) if reasons else "default style transition"

        return base_transition, duration, confidence, reasoning

    def suggest_cut_points(self, analysis: ClipAnalysis, style: EditStyle) -> Tuple[int, int]:
        """
        Suggest optimal start/end points for a clip.

        Args:
            analysis: ClipAnalysis for the clip
            style: Editing style

        Returns:
            Tuple of (adjusted_start_ms, adjusted_end_ms)
        """
        params = self.style_params[style]
        original_start = analysis.start_ms
        original_end = analysis.end_ms
        duration = original_end - original_start

        # Respect min/max duration
        target_duration = max(params["min_clip_duration_ms"],
                              min(duration, params["max_clip_duration_ms"]))

        if duration <= target_duration:
            return original_start, original_end

        # Need to trim - find best cut points using motion
        if analysis.motion and analysis.motion.frames:
            fps = 30  # Assume 30fps for proxy
            frames = analysis.motion.frames

            if params["cut_on_motion"]:
                # Find frame with good motion for cutting
                start_frame = self.motion_analyzer.suggest_cut_frame(analysis.motion, prefer_motion=True)
                # Find end cut point from the back
                # Look in the last portion of the clip
                trim_needed = duration - target_duration
                end_search_start = original_end - trim_needed - 2000
                end_candidates = [f for f in frames if f.frame_number * (1000/fps) > end_search_start]
                if end_candidates:
                    end_frame = min(end_candidates, key=lambda f: abs(f.magnitude - analysis.motion.avg_magnitude))
                    adjusted_end = int(end_frame.frame_number * (1000/fps))
                else:
                    adjusted_end = original_end
                adjusted_start = int(start_frame * (1000/fps))
            else:
                # Cut on still moments
                start_frame = self.motion_analyzer.suggest_cut_frame(analysis.motion, prefer_motion=False)
                adjusted_start = int(start_frame * (1000/fps))
                adjusted_end = adjusted_start + target_duration
        else:
            # No motion data - just trim evenly
            trim = (duration - target_duration) // 2
            adjusted_start = original_start + trim
            adjusted_end = original_end - trim

        # Clamp to original bounds
        adjusted_start = max(original_start, adjusted_start)
        adjusted_end = min(original_end, adjusted_end)

        return adjusted_start, adjusted_end

    def suggest_clip_order(self, clips: List[ClipAnalysis], style: EditStyle) -> List[int]:
        """
        Suggest optimal order for clips based on content.

        Args:
            clips: List of ClipAnalysis objects
            style: Editing style

        Returns:
            List of indices representing suggested order
        """
        if len(clips) <= 2:
            return list(range(len(clips)))

        # Build narrative arc: establishing -> action -> resolution
        # Score each clip for "intensity"
        intensities = []
        for clip in clips:
            intensity = 0.0
            if clip.motion:
                intensity += clip.motion.avg_magnitude * 2
            if clip.objects and clip.objects.has_consistent_subject:
                intensity += 0.3
            if clip.color and clip.color.segment_colors.is_golden_hour:
                intensity -= 0.2  # Golden hour often works as resolution
            intensities.append(intensity)

        # Find clips by role
        indices = list(range(len(clips)))

        # Sort by intensity
        sorted_by_intensity = sorted(zip(indices, intensities), key=lambda x: x[1])

        # Build sequence: low intensity -> high -> medium (resolution)
        # Opening: lower intensity
        # Middle: highest intensity
        # End: medium intensity

        n = len(clips)
        if n >= 3:
            # Pick lowest for opening
            opening = [sorted_by_intensity[0][0]]

            # Pick highest for middle
            middle = [sorted_by_intensity[-1][0]]
            if n >= 4:
                middle.append(sorted_by_intensity[-2][0])

            # Rest are in-between
            used = set(opening + middle)
            remaining = [i for i in indices if i not in used]

            # Build final sequence
            suggested_order = opening + remaining + middle

            # But prefer similar adjacent clips for flow
            # This is a simple reorder - more sophisticated would use graph optimization
            return suggested_order
        else:
            return indices

    def generate_edit_sequence(self, clips: List[ClipAnalysis], style: EditStyle,
                               reorder: bool = True) -> EditSequence:
        """
        Generate complete edit sequence for a set of clips.

        Args:
            clips: List of analyzed clips
            style: Editing style
            reorder: Whether to suggest reordering clips

        Returns:
            EditSequence with all decisions
        """
        if not clips:
            return EditSequence(decisions=[], total_duration_ms=0, style=style, was_reordered=False)

        # Optionally reorder clips
        if reorder and len(clips) > 2:
            order = self.suggest_clip_order(clips, style)
            ordered_clips = [clips[i] for i in order]
            was_reordered = order != list(range(len(clips)))
        else:
            ordered_clips = clips
            was_reordered = False

        decisions = []
        total_duration = 0

        for i, clip in enumerate(ordered_clips):
            # Suggest cut points for this clip
            adjusted_start, adjusted_end = self.suggest_cut_points(clip, style)

            # Suggest transition to next clip
            if i < len(ordered_clips) - 1:
                next_clip = ordered_clips[i + 1]
                trans_type, trans_duration, confidence, reasoning = self.suggest_transition(
                    clip, next_clip, style
                )
            else:
                # Last clip - no transition
                trans_type = "none"
                trans_duration = 0
                confidence = 1.0
                reasoning = "last clip in sequence"

            clip_duration = adjusted_end - adjusted_start
            total_duration += clip_duration

            decisions.append(EditDecision(
                clip_id=clip.clip_id,
                sequence_order=i,
                adjusted_start_ms=adjusted_start,
                adjusted_end_ms=adjusted_end,
                transition_type=trans_type,
                transition_duration_ms=trans_duration,
                confidence=confidence,
                reasoning=reasoning
            ))

        return EditSequence(
            decisions=decisions,
            total_duration_ms=total_duration,
            style=style,
            was_reordered=was_reordered
        )


def analyze_and_generate(video_paths: List[Tuple[str, str, int, int]],
                         style: str = "cinematic",
                         reorder: bool = True,
                         full_analysis: bool = True) -> Dict:
    """
    Convenience function to analyze clips and generate edit sequence.

    Args:
        video_paths: List of (clip_id, video_path, start_ms, end_ms)
        style: Editing style ("cinematic", "action", "social")
        reorder: Whether to reorder clips
        full_analysis: Whether to run full analysis (YOLO, CLIP)

    Returns:
        Dictionary with edit decisions
    """
    engine = EditSuggestionEngine(
        enable_yolo=full_analysis,
        enable_clip=full_analysis
    )

    edit_style = EditStyle(style)

    # Analyze all clips
    analyses = []
    for clip_id, video_path, start_ms, end_ms in video_paths:
        analysis = engine.analyze_clip(clip_id, video_path, start_ms, end_ms, full_analysis)
        analyses.append(analysis)

    # Generate edit sequence
    sequence = engine.generate_edit_sequence(analyses, edit_style, reorder)

    # Convert to JSON-serializable dict
    return {
        "total_duration_ms": sequence.total_duration_ms,
        "style": sequence.style.value,
        "was_reordered": sequence.was_reordered,
        "decisions": [
            {
                "clip_id": d.clip_id,
                "sequence_order": d.sequence_order,
                "adjusted_start_ms": d.adjusted_start_ms,
                "adjusted_end_ms": d.adjusted_end_ms,
                "transition_type": d.transition_type,
                "transition_duration_ms": d.transition_duration_ms,
                "confidence": d.confidence,
                "reasoning": d.reasoning
            }
            for d in sequence.decisions
        ]
    }
```

## python/skyclip_analyzer/motion.py

```python
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
```

## python/skyclip_analyzer/objects.py

```python
"""
Object detection using YOLO for content-aware analysis.

Detects:
- People, vehicles, boats, buildings
- Subject entry/exit frame direction
- Subject-centered vs landscape shots
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from enum import Enum


class ObjectCategory(Enum):
    """Categories of detected objects relevant to drone footage."""
    PERSON = "person"
    VEHICLE = "vehicle"  # cars, trucks, motorcycles
    BOAT = "boat"
    AIRCRAFT = "aircraft"
    BUILDING = "building"
    ANIMAL = "animal"
    OTHER = "other"


# COCO class mapping to our categories
COCO_TO_CATEGORY = {
    0: ObjectCategory.PERSON,  # person
    1: ObjectCategory.VEHICLE,  # bicycle
    2: ObjectCategory.VEHICLE,  # car
    3: ObjectCategory.VEHICLE,  # motorcycle
    5: ObjectCategory.VEHICLE,  # bus
    7: ObjectCategory.VEHICLE,  # truck
    8: ObjectCategory.BOAT,  # boat
    14: ObjectCategory.ANIMAL,  # bird
    15: ObjectCategory.ANIMAL,  # cat
    16: ObjectCategory.ANIMAL,  # dog
    17: ObjectCategory.ANIMAL,  # horse
    18: ObjectCategory.ANIMAL,  # sheep
    19: ObjectCategory.ANIMAL,  # cow
    20: ObjectCategory.ANIMAL,  # elephant
    21: ObjectCategory.ANIMAL,  # bear
    22: ObjectCategory.ANIMAL,  # zebra
    23: ObjectCategory.ANIMAL,  # giraffe
    # 4 = airplane, but usually not relevant for drone footage
}


@dataclass
class DetectedObject:
    """A single detected object."""
    category: ObjectCategory
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    center: Tuple[int, int]  # center point
    area_ratio: float  # ratio of frame area occupied


@dataclass
class FrameDetections:
    """All detections in a single frame."""
    frame_number: int
    objects: List[DetectedObject]
    has_subject: bool  # True if any significant object detected
    is_subject_centered: bool  # True if main subject is in center third


@dataclass
class ObjectAnalysis:
    """Complete object analysis for a video segment."""
    frame_detections: List[FrameDetections]
    primary_subject: Optional[ObjectCategory]
    subject_entry_direction: Optional[str]  # "left", "right", "top", "bottom"
    subject_exit_direction: Optional[str]
    avg_subjects_per_frame: float
    has_consistent_subject: bool


class ObjectDetector:
    """Detects objects in video using YOLO."""

    def __init__(self, model_size: str = "n", confidence_threshold: float = 0.5):
        """
        Initialize object detector.

        Args:
            model_size: YOLO model size ("n", "s", "m", "l", "x")
            confidence_threshold: Minimum confidence for detection
        """
        self.model = None
        self.model_size = model_size
        self.confidence_threshold = confidence_threshold
        self._model_loaded = False

    def _load_model(self):
        """Lazy load the YOLO model."""
        if self._model_loaded:
            return

        try:
            from ultralytics import YOLO
            self.model = YOLO(f"yolov8{self.model_size}.pt")
            self._model_loaded = True
        except ImportError:
            raise ImportError("ultralytics not installed. Run: pip install ultralytics")

    def detect_frame(self, frame: np.ndarray, frame_number: int) -> FrameDetections:
        """
        Detect objects in a single frame.

        Args:
            frame: BGR image array
            frame_number: Frame number for reference

        Returns:
            FrameDetections with all detected objects
        """
        self._load_model()

        height, width = frame.shape[:2]
        frame_area = height * width

        # Run detection
        results = self.model(frame, verbose=False, conf=self.confidence_threshold)

        objects = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Map to our category
                category = COCO_TO_CATEGORY.get(cls_id, ObjectCategory.OTHER)

                # Calculate metrics
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                obj_area = (x2 - x1) * (y2 - y1)
                area_ratio = obj_area / frame_area

                objects.append(DetectedObject(
                    category=category,
                    confidence=conf,
                    bbox=(x1, y1, x2, y2),
                    center=center,
                    area_ratio=area_ratio
                ))

        # Determine if there's a significant subject
        has_subject = any(obj.area_ratio > 0.01 for obj in objects)

        # Check if main subject is centered (in middle third)
        is_centered = False
        if objects:
            largest = max(objects, key=lambda o: o.area_ratio)
            center_x = largest.center[0]
            left_third = width / 3
            right_third = 2 * width / 3
            is_centered = left_third < center_x < right_third

        return FrameDetections(
            frame_number=frame_number,
            objects=objects,
            has_subject=has_subject,
            is_subject_centered=is_centered
        )

    def analyze_video(self, video_path: str, start_ms: int = 0, end_ms: Optional[int] = None,
                      sample_rate: int = 15) -> ObjectAnalysis:
        """
        Analyze objects throughout a video segment.

        Args:
            video_path: Path to video file
            start_ms: Start time in milliseconds
            end_ms: End time in milliseconds
            sample_rate: Analyze every Nth frame

        Returns:
            ObjectAnalysis with detection results
        """
        self._load_model()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        start_frame = int((start_ms / 1000) * fps)
        end_frame = int((end_ms / 1000) * fps) if end_ms else total_frames

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frame_detections = []
        category_counts: Dict[ObjectCategory, int] = {}
        frame_num = start_frame

        while frame_num < end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            if (frame_num - start_frame) % sample_rate == 0:
                detections = self.detect_frame(frame, frame_num)
                frame_detections.append(detections)

                # Count categories
                for obj in detections.objects:
                    category_counts[obj.category] = category_counts.get(obj.category, 0) + 1

            frame_num += 1

        cap.release()

        # Determine primary subject
        primary_subject = None
        if category_counts:
            primary_subject = max(category_counts.keys(), key=lambda k: category_counts[k])

        # Determine entry/exit directions
        entry_direction = None
        exit_direction = None

        if frame_detections and primary_subject:
            # Find first frame with subject
            for fd in frame_detections:
                subjects = [o for o in fd.objects if o.category == primary_subject]
                if subjects:
                    main_subject = max(subjects, key=lambda o: o.area_ratio)
                    cx = main_subject.center[0]
                    if cx < width / 3:
                        entry_direction = "left"
                    elif cx > 2 * width / 3:
                        entry_direction = "right"
                    else:
                        entry_direction = "center"
                    break

            # Find last frame with subject
            for fd in reversed(frame_detections):
                subjects = [o for o in fd.objects if o.category == primary_subject]
                if subjects:
                    main_subject = max(subjects, key=lambda o: o.area_ratio)
                    cx = main_subject.center[0]
                    if cx < width / 3:
                        exit_direction = "left"
                    elif cx > 2 * width / 3:
                        exit_direction = "right"
                    else:
                        exit_direction = "center"
                    break

        # Calculate average subjects per frame
        total_subjects = sum(len(fd.objects) for fd in frame_detections)
        avg_subjects = total_subjects / len(frame_detections) if frame_detections else 0

        # Check for consistent subject
        frames_with_subject = sum(1 for fd in frame_detections if fd.has_subject)
        has_consistent = frames_with_subject > len(frame_detections) * 0.5 if frame_detections else False

        return ObjectAnalysis(
            frame_detections=frame_detections,
            primary_subject=primary_subject,
            subject_entry_direction=entry_direction,
            subject_exit_direction=exit_direction,
            avg_subjects_per_frame=avg_subjects,
            has_consistent_subject=has_consistent
        )
```

## python/skyclip_analyzer/scene.py

```python
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
```

## python/skyclip_analyzer/semantic.py

```python
"""
Semantic understanding using CLIP embeddings.

Enables:
- Text-based search ("find waterfall shots")
- Scene categorization ("beach sunset", "mountain flyover")
- Grouping similar scenes for cohesive edits
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
import torch


@dataclass
class SemanticInfo:
    """Semantic information for a video segment."""
    embedding: np.ndarray  # 512 or 768-dim CLIP embedding
    top_descriptions: List[Tuple[str, float]]  # (description, confidence)
    scene_type: str  # "landscape", "action", "urban", etc.


# Common drone footage scene descriptions for zero-shot classification
SCENE_DESCRIPTIONS = [
    "aerial view of beach and ocean",
    "aerial view of mountains",
    "aerial view of forest and trees",
    "aerial view of city and buildings",
    "aerial view of fields and farmland",
    "aerial view of river or lake",
    "aerial view of sunset or sunrise",
    "aerial view of road and highway",
    "aerial view of snow and winter landscape",
    "aerial view of desert",
    "drone following a car or vehicle",
    "drone following a person",
    "drone flying over water",
    "drone revealing a landscape",
    "drone orbiting around a subject",
]

SCENE_TYPE_MAPPING = {
    "aerial view of beach and ocean": "beach",
    "aerial view of mountains": "mountain",
    "aerial view of forest and trees": "forest",
    "aerial view of city and buildings": "urban",
    "aerial view of fields and farmland": "rural",
    "aerial view of river or lake": "water",
    "aerial view of sunset or sunrise": "golden_hour",
    "aerial view of road and highway": "infrastructure",
    "aerial view of snow and winter landscape": "winter",
    "aerial view of desert": "desert",
    "drone following a car or vehicle": "action",
    "drone following a person": "action",
    "drone flying over water": "water",
    "drone revealing a landscape": "reveal",
    "drone orbiting around a subject": "orbit",
}


class SemanticAnalyzer:
    """Analyzes video semantics using CLIP."""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        """
        Initialize semantic analyzer.

        Args:
            model_name: CLIP model to use
        """
        self.model = None
        self.processor = None
        self.model_name = model_name
        self._model_loaded = False
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"

    def _load_model(self):
        """Lazy load the CLIP model."""
        if self._model_loaded:
            return

        try:
            from transformers import CLIPProcessor, CLIPModel

            self.model = CLIPModel.from_pretrained(self.model_name)
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            self._model_loaded = True
        except ImportError:
            raise ImportError("transformers not installed. Run: pip install transformers")

    def get_frame_embedding(self, frame: np.ndarray) -> np.ndarray:
        """
        Get CLIP embedding for a single frame.

        Args:
            frame: BGR image array

        Returns:
            Embedding vector (512-dim for base model)
        """
        self._load_model()

        # Convert BGR to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process image
        inputs = self.processor(images=rgb, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            # Normalize
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        return image_features.cpu().numpy().flatten()

    def classify_frame(self, frame: np.ndarray, descriptions: Optional[List[str]] = None) -> List[Tuple[str, float]]:
        """
        Zero-shot classify a frame using text descriptions.

        Args:
            frame: BGR image array
            descriptions: List of text descriptions (uses defaults if None)

        Returns:
            List of (description, confidence) sorted by confidence
        """
        self._load_model()

        if descriptions is None:
            descriptions = SCENE_DESCRIPTIONS

        # Convert BGR to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process image and text
        inputs = self.processor(
            text=descriptions,
            images=rgb,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)

        probs = probs.cpu().numpy().flatten()
        results = list(zip(descriptions, probs))
        results.sort(key=lambda x: x[1], reverse=True)

        return results

    def analyze_video(self, video_path: str, start_ms: int = 0, end_ms: Optional[int] = None,
                      sample_frames: int = 5) -> SemanticInfo:
        """
        Analyze semantic content of a video segment.

        Args:
            video_path: Path to video file
            start_ms: Start time in milliseconds
            end_ms: End time in milliseconds
            sample_frames: Number of frames to sample

        Returns:
            SemanticInfo with embedding and classifications
        """
        self._load_model()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        start_frame = int((start_ms / 1000) * fps)
        end_frame = int((end_ms / 1000) * fps) if end_ms else total_frames

        # Calculate frame interval for sampling
        frame_range = end_frame - start_frame
        interval = max(1, frame_range // sample_frames)

        embeddings = []
        all_classifications = []

        for i in range(sample_frames):
            frame_num = start_frame + (i * interval)
            if frame_num >= end_frame:
                break

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret:
                continue

            # Get embedding
            embedding = self.get_frame_embedding(frame)
            embeddings.append(embedding)

            # Classify frame
            classifications = self.classify_frame(frame)
            all_classifications.append(classifications)

        cap.release()

        # Average embeddings for segment representation
        if embeddings:
            avg_embedding = np.mean(embeddings, axis=0)
            # Renormalize
            avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)
        else:
            avg_embedding = np.zeros(512)

        # Aggregate classifications
        desc_scores = {}
        for classifications in all_classifications:
            for desc, score in classifications:
                desc_scores[desc] = desc_scores.get(desc, 0) + score

        # Average and sort
        if all_classifications:
            for desc in desc_scores:
                desc_scores[desc] /= len(all_classifications)

        top_descriptions = sorted(desc_scores.items(), key=lambda x: x[1], reverse=True)[:5]

        # Determine scene type
        scene_type = "landscape"  # default
        if top_descriptions:
            top_desc = top_descriptions[0][0]
            scene_type = SCENE_TYPE_MAPPING.get(top_desc, "landscape")

        return SemanticInfo(
            embedding=avg_embedding,
            top_descriptions=top_descriptions,
            scene_type=scene_type
        )

    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            embedding1: First CLIP embedding
            embedding2: Second CLIP embedding

        Returns:
            Similarity score 0-1
        """
        # Embeddings should already be normalized
        similarity = np.dot(embedding1, embedding2)
        # Clamp to 0-1 range
        return float(max(0, min(1, (similarity + 1) / 2)))

    def search_by_text(self, query: str, embeddings: List[np.ndarray]) -> List[Tuple[int, float]]:
        """
        Search video segments by text query.

        Args:
            query: Text query (e.g., "sunset over ocean")
            embeddings: List of segment embeddings

        Returns:
            List of (segment_index, similarity) sorted by similarity
        """
        self._load_model()

        # Get text embedding
        inputs = self.processor(text=[query], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        text_embedding = text_features.cpu().numpy().flatten()

        # Compare to all segment embeddings
        results = []
        for i, emb in enumerate(embeddings):
            similarity = self.compute_similarity(text_embedding, emb)
            results.append((i, similarity))

        results.sort(key=lambda x: x[1], reverse=True)
        return results
```

## src/App.css

```css
:root {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  font-size: 14px;
  line-height: 1.5;
  font-weight: 400;
  color: #1a1a1a;
  background-color: #f8f9fa;
  font-synthesis: none;
  text-rendering: optimizeLegibility;
  -webkit-font-smoothing: antialiased;
}

* {
  box-sizing: border-box;
}

.container {
  max-width: 1000px;
  margin: 0 auto;
  padding: 24px;
}

header {
  margin-bottom: 32px;
}

header h1 {
  margin: 0 0 4px 0;
  font-size: 28px;
  font-weight: 600;
}

.tagline {
  color: #666;
  margin: 0;
}

h2 {
  font-size: 18px;
  font-weight: 600;
  margin: 0 0 16px 0;
}

h3 {
  font-size: 16px;
  font-weight: 600;
  margin: 0 0 12px 0;
}

button {
  border-radius: 6px;
  border: 1px solid #d1d5db;
  padding: 8px 16px;
  font-size: 14px;
  font-weight: 500;
  font-family: inherit;
  color: #1a1a1a;
  background-color: #ffffff;
  cursor: pointer;
  transition: all 0.15s ease;
}

button:hover {
  background-color: #f3f4f6;
  border-color: #9ca3af;
}

button:active {
  background-color: #e5e7eb;
}

button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.ingest-button {
  background-color: #2563eb;
  color: white;
  border-color: #2563eb;
  margin-top: 16px;
}

.ingest-button:hover {
  background-color: #1d4ed8;
  border-color: #1d4ed8;
}

.error-banner {
  background-color: #fef2f2;
  border: 1px solid #fecaca;
  color: #dc2626;
  padding: 12px 16px;
  border-radius: 6px;
  margin-bottom: 24px;
}

.error {
  color: #dc2626;
}

section {
  background: white;
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  padding: 20px;
  margin-bottom: 24px;
}

.folder-info {
  margin: 16px 0;
  padding: 12px;
  background: #f9fafb;
  border-radius: 6px;
}

.folder-info p {
  margin: 4px 0;
}

.clips-table {
  width: 100%;
  border-collapse: collapse;
  margin-top: 16px;
  font-size: 13px;
}

.clips-table th,
.clips-table td {
  padding: 10px 12px;
  text-align: left;
  border-bottom: 1px solid #e5e7eb;
}

.clips-table th {
  background: #f9fafb;
  font-weight: 600;
  color: #374151;
}

.clips-table tr:hover {
  background: #f9fafb;
}

.ingest-result {
  margin-top: 20px;
  padding: 16px;
  background: #f0fdf4;
  border: 1px solid #bbf7d0;
  border-radius: 6px;
}

.ingest-result h3 {
  color: #16a34a;
}

.ingest-result p {
  margin: 4px 0;
  color: #166534;
}

.empty-state {
  color: #9ca3af;
  text-align: center;
  padding: 32px;
}

.flights-list {
  list-style: none;
  padding: 0;
  margin: 0;
}

.flights-list li {
  padding: 12px;
  border-bottom: 1px solid #e5e7eb;
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 12px;
}

.flights-list li:last-child {
  border-bottom: none;
}

.flights-list li:hover {
  background: #f9fafb;
}

.flight-info {
  flex: 1;
  cursor: pointer;
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.flight-path {
  font-size: 11px;
  color: #6b7280;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  max-width: 400px;
}

.delete-button {
  background: transparent;
  color: #dc2626;
  border: 1px solid #dc2626;
  padding: 4px 12px;
  font-size: 12px;
  border-radius: 4px;
  cursor: pointer;
}

.delete-button:hover {
  background: #dc2626;
  color: white;
}

.flight-meta {
  color: #6b7280;
  font-size: 13px;
}

.nav-tabs {
  display: flex;
  gap: 8px;
  margin-top: 16px;
}

.nav-tabs button {
  padding: 8px 16px;
  border: none;
  background: transparent;
  color: #6b7280;
  cursor: pointer;
  border-bottom: 2px solid transparent;
}

.nav-tabs button:hover {
  color: #1a1a1a;
}

.nav-tabs button.active {
  color: #2563eb;
  border-bottom-color: #2563eb;
}

.analyze-controls {
  background: #f9fafb;
  padding: 16px;
  border-radius: 8px;
  margin-bottom: 24px;
}

.analyze-controls h3 {
  margin-top: 0;
}

.profile-selector {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 12px;
  flex-wrap: wrap;
}

.profile-selector label {
  font-weight: 500;
}

.profile-selector select {
  padding: 8px 12px;
  border: 1px solid #d1d5db;
  border-radius: 6px;
  font-size: 14px;
  background: white;
}

.profile-desc {
  color: #6b7280;
  font-size: 13px;
}

.analyze-button {
  background-color: #059669;
  color: white;
  border-color: #059669;
}

.analyze-button:hover {
  background-color: #047857;
  border-color: #047857;
}

.section-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}

.back-button {
  background: transparent;
  border: none;
  color: #6b7280;
  cursor: pointer;
}

.back-button:hover {
  color: #1a1a1a;
}

.analyze-summary {
  background: #eff6ff;
  border: 1px solid #bfdbfe;
  padding: 16px;
  border-radius: 8px;
  margin-bottom: 24px;
}

.analyze-summary p {
  margin: 4px 0;
}

.segments-grid {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.segment-card {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 16px;
  background: white;
  border: 1px solid #e5e7eb;
  border-radius: 8px;
}

.segment-card:hover {
  border-color: #2563eb;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
}

.segment-rank {
  font-size: 18px;
  font-weight: 600;
  color: #6b7280;
  width: 40px;
}

.segment-thumbnail {
  width: 160px;
  height: 90px;
  border-radius: 6px;
  overflow: hidden;
  background: #1f2937;
  cursor: pointer;
  flex-shrink: 0;
}

.segment-thumbnail img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.thumbnail-placeholder {
  width: 100%;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #6b7280;
  font-size: 12px;
}

.segment-info {
  flex: 1;
}

.segment-time {
  font-weight: 500;
  font-size: 15px;
}

.segment-duration {
  color: #6b7280;
  font-size: 13px;
}

.segment-scores {
  text-align: right;
}

.score.primary {
  font-size: 24px;
  font-weight: 600;
  color: #2563eb;
}

.segment-signals {
  display: flex;
  gap: 12px;
  margin-top: 4px;
  font-size: 12px;
  color: #6b7280;
}

.segment-info {
  cursor: pointer;
}

.segment-actions {
  display: flex;
  gap: 8px;
}

.export-button {
  padding: 6px 12px;
  font-size: 12px;
  background-color: #059669;
  border-color: #059669;
  color: white;
}

.export-button:hover {
  background-color: #047857;
  border-color: #047857;
}

.export-button.source {
  background-color: #2563eb;
  border-color: #2563eb;
}

.export-button.source:hover {
  background-color: #1d4ed8;
  border-color: #1d4ed8;
}

.export-button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

/* Preview Modal */
.preview-modal {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.8);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
}

.preview-content {
  background: #1f2937;
  border-radius: 12px;
  padding: 20px;
  max-width: 90vw;
  max-height: 90vh;
  overflow: auto;
}

.preview-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}

.preview-header h3 {
  margin: 0;
  color: white;
}

.close-button {
  background: transparent;
  border: none;
  color: #9ca3af;
  cursor: pointer;
  font-size: 14px;
}

.close-button:hover {
  color: white;
}

.preview-info {
  margin-top: 16px;
  color: #9ca3af;
}

.preview-info p {
  margin: 4px 0;
}

.preview-note {
  font-size: 12px;
  font-style: italic;
}

.preview-actions {
  display: flex;
  gap: 12px;
  margin-top: 16px;
}

@media (prefers-color-scheme: dark) {
  :root {
    color: #e5e7eb;
    background-color: #111827;
  }

  section {
    background: #1f2937;
    border-color: #374151;
  }

  button {
    color: #e5e7eb;
    background-color: #374151;
    border-color: #4b5563;
  }

  button:hover {
    background-color: #4b5563;
    border-color: #6b7280;
  }

  .ingest-button {
    background-color: #2563eb;
    border-color: #2563eb;
    color: white;
  }

  .folder-info {
    background: #111827;
  }

  .clips-table th {
    background: #111827;
    color: #9ca3af;
  }

  .clips-table th,
  .clips-table td {
    border-color: #374151;
  }

  .clips-table tr:hover {
    background: #111827;
  }

  .flights-list li {
    border-color: #374151;
  }

  .flights-list li:hover {
    background: #111827;
  }

  .ingest-result {
    background: #064e3b;
    border-color: #065f46;
  }

  .ingest-result h3 {
    color: #34d399;
  }

  .ingest-result p {
    color: #a7f3d0;
  }

  .error-banner {
    background-color: #450a0a;
    border-color: #7f1d1d;
    color: #fca5a5;
  }

  .nav-tabs button {
    color: #9ca3af;
  }

  .nav-tabs button:hover {
    color: #e5e7eb;
  }

  .nav-tabs button.active {
    color: #60a5fa;
    border-bottom-color: #60a5fa;
  }

  .analyze-controls {
    background: #111827;
  }

  .profile-selector select {
    background: #1f2937;
    border-color: #374151;
    color: #e5e7eb;
  }

  .analyze-button {
    background-color: #059669;
    border-color: #059669;
    color: white;
  }

  .analyze-summary {
    background: #1e3a5f;
    border-color: #2563eb;
  }

  .segment-card {
    background: #1f2937;
    border-color: #374151;
  }

  .segment-card:hover {
    border-color: #60a5fa;
  }

  .score.primary {
    color: #60a5fa;
  }
}

/* Segment Selection */
.segments-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}

.segments-header h3 {
  margin: 0;
}

.selection-controls {
  display: flex;
  align-items: center;
  gap: 12px;
}

.selection-count {
  font-size: 14px;
  color: #6b7280;
}

.secondary-button {
  padding: 6px 12px;
  font-size: 13px;
  background: #374151;
  border: 1px solid #4b5563;
  border-radius: 4px;
  color: #e5e7eb;
  cursor: pointer;
}

.secondary-button:hover {
  background: #4b5563;
}

.segment-select {
  display: flex;
  align-items: center;
  padding-right: 8px;
}

.segment-select input[type="checkbox"] {
  width: 18px;
  height: 18px;
  cursor: pointer;
}

.segment-card.selected {
  border-color: #10b981;
  background: rgba(16, 185, 129, 0.1);
}

/* Highlight Reel Panel */
.highlight-reel-panel {
  margin-top: 24px;
  padding: 20px;
  background: #1e3a5f;
  border: 1px solid #2563eb;
  border-radius: 8px;
}

.highlight-reel-panel h3 {
  margin: 0 0 16px 0;
  color: #93c5fd;
}

.highlight-options {
  display: flex;
  align-items: center;
  gap: 16px;
  flex-wrap: wrap;
}

.style-selector {
  display: flex;
  align-items: center;
  gap: 8px;
}

.style-selector label {
  font-size: 14px;
  color: #9ca3af;
}

.style-selector select {
  padding: 8px 12px;
  border-radius: 4px;
  background: #1f2937;
  border: 1px solid #374151;
  color: #e5e7eb;
  font-size: 14px;
}

.generate-button {
  padding: 10px 20px;
  background: #059669;
  border: none;
  border-radius: 6px;
  color: white;
  font-size: 15px;
  font-weight: 500;
  cursor: pointer;
}

.generate-button:hover:not(:disabled) {
  background: #047857;
}

.generate-button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.python-warning {
  font-size: 13px;
  color: #fbbf24;
  margin: 0;
}

/* Highlight Editor Section */
.highlight-editor-section {
  padding: 20px 0;
}

.highlight-summary {
  display: flex;
  gap: 24px;
  padding: 16px 20px;
  background: #1e3a5f;
  border: 1px solid #2563eb;
  border-radius: 8px;
  margin-bottom: 24px;
}

.highlight-summary p {
  margin: 0;
  font-size: 14px;
}

.reorder-note {
  color: #fbbf24;
  font-style: italic;
}

/* Timeline Editor */
.timeline-editor {
  background: #111827;
  border: 1px solid #374151;
  border-radius: 8px;
  padding: 20px;
  margin-bottom: 24px;
}

.timeline-editor h3 {
  margin: 0 0 8px 0;
}

.timeline-help {
  font-size: 13px;
  color: #6b7280;
  margin: 0 0 16px 0;
}

.timeline-clips {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.timeline-clip {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 12px;
  background: #1f2937;
  border: 1px solid #374151;
  border-radius: 6px;
}

.clip-controls {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.reorder-btn {
  width: 24px;
  height: 24px;
  padding: 0;
  background: #374151;
  border: 1px solid #4b5563;
  border-radius: 4px;
  color: #9ca3af;
  cursor: pointer;
  font-size: 12px;
}

.reorder-btn:hover:not(:disabled) {
  background: #4b5563;
  color: #e5e7eb;
}

.reorder-btn:disabled {
  opacity: 0.3;
  cursor: not-allowed;
}

.clip-thumbnail {
  width: 120px;
  height: 68px;
  border-radius: 4px;
  overflow: hidden;
  background: #374151;
  flex-shrink: 0;
  cursor: pointer;
}

.clip-thumbnail img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.clip-info {
  flex: 1;
  min-width: 150px;
}

.clip-number {
  font-weight: 600;
  font-size: 15px;
  margin-bottom: 4px;
}

.clip-timing {
  font-size: 13px;
  color: #9ca3af;
}

.clip-duration {
  margin-left: 8px;
  color: #6b7280;
}

.clip-transition {
  display: flex;
  align-items: center;
  gap: 8px;
  min-width: 200px;
}

.clip-transition label {
  font-size: 13px;
  color: #6b7280;
}

.clip-transition select {
  padding: 6px 10px;
  border-radius: 4px;
  background: #374151;
  border: 1px solid #4b5563;
  color: #e5e7eb;
  font-size: 13px;
}

.confidence {
  font-size: 12px;
  color: #10b981;
  cursor: help;
}

.remove-btn {
  width: 28px;
  height: 28px;
  padding: 0;
  background: #7f1d1d;
  border: none;
  border-radius: 4px;
  color: #fca5a5;
  cursor: pointer;
  font-weight: bold;
}

.remove-btn:hover {
  background: #991b1b;
  color: #fee2e2;
}

/* Render Controls */
.render-controls {
  display: flex;
  justify-content: center;
  padding: 20px;
}

.render-button {
  padding: 14px 32px;
  background: linear-gradient(135deg, #059669 0%, #0d9488 100%);
  border: none;
  border-radius: 8px;
  color: white;
  font-size: 16px;
  font-weight: 600;
  cursor: pointer;
  box-shadow: 0 4px 12px rgba(5, 150, 105, 0.3);
}

.render-button:hover:not(:disabled) {
  transform: translateY(-1px);
  box-shadow: 0 6px 16px rgba(5, 150, 105, 0.4);
}

.render-button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
  transform: none;
}

.render-button.source {
  background: linear-gradient(135deg, #2563eb 0%, #3b82f6 100%);
  box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
  margin-left: 12px;
}

.render-button.source:hover:not(:disabled) {
  box-shadow: 0 6px 16px rgba(37, 99, 235, 0.4);
}

/* AI Director Section */
.director-section {
  margin-bottom: 20px;
  padding: 16px;
  background: rgba(16, 185, 129, 0.1);
  border: 1px solid rgba(16, 185, 129, 0.3);
  border-radius: 8px;
}

.director-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
}

.director-header h4 {
  margin: 0;
  color: #10b981;
  font-size: 15px;
}

.link-button {
  background: transparent;
  border: none;
  color: #60a5fa;
  font-size: 13px;
  cursor: pointer;
  padding: 0;
  text-decoration: underline;
}

.link-button:hover {
  color: #93c5fd;
}

.director-controls {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.director-controls textarea {
  width: 100%;
  padding: 12px;
  border-radius: 6px;
  background: #1f2937;
  border: 1px solid #374151;
  color: #e5e7eb;
  font-size: 14px;
  font-family: inherit;
  resize: vertical;
}

.director-controls textarea:focus {
  outline: none;
  border-color: #10b981;
}

.director-controls textarea::placeholder {
  color: #6b7280;
}

.director-options {
  display: flex;
  gap: 16px;
  align-items: center;
}

.director-options label {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 14px;
  color: #9ca3af;
}

.director-options input[type="number"] {
  width: 80px;
  padding: 6px 10px;
  border-radius: 4px;
  background: #1f2937;
  border: 1px solid #374151;
  color: #e5e7eb;
  font-size: 14px;
}

.director-options input[type="number"]:focus {
  outline: none;
  border-color: #10b981;
}

.director-button {
  padding: 12px 24px;
  background: linear-gradient(135deg, #10b981 0%, #059669 100%);
  border: none;
  border-radius: 6px;
  color: white;
  font-size: 15px;
  font-weight: 600;
  cursor: pointer;
  box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
}

.director-button:hover:not(:disabled) {
  transform: translateY(-1px);
  box-shadow: 0 6px 16px rgba(16, 185, 129, 0.4);
}

.director-button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
  transform: none;
}

.director-note {
  margin: 0;
  font-size: 12px;
  color: #6b7280;
  font-style: italic;
}

.director-setup {
  margin: 0;
  font-size: 14px;
  color: #9ca3af;
  line-height: 1.6;
}

.divider {
  display: flex;
  align-items: center;
  margin: 16px 0;
}

.divider::before,
.divider::after {
  content: "";
  flex: 1;
  height: 1px;
  background: #374151;
}

.divider span {
  padding: 0 16px;
  font-size: 13px;
  color: #6b7280;
}

/* Modal Overlay */
.modal-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.7);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
}

.modal-content {
  background: #1f2937;
  border: 1px solid #374151;
  border-radius: 12px;
  padding: 24px;
  max-width: 480px;
  width: 90%;
}

.modal-content h3 {
  margin: 0 0 12px 0;
  color: #e5e7eb;
}

.modal-content p {
  margin: 0 0 12px 0;
  color: #9ca3af;
  font-size: 14px;
  line-height: 1.6;
}

.modal-note {
  font-size: 13px;
}

.modal-note a {
  color: #60a5fa;
}

.modal-note a:hover {
  color: #93c5fd;
}

.modal-content input[type="password"] {
  width: 100%;
  padding: 12px;
  border-radius: 6px;
  background: #111827;
  border: 1px solid #374151;
  color: #e5e7eb;
  font-size: 14px;
  margin-bottom: 16px;
}

.modal-content input[type="password"]:focus {
  outline: none;
  border-color: #2563eb;
}

.modal-actions {
  display: flex;
  justify-content: flex-end;
  gap: 12px;
}

.primary-button {
  padding: 10px 20px;
  background: #2563eb;
  border: none;
  border-radius: 6px;
  color: white;
  font-size: 14px;
  font-weight: 500;
  cursor: pointer;
}

.primary-button:hover:not(:disabled) {
  background: #1d4ed8;
}

.primary-button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
```

## src/App.tsx

```tsx
import { useEffect, useState, useCallback, useMemo } from "react";
import { invoke, convertFileSrc } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import { DirectorInput } from "./DirectorInput";
import "./App.css";

interface ClipInfo {
  filename: string;
  source_path: string;
  srt_path: string | null;
  lrf_path: string | null;
  duration_sec: number | null;
  resolution: string | null;
  framerate: number | null;
}

interface IngestResult {
  flight_id: string;
  clips_count: number;
  lrf_used: number;
  proxies_generated: number;
}

interface AnalyzeResult {
  clip_id: string;
  segments_created: number;
  top_score: number;
}

interface Flight {
  id: string;
  name: string;
  import_date: string;
  source_path: string;
  total_clips: number | null;
}

interface SourceClip {
  id: string;
  flight_id: string;
  filename: string;
  source_path: string;
  proxy_path: string | null;
  srt_path: string | null;
  duration_sec: number | null;
}

interface Segment {
  id: string;
  source_clip_id: string;
  start_time_ms: number;
  end_time_ms: number;
  duration_ms: number;
  thumbnail_path: string | null;
  motion_magnitude: number | null;
  gimbal_smoothness: number | null;
  gps_speed_avg: number | null;
  is_selected: boolean;
}

interface SegmentWithScores {
  segment: Segment;
  scores: Record<string, number>;
}

interface ProfileInfo {
  id: string;
  name: string;
  description: string;
}

interface SegmentWithClip {
  segment: Segment;
  clip_id: string;
  clip_filename: string;
  proxy_path: string | null;
  source_path: string;
}

interface ExportResult {
  output_path: string;
  duration_sec: number;
}

interface RenderResult {
  output_path: string;
  duration_sec: number;
  clips_count: number;
}

interface RenderClipInput {
  segment_id: string;
  adjusted_start_ms: number;
  adjusted_end_ms: number;
  transition_type: string;
  transition_duration_ms: number;
}

interface EditDecision {
  clip_id: string;
  sequence_order: number;
  adjusted_start_ms: number;
  adjusted_end_ms: number;
  transition_type: string;
  transition_duration_ms: number;
  confidence: number;
  reasoning: string;
}

interface EditSequence {
  decisions: EditDecision[];
  total_duration_ms: number;
  style: string;
  was_reordered: boolean;
}

type View = "import" | "library" | "flight" | "analyze" | "highlight";

function App() {
  const [initialized, setInitialized] = useState(false);
  const [currentView, setCurrentView] = useState<View>("import");
  const [flights, setFlights] = useState<Flight[]>([]);
  const [scannedClips, setScannedClips] = useState<ClipInfo[]>([]);
  const [selectedFolder, setSelectedFolder] = useState<string | null>(null);
  const [isIngesting, setIsIngesting] = useState(false);
  const [ingestResult, setIngestResult] = useState<IngestResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Flight detail view state
  const [selectedFlight, setSelectedFlight] = useState<Flight | null>(null);
  const [flightClips, setFlightClips] = useState<SourceClip[]>([]);

  // Analysis state
  const [profiles, setProfiles] = useState<ProfileInfo[]>([]);
  const [selectedProfile, setSelectedProfile] = useState<string>("discovery");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analyzeResults, setAnalyzeResults] = useState<AnalyzeResult[]>([]);
  const [topSegments, setTopSegments] = useState<SegmentWithScores[]>([]);

  // Preview state
  const [previewSegment, setPreviewSegment] = useState<SegmentWithClip | null>(null);
  const [isExporting, setIsExporting] = useState(false);

  // Highlight reel state
  const [selectedSegments, setSelectedSegments] = useState<Set<string>>(new Set());
  const [editSequence, setEditSequence] = useState<EditSequence | null>(null);
  const [editStyle, setEditStyle] = useState<string>("cinematic");
  const [isGeneratingSequence, setIsGeneratingSequence] = useState(false);
  const [isRenderingHighlight, setIsRenderingHighlight] = useState(false);
  const [pythonAvailable, setPythonAvailable] = useState<boolean | null>(null);

  // AI Director state
  const [apiKeyConfigured, setApiKeyConfigured] = useState<boolean>(false);
  const [showApiKeyModal, setShowApiKeyModal] = useState(false);
  const [apiKeyInput, setApiKeyInput] = useState("");

  useEffect(() => {
    initApp();
  }, []);

  async function initApp() {
    try {
      await invoke("init_database");
      setInitialized(true);
      await loadFlights();
      await loadProfiles();
      // Check Python availability
      try {
        const available = await invoke<boolean>("check_python_available");
        setPythonAvailable(available);
      } catch {
        setPythonAvailable(false);
      }
      // Check if API key is configured
      try {
        const key = await invoke<string | null>("get_api_key");
        setApiKeyConfigured(!!key);
      } catch {
        setApiKeyConfigured(false);
      }
    } catch (e) {
      setError(`Failed to initialize: ${e}`);
    }
  }

  async function saveApiKey() {
    if (!apiKeyInput.trim()) return;
    try {
      await invoke("save_api_key", { apiKey: apiKeyInput.trim() });
      setApiKeyConfigured(true);
      setShowApiKeyModal(false);
      setApiKeyInput("");
    } catch (e) {
      setError(`Failed to save API key: ${e}`);
    }
  }

  async function clearApiKey() {
    try {
      await invoke("clear_api_key");
      setApiKeyConfigured(false);
    } catch (e) {
      setError(`Failed to clear API key: ${e}`);
    }
  }

  // Memoize director segments to avoid recalculation on every render
  const directorSegments = useMemo(() => {
    return Array.from(selectedSegments).map(id => {
      const s = topSegments.find(ts => ts.segment.id === id);
      if (!s) return null;
      return {
        id: s.segment.id,
        start_ms: s.segment.start_time_ms,
        end_ms: s.segment.end_time_ms,
        thumbnail_path: s.segment.thumbnail_path,
        gimbal_pitch_delta: null,
        gimbal_yaw_delta: null,
        gimbal_smoothness: s.segment.gimbal_smoothness,
        gps_speed: s.segment.gps_speed_avg,
        altitude_delta: null,
        score: s.scores[selectedProfile] || 50,
      };
    }).filter(Boolean) as any[];
  }, [selectedSegments, topSegments, selectedProfile]);

  const handleDirectorSequence = useCallback((sequence: EditSequence) => {
    setEditSequence(sequence);
    setCurrentView("highlight");
  }, []);

  const handleDirectorError = useCallback((errorMsg: string) => {
    setError(errorMsg);
  }, []);

  async function loadFlights() {
    try {
      const result = await invoke<Flight[]>("list_flights");
      setFlights(result);
    } catch (e) {
      setError(`Failed to load flights: ${e}`);
    }
  }

  async function loadProfiles() {
    try {
      const result = await invoke<ProfileInfo[]>("list_profiles");
      setProfiles(result);
      if (result.length > 0 && !result.find((p) => p.id === selectedProfile)) {
        setSelectedProfile(result[0].id);
      }
    } catch (e) {
      console.error("Failed to load profiles:", e);
    }
  }

  async function selectFolder() {
    try {
      const folder = await open({
        directory: true,
        title: "Select DJI Footage Folder",
      });

      if (folder) {
        setSelectedFolder(folder as string);
        setScannedClips([]);
        setIngestResult(null);
        setError(null);

        const clips = await invoke<ClipInfo[]>("scan_folder", {
          folderPath: folder,
        });
        setScannedClips(clips);
      }
    } catch (e) {
      setError(`Failed to scan folder: ${e}`);
    }
  }

  async function startIngest() {
    if (!selectedFolder) return;

    setIsIngesting(true);
    setError(null);

    try {
      const flightName = `Flight ${new Date().toLocaleDateString()}`;
      const result = await invoke<IngestResult>("ingest_folder", {
        folderPath: selectedFolder,
        flightName,
      });

      setIngestResult(result);
      await loadFlights();
    } catch (e) {
      setError(`Ingest failed: ${e}`);
    } finally {
      setIsIngesting(false);
    }
  }

  async function openFlight(flight: Flight) {
    setSelectedFlight(flight);
    setCurrentView("flight");
    setError(null);

    try {
      const clips = await invoke<SourceClip[]>("get_flight_clips", {
        flightId: flight.id,
      });
      setFlightClips(clips);
    } catch (e) {
      setError(`Failed to load clips: ${e}`);
    }
  }

  async function handleDeleteFlight(flightId: string) {
    if (!confirm("Delete this flight and all its data?")) return;

    try {
      await invoke("delete_flight", { flightId });
      await loadFlights();
      if (selectedFlight?.id === flightId) {
        setSelectedFlight(null);
        setCurrentView("library");
      }
    } catch (e) {
      setError(`Failed to delete flight: ${e}`);
    }
  }

  async function analyzeFlight() {
    if (!selectedFlight) return;

    setIsAnalyzing(true);
    setError(null);
    setAnalyzeResults([]);
    setTopSegments([]);

    try {
      const results = await invoke<AnalyzeResult[]>("analyze_flight", {
        flightId: selectedFlight.id,
        profileId: selectedProfile,
      });

      setAnalyzeResults(results);

      // Load top segments
      const segments = await invoke<SegmentWithScores[]>("get_top_segments", {
        flightId: selectedFlight.id,
        profileId: selectedProfile,
        limit: 20,
      });

      setTopSegments(segments);
      setCurrentView("analyze");
    } catch (e) {
      setError(`Analysis failed: ${e}`);
    } finally {
      setIsAnalyzing(false);
    }
  }

  async function openPreview(segmentId: string) {
    try {
      const segmentWithClip = await invoke<SegmentWithClip>("get_segment_with_clip", {
        segmentId,
      });
      setPreviewSegment(segmentWithClip);
    } catch (e) {
      setError(`Failed to load segment: ${e}`);
    }
  }

  async function exportSegment(segmentId: string, useSource: boolean) {
    const { save } = await import("@tauri-apps/plugin-dialog");

    const outputPath = await save({
      title: "Export Segment",
      filters: [{ name: "Video", extensions: ["mp4"] }],
      defaultPath: `segment_${segmentId.slice(0, 8)}.mp4`,
    });

    if (!outputPath) return;

    setIsExporting(true);
    try {
      const result = await invoke<ExportResult>("export_segment", {
        segmentId,
        outputPath,
        useSource,
      });
      alert(`Exported ${result.duration_sec.toFixed(1)}s clip to:\n${result.output_path}`);
    } catch (e) {
      setError(`Export failed: ${e}`);
    } finally {
      setIsExporting(false);
    }
  }

  function toggleSegmentSelection(segmentId: string) {
    setSelectedSegments((prev) => {
      const newSet = new Set(prev);
      if (newSet.has(segmentId)) {
        newSet.delete(segmentId);
      } else {
        newSet.add(segmentId);
      }
      return newSet;
    });
    // Clear edit sequence when selection changes
    setEditSequence(null);
  }

  function selectAllSegments() {
    setSelectedSegments(new Set(topSegments.map((s) => s.segment.id)));
    setEditSequence(null);
  }

  function clearSelection() {
    setSelectedSegments(new Set());
    setEditSequence(null);
  }

  async function generateEditSequence() {
    if (selectedSegments.size < 2) {
      setError("Select at least 2 segments to create a highlight reel");
      return;
    }

    setIsGeneratingSequence(true);
    setError(null);

    try {
      const segmentIds = Array.from(selectedSegments);
      const sequence = await invoke<EditSequence>("generate_edit_sequence", {
        segmentIds,
        style: editStyle,
        reorder: true,
      });
      setEditSequence(sequence);
      setCurrentView("highlight");
    } catch (e) {
      setError(`Failed to generate edit sequence: ${e}`);
    } finally {
      setIsGeneratingSequence(false);
    }
  }

  function updateTransitionType(index: number, newType: string) {
    if (!editSequence) return;
    const newDecisions = [...editSequence.decisions];
    newDecisions[index] = { ...newDecisions[index], transition_type: newType };
    setEditSequence({ ...editSequence, decisions: newDecisions });
  }

  function moveClipUp(index: number) {
    if (!editSequence || index === 0) return;
    const newDecisions = [...editSequence.decisions];
    [newDecisions[index - 1], newDecisions[index]] = [newDecisions[index], newDecisions[index - 1]];
    // Update sequence orders
    newDecisions.forEach((d, i) => (d.sequence_order = i));
    setEditSequence({ ...editSequence, decisions: newDecisions });
  }

  function moveClipDown(index: number) {
    if (!editSequence || index >= editSequence.decisions.length - 1) return;
    const newDecisions = [...editSequence.decisions];
    [newDecisions[index], newDecisions[index + 1]] = [newDecisions[index + 1], newDecisions[index]];
    // Update sequence orders
    newDecisions.forEach((d, i) => (d.sequence_order = i));
    setEditSequence({ ...editSequence, decisions: newDecisions });
  }

  function removeFromSequence(index: number) {
    if (!editSequence) return;
    const newDecisions = editSequence.decisions.filter((_, i) => i !== index);
    newDecisions.forEach((d, i) => (d.sequence_order = i));
    const newDuration = newDecisions.reduce(
      (sum, d) => sum + (d.adjusted_end_ms - d.adjusted_start_ms),
      0
    );
    setEditSequence({ ...editSequence, decisions: newDecisions, total_duration_ms: newDuration });
  }

  async function renderHighlightReel(useSource: boolean = false) {
    if (!editSequence || editSequence.decisions.length === 0) return;

    const { save } = await import("@tauri-apps/plugin-dialog");

    const outputPath = await save({
      title: "Save Highlight Reel",
      filters: [{ name: "Video", extensions: ["mp4"] }],
      defaultPath: `highlight_${editStyle}_${Date.now()}.mp4`,
    });

    if (!outputPath) return;

    setIsRenderingHighlight(true);
    setError(null);

    try {
      // Build clips array from edit sequence decisions
      const clips: RenderClipInput[] = editSequence.decisions.map((decision) => ({
        segment_id: decision.clip_id,
        adjusted_start_ms: decision.adjusted_start_ms,
        adjusted_end_ms: decision.adjusted_end_ms,
        transition_type: decision.transition_type,
        transition_duration_ms: decision.transition_duration_ms,
      }));

      const result = await invoke<RenderResult>("render_highlight_reel", {
        clips,
        outputPath,
        useSource,
      });

      alert(
        `Highlight reel exported!\n\n` +
        `Duration: ${result.duration_sec.toFixed(1)}s\n` +
        `Clips: ${result.clips_count}\n` +
        `Location: ${result.output_path}`
      );
    } catch (e) {
      setError(`Failed to render highlight reel: ${e}`);
    } finally {
      setIsRenderingHighlight(false);
    }
  }

  function getSegmentById(id: string): SegmentWithScores | undefined {
    return topSegments.find((s) => s.segment.id === id);
  }

  function formatDuration(sec: number | null): string {
    if (!sec) return "--:--";
    const mins = Math.floor(sec / 60);
    const secs = Math.floor(sec % 60);
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  }

  function formatTimeMs(ms: number): string {
    const totalSec = Math.floor(ms / 1000);
    const mins = Math.floor(totalSec / 60);
    const secs = totalSec % 60;
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  }

  if (!initialized) {
    return (
      <main className="container">
        <h1>SkyClip</h1>
        <p>Initializing...</p>
        {error && <p className="error">{error}</p>}
      </main>
    );
  }

  return (
    <main className="container">
      <header>
        <h1>SkyClip</h1>
        <p className="tagline">Drone footage analysis and highlight extraction</p>
        <nav className="nav-tabs">
          <button
            className={currentView === "import" ? "active" : ""}
            onClick={() => setCurrentView("import")}
          >
            Import
          </button>
          <button
            className={currentView === "library" ? "active" : ""}
            onClick={() => setCurrentView("library")}
          >
            Library ({flights.length})
          </button>
          {selectedFlight && (
            <button
              className={currentView === "flight" || currentView === "analyze" ? "active" : ""}
              onClick={() => setCurrentView("flight")}
            >
              {selectedFlight.name}
            </button>
          )}
        </nav>
      </header>

      {error && <div className="error-banner">{error}</div>}

      {currentView === "import" && (
        <section className="ingest-section">
          <h2>Import Footage</h2>
          <button onClick={selectFolder} disabled={isIngesting}>
            Select DJI Folder
          </button>

          {selectedFolder && (
            <div className="folder-info">
              <p>
                <strong>Selected:</strong> {selectedFolder}
              </p>
              <p>
                <strong>Clips found:</strong> {scannedClips.length}
              </p>
            </div>
          )}

          {scannedClips.length > 0 && (
            <>
              <table className="clips-table">
                <thead>
                  <tr>
                    <th>Filename</th>
                    <th>Duration</th>
                    <th>Resolution</th>
                    <th>FPS</th>
                    <th>SRT</th>
                    <th>LRF</th>
                  </tr>
                </thead>
                <tbody>
                  {scannedClips.map((clip) => (
                    <tr key={clip.source_path}>
                      <td>{clip.filename}</td>
                      <td>{formatDuration(clip.duration_sec)}</td>
                      <td>{clip.resolution || "--"}</td>
                      <td>{clip.framerate?.toFixed(1) || "--"}</td>
                      <td>{clip.srt_path ? "Yes" : "No"}</td>
                      <td>{clip.lrf_path ? "Yes" : "No"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>

              <button
                onClick={startIngest}
                disabled={isIngesting}
                className="ingest-button"
              >
                {isIngesting ? "Importing..." : "Import Footage"}
              </button>
            </>
          )}

          {ingestResult && (
            <div className="ingest-result">
              <h3>Import Complete</h3>
              <p>Clips imported: {ingestResult.clips_count}</p>
              <p>LRF proxies used: {ingestResult.lrf_used}</p>
              <p>Proxies generated: {ingestResult.proxies_generated}</p>
            </div>
          )}
        </section>
      )}

      {currentView === "library" && (
        <section className="library-section">
          <h2>Library</h2>
          {flights.length === 0 ? (
            <p className="empty-state">No flights imported yet</p>
          ) : (
            <ul className="flights-list">
              {flights.map((flight) => (
                <li key={flight.id}>
                  <div className="flight-info" onClick={() => openFlight(flight)}>
                    <strong>{flight.name}</strong>
                    <span className="flight-path">{flight.source_path}</span>
                    <span className="flight-meta">
                      {flight.total_clips} clips &bull;{" "}
                      {new Date(flight.import_date).toLocaleDateString()}
                    </span>
                  </div>
                  <button
                    className="delete-button"
                    onClick={(e) => {
                      e.stopPropagation();
                      handleDeleteFlight(flight.id);
                    }}
                  >
                    Delete
                  </button>
                </li>
              ))}
            </ul>
          )}
        </section>
      )}

      {currentView === "flight" && selectedFlight && (
        <section className="flight-detail-section">
          <h2>{selectedFlight.name}</h2>
          <p className="flight-meta">
            {flightClips.length} clips &bull;{" "}
            {new Date(selectedFlight.import_date).toLocaleDateString()}
          </p>

          <div className="analyze-controls">
            <h3>Analyze Footage</h3>
            <div className="profile-selector">
              <label>Profile:</label>
              <select
                value={selectedProfile}
                onChange={(e) => setSelectedProfile(e.target.value)}
                disabled={isAnalyzing}
              >
                {profiles.map((profile) => (
                  <option key={profile.id} value={profile.id}>
                    {profile.name}
                  </option>
                ))}
              </select>
              {profiles.find((p) => p.id === selectedProfile) && (
                <span className="profile-desc">
                  {profiles.find((p) => p.id === selectedProfile)?.description}
                </span>
              )}
            </div>
            <button
              onClick={analyzeFlight}
              disabled={isAnalyzing}
              className="analyze-button"
            >
              {isAnalyzing ? "Analyzing..." : "Analyze Flight"}
            </button>
          </div>

          <h3>Clips</h3>
          <table className="clips-table">
            <thead>
              <tr>
                <th>Filename</th>
                <th>Duration</th>
                <th>SRT</th>
                <th>Proxy</th>
              </tr>
            </thead>
            <tbody>
              {flightClips.map((clip) => (
                <tr key={clip.id}>
                  <td>{clip.filename}</td>
                  <td>{formatDuration(clip.duration_sec)}</td>
                  <td>{clip.srt_path ? "Yes" : "No"}</td>
                  <td>{clip.proxy_path ? "Ready" : "Missing"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}

      {currentView === "analyze" && selectedFlight && (
        <section className="analyze-results-section">
          <div className="section-header">
            <h2>Analysis Results</h2>
            <button onClick={() => setCurrentView("flight")} className="back-button">
              Back to Flight
            </button>
          </div>

          <div className="analyze-summary">
            <p>
              <strong>Profile:</strong> {profiles.find((p) => p.id === selectedProfile)?.name}
            </p>
            <p>
              <strong>Clips analyzed:</strong> {analyzeResults.length}
            </p>
            <p>
              <strong>Segments found:</strong>{" "}
              {analyzeResults.reduce((sum, r) => sum + r.segments_created, 0)}
            </p>
          </div>

          <div className="segments-header">
            <h3>Top Segments</h3>
            {topSegments.length > 0 && (
              <div className="selection-controls">
                <span className="selection-count">{selectedSegments.size} selected</span>
                <button onClick={selectAllSegments} className="secondary-button">
                  Select All
                </button>
                <button onClick={clearSelection} className="secondary-button">
                  Clear
                </button>
              </div>
            )}
          </div>

          {topSegments.length === 0 ? (
            <p className="empty-state">No segments found matching profile criteria</p>
          ) : (
            <>
              <div className="segments-grid">
                {topSegments.map((item, idx) => (
                  <div
                    key={item.segment.id}
                    className={`segment-card ${selectedSegments.has(item.segment.id) ? "selected" : ""}`}
                  >
                    <div className="segment-select">
                      <input
                        type="checkbox"
                        checked={selectedSegments.has(item.segment.id)}
                        onChange={() => toggleSegmentSelection(item.segment.id)}
                      />
                    </div>
                    <div className="segment-rank">#{idx + 1}</div>
                    <div className="segment-thumbnail" onClick={() => openPreview(item.segment.id)}>
                      {item.segment.thumbnail_path ? (
                        <img
                          src={convertFileSrc(item.segment.thumbnail_path)}
                          alt={`Segment ${idx + 1}`}
                        />
                      ) : (
                        <div className="thumbnail-placeholder">No Preview</div>
                      )}
                    </div>
                    <div className="segment-info" onClick={() => openPreview(item.segment.id)}>
                      <div className="segment-time">
                        {formatTimeMs(item.segment.start_time_ms)} -{" "}
                        {formatTimeMs(item.segment.end_time_ms)}
                      </div>
                      <div className="segment-duration">
                        {(item.segment.duration_ms / 1000).toFixed(1)}s
                      </div>
                    </div>
                    <div className="segment-scores">
                      <div className="score primary">
                        {item.scores[selectedProfile]?.toFixed(0) || "--"}
                      </div>
                      <div className="segment-signals">
                        {item.segment.gimbal_smoothness && (
                          <span title="Gimbal Smoothness">
                            Smooth: {(item.segment.gimbal_smoothness * 100).toFixed(0)}%
                          </span>
                        )}
                        {item.segment.gps_speed_avg && (
                          <span title="GPS Speed">
                            Speed: {item.segment.gps_speed_avg.toFixed(1)} m/s
                          </span>
                        )}
                        {item.segment.motion_magnitude && (
                          <span title="Motion">
                            Motion: {item.segment.motion_magnitude.toFixed(1)}
                          </span>
                        )}
                      </div>
                    </div>
                    <div className="segment-actions">
                      <button
                        onClick={() => exportSegment(item.segment.id, false)}
                        disabled={isExporting}
                        className="export-button"
                      >
                        Export (Quick)
                      </button>
                      <button
                        onClick={() => exportSegment(item.segment.id, true)}
                        disabled={isExporting}
                        className="export-button source"
                      >
                        Export (4K)
                      </button>
                    </div>
                  </div>
                ))}
              </div>

              {selectedSegments.size >= 2 && (
                <div className="highlight-reel-panel">
                  <h3>Create Highlight Reel</h3>

                  {/* AI Director Mode */}
                  <div className="director-section">
                    <div className="director-header">
                      <h4>AI Director Mode</h4>
                      {apiKeyConfigured ? (
                        <button onClick={clearApiKey} className="link-button">
                          Remove API Key
                        </button>
                      ) : (
                        <button onClick={() => setShowApiKeyModal(true)} className="link-button">
                          Add API Key
                        </button>
                      )}
                    </div>

                    {apiKeyConfigured ? (
                      <DirectorInput
                        segments={directorSegments}
                        onSequenceGenerated={handleDirectorSequence}
                        onError={handleDirectorError}
                      />
                    ) : (
                      <p className="director-setup">
                        Add your Anthropic API key to enable AI-directed editing. The AI will see your clip thumbnails and telemetry data to make intelligent edit decisions.
                      </p>
                    )}
                  </div>

                  <div className="divider">
                    <span>or use presets</span>
                  </div>

                  <div className="highlight-options">
                    <div className="style-selector">
                      <label>Edit Style:</label>
                      <select
                        value={editStyle}
                        onChange={(e) => setEditStyle(e.target.value)}
                        disabled={isGeneratingSequence}
                      >
                        <option value="cinematic">Cinematic (smooth, longer takes)</option>
                        <option value="action">Action (fast cuts, high energy)</option>
                        <option value="social">Social (short, punchy)</option>
                      </select>
                    </div>
                    <button
                      onClick={generateEditSequence}
                      disabled={isGeneratingSequence || selectedSegments.size < 2}
                      className="generate-button"
                    >
                      {isGeneratingSequence ? "Generating..." : `Generate with Preset (${selectedSegments.size} clips)`}
                    </button>
                    {pythonAvailable === false && (
                      <p className="python-warning">
                        Python with OpenCV not detected. Using basic transitions.
                      </p>
                    )}
                  </div>
                </div>
              )}
            </>
          )}
        </section>
      )}

      {currentView === "highlight" && editSequence && (
        <section className="highlight-editor-section">
          <div className="section-header">
            <h2>Highlight Reel Editor</h2>
            <button onClick={() => setCurrentView("analyze")} className="back-button">
              Back to Segments
            </button>
          </div>

          <div className="highlight-summary">
            <p>
              <strong>Style:</strong> {editSequence.style}
            </p>
            <p>
              <strong>Total Duration:</strong> {(editSequence.total_duration_ms / 1000).toFixed(1)}s
            </p>
            <p>
              <strong>Clips:</strong> {editSequence.decisions.length}
            </p>
            {editSequence.was_reordered && (
              <p className="reorder-note">Clips were reordered for better flow</p>
            )}
          </div>

          <div className="timeline-editor">
            <h3>Timeline</h3>
            <p className="timeline-help">
              Drag to reorder, change transitions, or remove clips. AI suggestions shown with confidence %.
            </p>

            <div className="timeline-clips">
              {editSequence.decisions.map((decision, idx) => {
                const segment = getSegmentById(decision.clip_id);
                return (
                  <div key={decision.clip_id} className="timeline-clip">
                    <div className="clip-controls">
                      <button
                        onClick={() => moveClipUp(idx)}
                        disabled={idx === 0}
                        className="reorder-btn"
                        title="Move up"
                      >
                        ^
                      </button>
                      <button
                        onClick={() => moveClipDown(idx)}
                        disabled={idx === editSequence.decisions.length - 1}
                        className="reorder-btn"
                        title="Move down"
                      >
                        v
                      </button>
                    </div>

                    <div className="clip-thumbnail">
                      {segment?.segment.thumbnail_path ? (
                        <img
                          src={convertFileSrc(segment.segment.thumbnail_path)}
                          alt={`Clip ${idx + 1}`}
                          onClick={() => openPreview(decision.clip_id)}
                        />
                      ) : (
                        <div className="thumbnail-placeholder">Preview</div>
                      )}
                    </div>

                    <div className="clip-info">
                      <div className="clip-number">Clip {idx + 1}</div>
                      <div className="clip-timing">
                        {formatTimeMs(decision.adjusted_start_ms)} - {formatTimeMs(decision.adjusted_end_ms)}
                        <span className="clip-duration">
                          ({((decision.adjusted_end_ms - decision.adjusted_start_ms) / 1000).toFixed(1)}s)
                        </span>
                      </div>
                    </div>

                    <div className="clip-transition">
                      {idx < editSequence.decisions.length - 1 && (
                        <>
                          <label>Transition:</label>
                          <select
                            value={decision.transition_type}
                            onChange={(e) => updateTransitionType(idx, e.target.value)}
                          >
                            <option value="cut">Hard Cut</option>
                            <option value="dissolve">Dissolve</option>
                            <option value="dip_black">Dip to Black</option>
                          </select>
                          <span className="confidence" title={decision.reasoning}>
                            {(decision.confidence * 100).toFixed(0)}% confident
                          </span>
                        </>
                      )}
                    </div>

                    <button
                      onClick={() => removeFromSequence(idx)}
                      className="remove-btn"
                      title="Remove from highlight"
                    >
                      X
                    </button>
                  </div>
                );
              })}
            </div>
          </div>

          <div className="render-controls">
            <button
              onClick={() => renderHighlightReel(false)}
              disabled={isRenderingHighlight || editSequence.decisions.length === 0}
              className="render-button"
            >
              {isRenderingHighlight ? "Rendering..." : "Export (Quick)"}
            </button>
            <button
              onClick={() => renderHighlightReel(true)}
              disabled={isRenderingHighlight || editSequence.decisions.length === 0}
              className="render-button source"
            >
              {isRenderingHighlight ? "Rendering..." : "Export (4K Source)"}
            </button>
          </div>
        </section>
      )}

      {/* Preview Modal */}
      {previewSegment && (
        <div className="preview-modal" onClick={() => setPreviewSegment(null)}>
          <div className="preview-content" onClick={(e) => e.stopPropagation()}>
            <div className="preview-header">
              <h3>Preview: {previewSegment.clip_filename}</h3>
              <button onClick={() => setPreviewSegment(null)} className="close-button">
                Close
              </button>
            </div>
            <video
              controls
              autoPlay
              src={convertFileSrc(previewSegment.proxy_path || previewSegment.source_path)}
              style={{ maxWidth: "100%", maxHeight: "60vh" }}
              onLoadedMetadata={(e) => {
                const video = e.currentTarget;
                video.currentTime = previewSegment.segment.start_time_ms / 1000;
              }}
              onTimeUpdate={(e) => {
                const video = e.currentTarget;
                const endTime = previewSegment.segment.end_time_ms / 1000;
                if (video.currentTime >= endTime) {
                  video.pause();
                  video.currentTime = previewSegment.segment.start_time_ms / 1000;
                }
              }}
            />
            <div className="preview-info">
              <p>
                Segment: {formatTimeMs(previewSegment.segment.start_time_ms)} -{" "}
                {formatTimeMs(previewSegment.segment.end_time_ms)} (
                {(previewSegment.segment.duration_ms / 1000).toFixed(1)}s)
              </p>
              <p className="preview-note">
                Preview loops the selected segment. Use scrubber to explore full clip.
              </p>
            </div>
            <div className="preview-actions">
              <button
                onClick={() => exportSegment(previewSegment.segment.id, false)}
                disabled={isExporting}
                className="export-button"
              >
                {isExporting ? "Exporting..." : "Export (Quick)"}
              </button>
              <button
                onClick={() => exportSegment(previewSegment.segment.id, true)}
                disabled={isExporting}
                className="export-button source"
              >
                {isExporting ? "Exporting..." : "Export (4K Source)"}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* API Key Modal */}
      {showApiKeyModal && (
        <div className="modal-overlay" onClick={() => setShowApiKeyModal(false)}>
          <div className="modal-content api-key-modal" onClick={(e) => e.stopPropagation()}>
            <h3>Add Anthropic API Key</h3>
            <p>
              Enter your Anthropic API key to enable AI Director mode. Your key is stored locally on your machine.
            </p>
            <p className="modal-note">
              Get your API key at{" "}
              <a href="https://console.anthropic.com/settings/keys" target="_blank" rel="noopener noreferrer">
                console.anthropic.com
              </a>
            </p>
            <input
              type="password"
              placeholder="sk-ant-..."
              value={apiKeyInput}
              onChange={(e) => setApiKeyInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && saveApiKey()}
            />
            <div className="modal-actions">
              <button onClick={() => setShowApiKeyModal(false)} className="secondary-button">
                Cancel
              </button>
              <button onClick={saveApiKey} disabled={!apiKeyInput.trim()} className="primary-button">
                Save API Key
              </button>
            </div>
          </div>
        </div>
      )}
    </main>
  );
}

export default App;
```

## src/DirectorInput.tsx

```tsx
import { useState } from "react";
import { invoke } from "@tauri-apps/api/core";

interface SegmentData {
  id: string;
  start_ms: number;
  end_ms: number;
  thumbnail_path: string | null;
  gimbal_pitch_delta: number | null;
  gimbal_yaw_delta: number | null;
  gimbal_smoothness: number | null;
  gps_speed: number | null;
  altitude_delta: number | null;
  score: number;
}

interface EditSequence {
  decisions: any[];
  total_duration_ms: number;
  style: string;
  was_reordered: boolean;
}

interface DirectorInputProps {
  segments: SegmentData[];
  onSequenceGenerated: (sequence: EditSequence) => void;
  onError: (error: string) => void;
  disabled?: boolean;
}

export function DirectorInput({ segments, onSequenceGenerated, onError, disabled }: DirectorInputProps) {
  const [prompt, setPrompt] = useState("");
  const [targetDuration, setTargetDuration] = useState<string>("");
  const [isGenerating, setIsGenerating] = useState(false);

  async function handleGenerate() {
    console.log("Director: handleGenerate called", { prompt, segmentsCount: segments.length });

    if (!prompt.trim() || segments.length < 2) {
      console.log("Director: Early return - prompt empty or not enough segments");
      return;
    }

    setIsGenerating(true);
    console.log("Director: Calling API...");

    try {
      const sequence = await invoke<EditSequence>("director_generate_edit", {
        prompt: prompt.trim(),
        segments,
        targetDurationSec: targetDuration ? parseInt(targetDuration) : null,
      });

      console.log("Director: Got sequence", sequence);
      onSequenceGenerated(sequence);
      setPrompt("");
    } catch (e) {
      console.error("Director: Error", e);
      onError(`AI Director failed: ${e}`);
    } finally {
      setIsGenerating(false);
    }
  }

  return (
    <div className="director-controls">
      <textarea
        placeholder="Describe your vision... e.g., 'Make a 30-second dramatic sunset reveal, start with an establishing shot, build to the most exciting moment, end on a calm beach scene. Cinematic feel with smooth transitions.'"
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
        disabled={isGenerating || disabled}
        rows={3}
      />
      <div className="director-options">
        <label>
          Target duration:
          <input
            type="number"
            placeholder="Auto"
            value={targetDuration}
            onChange={(e) => setTargetDuration(e.target.value)}
            disabled={isGenerating || disabled}
            min={5}
            max={300}
          />
          <span>seconds</span>
        </label>
      </div>
      <button
        onClick={() => {
          console.log("Director: Button clicked!");
          handleGenerate();
        }}
        disabled={isGenerating || !prompt.trim() || segments.length < 2 || disabled}
        className="director-button"
      >
        {isGenerating ? "AI is thinking..." : `Ask AI Director (${segments.length} clips)`}
      </button>
      <p className="director-note">
        Uses Claude API (~$0.07-0.25 per request depending on # of clips)
      </p>
    </div>
  );
}
```

## src/main.tsx

```tsx
import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
```

## src-tauri/build.rs

```rust
fn main() {
    tauri_build::build()
}
```

## src-tauri/src/commands/analyze.rs

```rust
use crate::commands::ingest::AppState;
use crate::models::Segment;
use crate::services::{ScoreCalculator, SrtParser, TelemetryAnalyzer};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use tauri::State;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyzeResult {
    pub clip_id: String,
    pub segments_created: u32,
    pub top_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentWithScores {
    pub segment: Segment,
    pub scores: HashMap<String, f64>,
}

/// Analyze a single clip and generate scored segments
#[tauri::command]
pub async fn analyze_clip(
    state: State<'_, AppState>,
    clip_id: String,
    profile_id: Option<String>,
) -> Result<AnalyzeResult, String> {
    let db_guard = state.db.lock().await;
    let db = db_guard.as_ref().ok_or("Database not initialized")?;
    let app_data_dir = state.app_data_dir.lock().await.clone();

    // Get the clip
    let clip = db
        .get_clip(&clip_id)
        .await
        .map_err(|e| e.to_string())?
        .ok_or("Clip not found")?;

    // Check for SRT file
    let srt_path = clip.srt_path.ok_or("Clip has no SRT telemetry file")?;

    // Parse telemetry
    let parser = SrtParser::new();
    let frames = parser.parse_file(&srt_path).map_err(|e| e.to_string())?;

    if frames.is_empty() {
        return Err("No telemetry frames found in SRT file".to_string());
    }

    // Initialize analyzer and score calculator
    let analyzer = TelemetryAnalyzer::new();
    let mut score_calc = ScoreCalculator::new();

    // Load profiles
    let profiles_dir = app_data_dir.join("profiles");
    if profiles_dir.exists() {
        score_calc
            .load_profiles_from_dir(&profiles_dir)
            .map_err(|e| e.to_string())?;
    }

    // Also try loading from bundled profiles
    let bundled_profiles = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map(|p| p.join("profiles"));
    if let Some(bundled) = bundled_profiles {
        if bundled.exists() {
            let _ = score_calc.load_profiles_from_dir(&bundled);
        }
    }

    // If no profiles loaded, use default discovery profile
    if score_calc.get_profiles().is_empty() {
        score_calc
            .load_profile(DEFAULT_DISCOVERY_PROFILE)
            .map_err(|e| e.to_string())?;
    }

    // Detect segments based on profile thresholds
    let active_profile = profile_id.as_deref().unwrap_or("discovery");
    let (min_dur, max_dur) = match score_calc.get_profile(active_profile) {
        Some(p) => (
            p.thresholds.min_duration_sec.unwrap_or(5.0),
            p.thresholds.max_duration_sec.unwrap_or(30.0),
        ),
        None => (5.0, 30.0),
    };

    let segment_indices = analyzer.detect_segments(&frames, min_dur, max_dur);

    // Delete existing segments for this clip
    db.delete_segments_for_clip(&clip_id)
        .await
        .map_err(|e| e.to_string())?;

    let thumbnails_dir = app_data_dir.join("thumbnails").join(&clip_id);
    let mut segments_created = 0u32;
    let mut top_score = 0.0f64;

    for (start_idx, end_idx) in segment_indices {
        let segment_frames = &frames[start_idx..end_idx];
        if segment_frames.is_empty() {
            continue;
        }

        let start_time_ms = segment_frames.first().map(|f| f.start_time_ms).unwrap_or(0);
        let end_time_ms = segment_frames.last().map(|f| f.end_time_ms).unwrap_or(0);
        let duration_sec = (end_time_ms - start_time_ms) as f64 / 1000.0;

        // Analyze segment
        let signals = analyzer.analyze_frames(segment_frames);

        // Calculate scores for all profiles
        let scores = score_calc.calculate_all_scores(&signals);

        // Get score for active profile
        let active_score = scores.get(active_profile).copied().unwrap_or(0.0);

        // Check if passes thresholds
        if !score_calc.passes_thresholds(active_profile, &signals, duration_sec) {
            continue;
        }

        // Find thumbnail for this segment (use frame at segment start)
        // FFmpeg outputs 1-indexed thumbnails: thumb_0001.jpg = second 0, thumb_0002.jpg = second 1, etc.
        let thumb_second = (start_time_ms / 1000) as u32 + 1;
        let thumbnail_path = thumbnails_dir.join(format!("thumb_{:04}.jpg", thumb_second));
        let thumbnail_path_str = if thumbnail_path.exists() {
            Some(thumbnail_path.to_string_lossy().to_string())
        } else {
            // Try finding any thumbnail as fallback
            std::fs::read_dir(&thumbnails_dir)
                .ok()
                .and_then(|mut entries| {
                    entries.find_map(|e| {
                        e.ok().and_then(|entry| {
                            let name = entry.file_name().to_string_lossy().to_string();
                            if name.starts_with("thumb_") && name.ends_with(".jpg") {
                                Some(entry.path().to_string_lossy().to_string())
                            } else {
                                None
                            }
                        })
                    })
                })
        };

        // Create segment
        let mut segment = Segment::new(clip_id.clone(), start_time_ms, end_time_ms);
        segment.thumbnail_path = thumbnail_path_str;
        segment.motion_magnitude = Some(signals.motion_magnitude);
        segment.gimbal_pitch_delta_avg = Some(signals.gimbal_pitch_delta_avg);
        segment.gimbal_yaw_delta_avg = Some(signals.gimbal_yaw_delta_avg);
        segment.gimbal_smoothness = Some(signals.gimbal_smoothness);
        segment.altitude_delta = Some(signals.altitude_delta);
        segment.gps_speed_avg = Some(signals.gps_speed_avg);
        segment.iso_avg = Some(signals.iso_avg);

        db.insert_segment(&segment)
            .await
            .map_err(|e| e.to_string())?;

        // Store scores for this segment
        for (profile_id, score) in &scores {
            db.insert_segment_score(&segment.id, profile_id, *score)
                .await
                .map_err(|e| e.to_string())?;
        }

        segments_created += 1;
        if active_score > top_score {
            top_score = active_score;
        }
    }

    Ok(AnalyzeResult {
        clip_id,
        segments_created,
        top_score,
    })
}

/// Analyze all clips in a flight
#[tauri::command]
pub async fn analyze_flight(
    state: State<'_, AppState>,
    flight_id: String,
    profile_id: Option<String>,
) -> Result<Vec<AnalyzeResult>, String> {
    let db_guard = state.db.lock().await;
    let db = db_guard.as_ref().ok_or("Database not initialized")?;

    let clips = db
        .get_clips_for_flight(&flight_id)
        .await
        .map_err(|e| e.to_string())?;

    drop(db_guard); // Release lock before calling analyze_clip

    let mut results = Vec::new();
    for clip in clips {
        if clip.srt_path.is_some() {
            match analyze_clip(state.clone(), clip.id.clone(), profile_id.clone()).await {
                Ok(result) => results.push(result),
                Err(e) => {
                    eprintln!("Failed to analyze clip {}: {}", clip.filename, e);
                }
            }
        }
    }

    Ok(results)
}

/// Get segments for a clip with their scores
#[tauri::command]
pub async fn get_clip_segments(
    state: State<'_, AppState>,
    clip_id: String,
) -> Result<Vec<SegmentWithScores>, String> {
    let db_guard = state.db.lock().await;
    let db = db_guard.as_ref().ok_or("Database not initialized")?;

    let segments = db
        .get_segments_for_clip(&clip_id)
        .await
        .map_err(|e| e.to_string())?;

    let mut results = Vec::new();
    for segment in segments {
        let scores = db
            .get_segment_scores(&segment.id)
            .await
            .map_err(|e| e.to_string())?;
        results.push(SegmentWithScores { segment, scores });
    }

    Ok(results)
}

/// Get top segments across a flight, sorted by score
#[tauri::command]
pub async fn get_top_segments(
    state: State<'_, AppState>,
    flight_id: String,
    profile_id: String,
    limit: Option<u32>,
) -> Result<Vec<SegmentWithScores>, String> {
    let db_guard = state.db.lock().await;
    let db = db_guard.as_ref().ok_or("Database not initialized")?;

    let limit = limit.unwrap_or(20);
    let segments = db
        .get_top_segments_for_flight(&flight_id, &profile_id, limit)
        .await
        .map_err(|e| e.to_string())?;

    let mut results = Vec::new();
    for segment in segments {
        let scores = db
            .get_segment_scores(&segment.id)
            .await
            .map_err(|e| e.to_string())?;
        results.push(SegmentWithScores { segment, scores });
    }

    Ok(results)
}

/// List available profiles
#[tauri::command]
pub async fn list_profiles(state: State<'_, AppState>) -> Result<Vec<ProfileInfo>, String> {
    let app_data_dir = state.app_data_dir.lock().await.clone();
    let mut score_calc = ScoreCalculator::new();

    // Load profiles
    let profiles_dir = app_data_dir.join("profiles");
    if profiles_dir.exists() {
        let _ = score_calc.load_profiles_from_dir(&profiles_dir);
    }

    let bundled_profiles = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map(|p| p.join("profiles"));
    if let Some(bundled) = bundled_profiles {
        if bundled.exists() {
            let _ = score_calc.load_profiles_from_dir(&bundled);
        }
    }

    if score_calc.get_profiles().is_empty() {
        let _ = score_calc.load_profile(DEFAULT_DISCOVERY_PROFILE);
    }

    Ok(score_calc
        .get_profiles()
        .iter()
        .map(|p| ProfileInfo {
            id: p.id.clone(),
            name: p.name.clone(),
            description: p.description.clone(),
        })
        .collect())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileInfo {
    pub id: String,
    pub name: String,
    pub description: String,
}

/// Get a segment with its source clip info (for preview/export)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentWithClip {
    pub segment: Segment,
    pub clip_id: String,
    pub clip_filename: String,
    pub proxy_path: Option<String>,
    pub source_path: String,
}

#[tauri::command]
pub async fn get_segment_with_clip(
    state: State<'_, AppState>,
    segment_id: String,
) -> Result<SegmentWithClip, String> {
    let db_guard = state.db.lock().await;
    let db = db_guard.as_ref().ok_or("Database not initialized")?;

    let segment = db
        .get_segment(&segment_id)
        .await
        .map_err(|e| e.to_string())?
        .ok_or("Segment not found")?;

    let clip = db
        .get_clip(&segment.source_clip_id)
        .await
        .map_err(|e| e.to_string())?
        .ok_or("Clip not found")?;

    Ok(SegmentWithClip {
        segment,
        clip_id: clip.id,
        clip_filename: clip.filename,
        proxy_path: clip.proxy_path,
        source_path: clip.source_path,
    })
}

/// Export a segment to a file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportResult {
    pub output_path: String,
    pub duration_sec: f64,
}

#[tauri::command]
pub async fn export_segment(
    state: State<'_, AppState>,
    segment_id: String,
    output_path: String,
    use_source: bool,
) -> Result<ExportResult, String> {
    use crate::services::FFmpeg;

    let db_guard = state.db.lock().await;
    let db = db_guard.as_ref().ok_or("Database not initialized")?;

    let segment = db
        .get_segment(&segment_id)
        .await
        .map_err(|e| e.to_string())?
        .ok_or("Segment not found")?;

    let clip = db
        .get_clip(&segment.source_clip_id)
        .await
        .map_err(|e| e.to_string())?
        .ok_or("Clip not found")?;

    drop(db_guard);

    let ffmpeg = FFmpeg::new().map_err(|e| e.to_string())?;

    // Use source for quality, proxy for speed
    let input_path = if use_source {
        &clip.source_path
    } else {
        clip.proxy_path.as_ref().unwrap_or(&clip.source_path)
    };

    let start_sec = segment.start_time_ms as f64 / 1000.0;
    let end_sec = segment.end_time_ms as f64 / 1000.0;

    // Use fast export (stream copy) for quick results
    ffmpeg
        .export_fast(input_path, &output_path, start_sec, end_sec)
        .map_err(|e| e.to_string())?;

    Ok(ExportResult {
        output_path,
        duration_sec: (segment.end_time_ms - segment.start_time_ms) as f64 / 1000.0,
    })
}

const DEFAULT_DISCOVERY_PROFILE: &str = r#"{
  "id": "discovery",
  "name": "Discovery",
  "description": "Balanced scoring. Find all potentially good moments.",
  "weights": {
    "gimbal_smoothness": 0.20,
    "gimbal_pitch_delta": 0.15,
    "gimbal_yaw_delta": 0.15,
    "gps_speed": 0.15,
    "motion_magnitude": 0.15,
    "altitude_delta": 0.10,
    "iso_penalty": 0.10
  },
  "thresholds": {
    "min_duration_sec": 3,
    "max_duration_sec": 30
  }
}"#;
```

## src-tauri/src/commands/director.rs

```rust
use crate::services::{Director, SegmentContext, EditSequence, EditDecision};
use std::fs;
use std::path::PathBuf;

/// Simple file-based storage for API key (in app data directory)
fn get_api_key_path() -> PathBuf {
    let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
    let config_dir = home.join(".skyclip");
    fs::create_dir_all(&config_dir).ok();
    config_dir.join("anthropic_api_key")
}

#[tauri::command]
pub async fn save_api_key(api_key: String) -> Result<(), String> {
    let path = get_api_key_path();
    fs::write(&path, &api_key).map_err(|e| format!("Failed to save API key: {}", e))?;
    Ok(())
}

#[tauri::command]
pub async fn get_api_key() -> Result<Option<String>, String> {
    let path = get_api_key_path();
    if path.exists() {
        let key = fs::read_to_string(&path).map_err(|e| format!("Failed to read API key: {}", e))?;
        let key = key.trim().to_string();
        if key.is_empty() {
            Ok(None)
        } else {
            Ok(Some(key))
        }
    } else {
        Ok(None)
    }
}

#[tauri::command]
pub async fn clear_api_key() -> Result<(), String> {
    let path = get_api_key_path();
    if path.exists() {
        fs::remove_file(&path).map_err(|e| format!("Failed to clear API key: {}", e))?;
    }
    Ok(())
}

/// Segment info from frontend
#[derive(Debug, Clone, serde::Deserialize)]
pub struct SegmentInput {
    pub id: String,
    pub start_ms: i64,
    pub end_ms: i64,
    pub thumbnail_path: Option<String>,
    pub gimbal_pitch_delta: Option<f64>,
    pub gimbal_yaw_delta: Option<f64>,
    pub gimbal_smoothness: Option<f64>,
    pub gps_speed: Option<f64>,
    pub altitude_delta: Option<f64>,
    pub score: Option<f64>,
}

#[tauri::command]
pub async fn director_generate_edit(
    prompt: String,
    segments: Vec<SegmentInput>,
    target_duration_sec: Option<f64>,
) -> Result<EditSequence, String> {
    // Get API key
    let api_key = get_api_key().await?
        .ok_or_else(|| "No Anthropic API key configured. Please add your API key in settings.".to_string())?;

    // Convert to SegmentContext
    let contexts: Vec<SegmentContext> = segments
        .iter()
        .map(|s| SegmentContext {
            id: s.id.clone(),
            duration_sec: (s.end_ms - s.start_ms) as f64 / 1000.0,
            start_ms: s.start_ms,
            end_ms: s.end_ms,
            gimbal_pitch_delta: s.gimbal_pitch_delta.unwrap_or(0.0),
            gimbal_yaw_delta: s.gimbal_yaw_delta.unwrap_or(0.0),
            gimbal_smoothness: s.gimbal_smoothness.unwrap_or(1.0),
            gps_speed: s.gps_speed.unwrap_or(0.0),
            altitude_delta: s.altitude_delta.unwrap_or(0.0),
            score: s.score.unwrap_or(50.0),
        })
        .collect();

    // Collect thumbnail paths
    let thumbnail_paths: Vec<String> = segments
        .iter()
        .filter_map(|s| s.thumbnail_path.clone())
        .collect();

    // Call Claude
    let director = Director::new(api_key);
    let response = director
        .generate_edit(&prompt, contexts, thumbnail_paths, target_duration_sec)
        .await
        .map_err(|e| format!("Director API error: {}", e))?;

    // Convert to EditSequence format
    let decisions: Vec<EditDecision> = response
        .edit_sequence
        .into_iter()
        .map(|d| EditDecision {
            clip_id: d.segment_id,
            sequence_order: d.sequence_order,
            adjusted_start_ms: d.in_point_ms,
            adjusted_end_ms: d.out_point_ms,
            transition_type: d.transition_to_next,
            transition_duration_ms: d.transition_duration_ms,
            confidence: 0.9, // AI suggestions get high confidence
            reasoning: d.reasoning,
        })
        .collect();

    let total_duration_ms = (response.total_duration_sec * 1000.0) as i64;

    Ok(EditSequence {
        decisions,
        total_duration_ms,
        style: format!("AI Director: {}", response.style_notes),
        was_reordered: true, // AI may have reordered
    })
}
```

## src-tauri/src/commands/ingest.rs

```rust
use crate::models::{Flight, SourceClip, TelemetryFrame};
use crate::services::{Database, FFmpeg, SrtParser};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use tauri::State;
use tokio::sync::Mutex;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestProgress {
    pub stage: String,
    pub current: u32,
    pub total: u32,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestResult {
    pub flight_id: String,
    pub clips_count: u32,
    pub lrf_used: u32,
    pub proxies_generated: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClipInfo {
    pub filename: String,
    pub source_path: String,
    pub srt_path: Option<String>,
    pub lrf_path: Option<String>,
    pub duration_sec: Option<f64>,
    pub resolution: Option<String>,
    pub framerate: Option<f64>,
}

pub struct AppState {
    pub db: Mutex<Option<Database>>,
    pub app_data_dir: Mutex<PathBuf>,
}

/// Initialize the database
#[tauri::command]
pub async fn init_database(state: State<'_, AppState>) -> Result<(), String> {
    let app_data_dir = state.app_data_dir.lock().await.clone();
    let db_path = app_data_dir.join("library.db");

    // Ensure directory exists
    std::fs::create_dir_all(&app_data_dir).map_err(|e| e.to_string())?;

    let db = Database::new(&db_path)
        .await
        .map_err(|e| e.to_string())?;

    *state.db.lock().await = Some(db);
    Ok(())
}

/// Scan a folder for DJI footage
#[tauri::command]
pub async fn scan_folder(folder_path: String) -> Result<Vec<ClipInfo>, String> {
    let folder = Path::new(&folder_path);
    if !folder.exists() {
        return Err("Folder does not exist".to_string());
    }

    let mut clips = Vec::new();
    let ffmpeg = FFmpeg::new().map_err(|e| e.to_string())?;

    // Look for DJI folder structure
    let dcim_path = folder.join("DCIM");
    let media_folders = if dcim_path.exists() {
        find_media_folders(&dcim_path)?
    } else {
        // Direct folder with media files
        vec![folder.to_path_buf()]
    };

    for media_folder in media_folders {
        let entries = std::fs::read_dir(&media_folder).map_err(|e| e.to_string())?;

        for entry in entries.flatten() {
            let path = entry.path();
            if let Some(ext) = path.extension() {
                let ext_lower = ext.to_string_lossy().to_lowercase();
                if ext_lower == "mp4" || ext_lower == "mov" {
                    let filename = path.file_name().unwrap().to_string_lossy().to_string();
                    let stem = path.file_stem().unwrap().to_string_lossy().to_string();

                    // Look for matching SRT file
                    let srt_path = media_folder.join(format!("{stem}.SRT"));
                    let srt_exists = srt_path.exists();

                    // Look for matching LRF file
                    let lrf_path = find_lrf_file(folder, &media_folder, &stem);

                    // Get video info
                    let video_info = ffmpeg.probe(&path).ok();

                    clips.push(ClipInfo {
                        filename,
                        source_path: path.to_string_lossy().to_string(),
                        srt_path: if srt_exists {
                            Some(srt_path.to_string_lossy().to_string())
                        } else {
                            None
                        },
                        lrf_path: lrf_path.map(|p| p.to_string_lossy().to_string()),
                        duration_sec: video_info.as_ref().map(|v| v.duration_sec),
                        resolution: video_info
                            .as_ref()
                            .map(|v| format!("{}x{}", v.width, v.height)),
                        framerate: video_info.as_ref().map(|v| v.framerate),
                    });
                }
            }
        }
    }

    clips.sort_by(|a, b| a.filename.cmp(&b.filename));
    Ok(clips)
}

/// Ingest footage from a folder
#[tauri::command]
pub async fn ingest_folder(
    state: State<'_, AppState>,
    folder_path: String,
    flight_name: String,
) -> Result<IngestResult, String> {
    let db_guard = state.db.lock().await;
    let db = db_guard.as_ref().ok_or("Database not initialized")?;
    let app_data_dir = state.app_data_dir.lock().await.clone();

    let proxies_dir = app_data_dir.join("proxies");
    let thumbnails_dir = app_data_dir.join("thumbnails");
    let srt_dir = app_data_dir.join("srt");
    std::fs::create_dir_all(&proxies_dir).map_err(|e| e.to_string())?;
    std::fs::create_dir_all(&thumbnails_dir).map_err(|e| e.to_string())?;
    std::fs::create_dir_all(&srt_dir).map_err(|e| e.to_string())?;

    // Scan for clips
    let clips = scan_folder(folder_path.clone()).await?;
    if clips.is_empty() {
        return Err("No video clips found in folder".to_string());
    }

    // Create flight record
    let mut flight = Flight::new(flight_name, folder_path.clone());
    flight.total_clips = Some(clips.len() as i32);

    db.insert_flight(&flight).await.map_err(|e| e.to_string())?;

    let ffmpeg = FFmpeg::new().map_err(|e| e.to_string())?;
    let srt_parser = SrtParser::new();

    let mut lrf_used = 0u32;
    let mut proxies_generated = 0u32;

    for clip_info in &clips {
        let mut source_clip = SourceClip::new(
            flight.id.clone(),
            clip_info.filename.clone(),
            clip_info.source_path.clone(),
        );

        source_clip.srt_path = clip_info.srt_path.clone();
        source_clip.duration_sec = clip_info.duration_sec;

        if let Some(res) = &clip_info.resolution {
            if let Some((w, h)) = res.split_once('x') {
                source_clip.resolution_width = w.parse().ok();
                source_clip.resolution_height = h.parse().ok();
            }
        }
        source_clip.framerate = clip_info.framerate;

        // Copy and parse SRT file if available
        if let Some(srt_path) = &clip_info.srt_path {
            // Copy SRT to local storage so analysis works without source media
            let srt_filename = format!("{}.srt", source_clip.id);
            let local_srt_path = srt_dir.join(&srt_filename);
            if std::fs::copy(srt_path, &local_srt_path).is_ok() {
                source_clip.srt_path = Some(local_srt_path.to_string_lossy().to_string());
            }

            // Parse SRT for metadata
            if let Ok(frames) = srt_parser.parse_file(srt_path) {
                // Extract first timestamp as recorded_at
                if let Some(first_frame) = frames.first() {
                    source_clip.recorded_at = first_frame.timestamp;
                }
            }
        }

        // Handle proxy generation
        let proxy_filename = format!("{}.mp4", source_clip.id);
        let proxy_path = proxies_dir.join(&proxy_filename);

        if let Some(lrf_path) = &clip_info.lrf_path {
            // Validate and use LRF file
            if validate_lrf(lrf_path, clip_info.duration_sec) {
                std::fs::copy(lrf_path, &proxy_path).map_err(|e| e.to_string())?;
                source_clip.proxy_path = Some(proxy_path.to_string_lossy().to_string());
                source_clip.proxy_source = Some("lrf".to_string());
                lrf_used += 1;
            } else {
                // LRF invalid, generate proxy
                let proxy_path_str = proxy_path.to_string_lossy().to_string();
                ffmpeg
                    .generate_proxy(&clip_info.source_path, &proxy_path_str)
                    .map_err(|e| e.to_string())?;
                source_clip.proxy_path = Some(proxy_path.to_string_lossy().to_string());
                source_clip.proxy_source = Some("generated".to_string());
                proxies_generated += 1;
            }
        } else {
            // No LRF, generate proxy
            let proxy_path_str = proxy_path.to_string_lossy().to_string();
            ffmpeg
                .generate_proxy(&clip_info.source_path, &proxy_path_str)
                .map_err(|e| e.to_string())?;
            source_clip.proxy_path = Some(proxy_path.to_string_lossy().to_string());
            source_clip.proxy_source = Some("generated".to_string());
            proxies_generated += 1;
        }

        // Extract thumbnails from proxy (much faster than 4K source)
        let clip_thumb_dir = thumbnails_dir.join(&source_clip.id);
        std::fs::create_dir_all(&clip_thumb_dir).map_err(|e| e.to_string())?;

        let clip_thumb_dir_str = clip_thumb_dir.to_string_lossy().to_string();
        // Use proxy if available, otherwise fall back to source
        let thumb_source = source_clip
            .proxy_path
            .as_ref()
            .unwrap_or(&clip_info.source_path);
        let _thumbnails = ffmpeg
            .extract_thumbnails(thumb_source, &clip_thumb_dir_str, "thumb")
            .map_err(|e| e.to_string())?;

        db.insert_clip(&source_clip)
            .await
            .map_err(|e| e.to_string())?;
    }

    Ok(IngestResult {
        flight_id: flight.id,
        clips_count: clips.len() as u32,
        lrf_used,
        proxies_generated,
    })
}

/// List all flights
#[tauri::command]
pub async fn list_flights(state: State<'_, AppState>) -> Result<Vec<Flight>, String> {
    let db_guard = state.db.lock().await;
    let db = db_guard.as_ref().ok_or("Database not initialized")?;
    db.list_flights().await.map_err(|e| e.to_string())
}

/// Delete a flight and all associated data
#[tauri::command]
pub async fn delete_flight(state: State<'_, AppState>, flight_id: String) -> Result<(), String> {
    let db_guard = state.db.lock().await;
    let db = db_guard.as_ref().ok_or("Database not initialized")?;
    db.delete_flight(&flight_id).await.map_err(|e| e.to_string())
}

/// Get clips for a flight
#[tauri::command]
pub async fn get_flight_clips(
    state: State<'_, AppState>,
    flight_id: String,
) -> Result<Vec<SourceClip>, String> {
    let db_guard = state.db.lock().await;
    let db = db_guard.as_ref().ok_or("Database not initialized")?;
    db.get_clips_for_flight(&flight_id)
        .await
        .map_err(|e| e.to_string())
}

/// Parse an SRT file and return telemetry frames
#[tauri::command]
pub fn parse_srt(srt_path: String) -> Result<Vec<TelemetryFrame>, String> {
    let parser = SrtParser::new();
    parser.parse_file(&srt_path).map_err(|e| e.to_string())
}

// Helper functions

fn find_media_folders(dcim_path: &Path) -> Result<Vec<PathBuf>, String> {
    let mut folders = Vec::new();

    for entry in std::fs::read_dir(dcim_path).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        let path = entry.path();
        if path.is_dir() {
            let name = path.file_name().unwrap().to_string_lossy();
            // DJI uses 100MEDIA, 101MEDIA, etc.
            if name.ends_with("MEDIA") || name == "PANORAMA" || name == "TIMELAPSE" {
                folders.push(path);
            }
        }
    }

    Ok(folders)
}

fn find_lrf_file(base_folder: &Path, media_folder: &Path, stem: &str) -> Option<PathBuf> {
    // Strategy 1: Check same folder as MP4 (some DJI drones put LRF alongside MP4)
    let same_folder_lrf = media_folder.join(format!("{stem}.LRF"));
    if same_folder_lrf.exists() {
        return Some(same_folder_lrf);
    }

    // Strategy 2: Check separate LRF/ folder at base level
    // Structure: base/LRF/100/ or base/LRF/100MEDIA/
    let lrf_base = base_folder.join("LRF");
    if lrf_base.exists() {
        // Try matching folder name (100MEDIA -> 100MEDIA or 100)
        if let Some(media_name) = media_folder.file_name() {
            let media_name_str = media_name.to_string_lossy();

            // Try exact match first (100MEDIA)
            let lrf_exact = lrf_base.join(&*media_name_str).join(format!("{stem}.LRF"));
            if lrf_exact.exists() {
                return Some(lrf_exact);
            }

            // Try without MEDIA suffix (100MEDIA -> 100)
            if media_name_str.ends_with("MEDIA") {
                let short_name = media_name_str.trim_end_matches("MEDIA");
                let lrf_short = lrf_base.join(short_name).join(format!("{stem}.LRF"));
                if lrf_short.exists() {
                    return Some(lrf_short);
                }
            }
        }

        // Fallback: search recursively in LRF folder
        for entry in walkdir(lrf_base).ok()?.flatten() {
            let path = entry.path();
            if path.extension().map(|e| e.to_string_lossy().to_lowercase()) == Some("lrf".to_string()) {
                if path.file_stem().map(|s| s.to_string_lossy()) == Some(stem.into()) {
                    return Some(path.to_path_buf());
                }
            }
        }
    }

    None
}

fn walkdir(path: PathBuf) -> Result<impl Iterator<Item = Result<std::fs::DirEntry, std::io::Error>>, std::io::Error> {
    fn walk_recursive(path: PathBuf) -> Box<dyn Iterator<Item = Result<std::fs::DirEntry, std::io::Error>>> {
        match std::fs::read_dir(&path) {
            Ok(entries) => {
                let iter = entries.flat_map(move |entry| {
                    match entry {
                        Ok(e) => {
                            let path = e.path();
                            if path.is_dir() {
                                let sub = walk_recursive(path);
                                Box::new(std::iter::once(Ok(e)).chain(sub)) as Box<dyn Iterator<Item = _>>
                            } else {
                                Box::new(std::iter::once(Ok(e)))
                            }
                        }
                        Err(e) => Box::new(std::iter::once(Err(e))),
                    }
                });
                Box::new(iter)
            }
            Err(e) => Box::new(std::iter::once(Err(e))),
        }
    }
    Ok(walk_recursive(path))
}

fn validate_lrf(lrf_path: &str, expected_duration: Option<f64>) -> bool {
    let path = Path::new(lrf_path);

    // Check file exists and is non-zero
    match std::fs::metadata(path) {
        Ok(meta) if meta.len() > 0 => {}
        _ => return false,
    }

    // Validate with ffprobe
    let ffmpeg = match FFmpeg::new() {
        Ok(f) => f,
        Err(_) => return false,
    };

    let lrf_info = match ffmpeg.probe(path) {
        Ok(info) => info,
        Err(_) => return false,
    };

    // Check duration matches (±1 frame tolerance at 30fps = ~33ms)
    if let Some(expected) = expected_duration {
        let diff = (lrf_info.duration_sec - expected).abs();
        if diff > 0.1 {
            // More than 100ms difference
            return false;
        }
    }

    true
}
```

## src-tauri/src/commands/mod.rs

```rust
mod ingest;
mod analyze;
mod visual_analysis;
mod director;

pub use ingest::*;
pub use analyze::*;
pub use visual_analysis::*;
pub use director::*;
```

## src-tauri/src/commands/visual_analysis.rs

```rust
use serde::{Deserialize, Serialize};
use tauri::State;

use crate::commands::ingest::AppState;
use crate::services::{PythonSidecar, ClipInfo, FFmpeg, ConcatClip};

/// Visual analysis result for a segment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualAnalysisResult {
    pub segment_id: String,
    pub motion_avg: Option<f64>,
    pub motion_direction: Option<f64>,
    pub dominant_color: Option<(u8, u8, u8)>,
    pub is_golden_hour: Option<bool>,
    pub scene_type: Option<String>,
    pub has_subject: Option<bool>,
}

/// Edit decision for frontend display
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EditDecisionResponse {
    pub clip_id: String,
    pub sequence_order: i32,
    pub adjusted_start_ms: i64,
    pub adjusted_end_ms: i64,
    pub transition_type: String,
    pub transition_duration_ms: i64,
    pub confidence: f64,
    pub reasoning: String,
}

/// Edit sequence for frontend display
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EditSequenceResponse {
    pub decisions: Vec<EditDecisionResponse>,
    pub total_duration_ms: i64,
    pub style: String,
    pub was_reordered: bool,
}

/// Check if Python sidecar is available
#[tauri::command]
pub async fn check_python_available() -> Result<bool, String> {
    match PythonSidecar::new() {
        Ok(sidecar) => Ok(sidecar.is_available()),
        Err(_) => Ok(false),
    }
}

/// Install Python dependencies
#[tauri::command]
pub async fn install_python_deps() -> Result<(), String> {
    let sidecar = PythonSidecar::new()?;
    sidecar.install_dependencies()
}

/// Run visual analysis on a segment
#[tauri::command]
pub async fn analyze_segment_visual(
    state: State<'_, AppState>,
    segment_id: String,
    include_objects: bool,
    include_semantic: bool,
) -> Result<VisualAnalysisResult, String> {
    let db_guard = state.db.lock().await;
    let db = db_guard.as_ref().ok_or("Database not initialized")?;

    // Get segment and clip info
    let segment = db
        .get_segment(&segment_id)
        .await
        .map_err(|e| e.to_string())?
        .ok_or("Segment not found")?;

    let clip = db
        .get_clip(&segment.source_clip_id)
        .await
        .map_err(|e| e.to_string())?
        .ok_or("Clip not found")?;

    // Use proxy if available, otherwise source
    let video_path = clip.proxy_path.as_ref().unwrap_or(&clip.source_path);

    // Run Python analysis
    let sidecar = PythonSidecar::new()?;
    let analysis = sidecar.analyze_clip(
        video_path,
        segment.start_time_ms,
        Some(segment.end_time_ms),
        include_objects,
        include_semantic,
    )?;

    // Extract key metrics for frontend
    Ok(VisualAnalysisResult {
        segment_id,
        motion_avg: analysis.motion.as_ref().map(|m| m.avg_magnitude),
        motion_direction: analysis.motion.as_ref().map(|m| m.dominant_direction),
        dominant_color: analysis
            .color
            .as_ref()
            .and_then(|c| c.dominant_colors.first().cloned()),
        is_golden_hour: analysis.color.as_ref().map(|c| c.is_golden_hour),
        scene_type: analysis.semantic.as_ref().map(|s| s.scene_type.clone()),
        has_subject: analysis.objects.as_ref().map(|o| o.has_consistent_subject),
    })
}

/// Generate an edit sequence for selected segments
#[tauri::command]
pub async fn generate_edit_sequence(
    state: State<'_, AppState>,
    segment_ids: Vec<String>,
    style: String,
    reorder: bool,
) -> Result<EditSequenceResponse, String> {
    let db_guard = state.db.lock().await;
    let db = db_guard.as_ref().ok_or("Database not initialized")?;

    // Build clip info list
    let mut clips = Vec::new();
    for segment_id in &segment_ids {
        let segment = db
            .get_segment(segment_id)
            .await
            .map_err(|e| e.to_string())?
            .ok_or("Segment not found")?;

        let clip = db
            .get_clip(&segment.source_clip_id)
            .await
            .map_err(|e| e.to_string())?
            .ok_or("Clip not found")?;

        let video_path = clip.proxy_path.as_ref().unwrap_or(&clip.source_path);

        clips.push(ClipInfo {
            clip_id: segment_id.clone(),
            video_path: video_path.clone(),
            start_ms: segment.start_time_ms,
            end_ms: segment.end_time_ms,
        });
    }

    // Try Python edit sequence generation, fall back to basic if unavailable
    let sequence = match PythonSidecar::new() {
        Ok(sidecar) => {
            match sidecar.generate_edit_sequence(clips.clone(), &style, reorder, false) {
                Ok(seq) => seq,
                Err(_) => create_fallback_sequence(clips, &style),
            }
        }
        Err(_) => create_fallback_sequence(clips, &style),
    };

    Ok(EditSequenceResponse {
        decisions: sequence
            .decisions
            .into_iter()
            .map(|d| EditDecisionResponse {
                clip_id: d.clip_id,
                sequence_order: d.sequence_order,
                adjusted_start_ms: d.adjusted_start_ms,
                adjusted_end_ms: d.adjusted_end_ms,
                transition_type: d.transition_type,
                transition_duration_ms: d.transition_duration_ms,
                confidence: d.confidence,
                reasoning: d.reasoning,
            })
            .collect(),
        total_duration_ms: sequence.total_duration_ms,
        style: sequence.style,
        was_reordered: sequence.was_reordered,
    })
}

/// Create a basic edit sequence without Python analysis
fn create_fallback_sequence(clips: Vec<ClipInfo>, style: &str) -> crate::services::EditSequence {
    let (default_transition, transition_duration) = match style {
        "action" => ("cut", 200),
        "social" => ("cut", 300),
        _ => ("dissolve", 500), // cinematic default
    };

    let mut total_duration_ms = 0i64;
    let decisions: Vec<crate::services::EditDecision> = clips
        .iter()
        .enumerate()
        .map(|(i, clip)| {
            let duration = clip.end_ms - clip.start_ms;
            total_duration_ms += duration;
            crate::services::EditDecision {
                clip_id: clip.clip_id.clone(),
                sequence_order: i as i32,
                adjusted_start_ms: clip.start_ms,
                adjusted_end_ms: clip.end_ms,
                transition_type: default_transition.to_string(),
                transition_duration_ms: transition_duration,
                confidence: 0.5,
                reasoning: format!("Basic {} transition (Python unavailable)", style),
            }
        })
        .collect();

    crate::services::EditSequence {
        decisions,
        total_duration_ms,
        style: style.to_string(),
        was_reordered: false,
    }
}

/// Suggest transition between two segments
#[tauri::command]
pub async fn suggest_transition(
    state: State<'_, AppState>,
    segment_a_id: String,
    segment_b_id: String,
    style: String,
) -> Result<(String, i64, f64, String), String> {
    let db_guard = state.db.lock().await;
    let db = db_guard.as_ref().ok_or("Database not initialized")?;

    // Get segment and clip info for both
    let segment_a = db
        .get_segment(&segment_a_id)
        .await
        .map_err(|e| e.to_string())?
        .ok_or("Segment A not found")?;
    let clip_a = db
        .get_clip(&segment_a.source_clip_id)
        .await
        .map_err(|e| e.to_string())?
        .ok_or("Clip A not found")?;

    let segment_b = db
        .get_segment(&segment_b_id)
        .await
        .map_err(|e| e.to_string())?
        .ok_or("Segment B not found")?;
    let clip_b = db
        .get_clip(&segment_b.source_clip_id)
        .await
        .map_err(|e| e.to_string())?
        .ok_or("Clip B not found")?;

    let video_a = clip_a.proxy_path.as_ref().unwrap_or(&clip_a.source_path);
    let video_b = clip_b.proxy_path.as_ref().unwrap_or(&clip_b.source_path);

    let sidecar = PythonSidecar::new()?;
    sidecar.suggest_transition(
        ClipInfo {
            clip_id: segment_a_id,
            video_path: video_a.clone(),
            start_ms: segment_a.start_time_ms,
            end_ms: segment_a.end_time_ms,
        },
        ClipInfo {
            clip_id: segment_b_id,
            video_path: video_b.clone(),
            start_ms: segment_b.start_time_ms,
            end_ms: segment_b.end_time_ms,
        },
        &style,
    )
}

/// Render input for highlight reel
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RenderClipInput {
    pub segment_id: String,
    pub adjusted_start_ms: i64,
    pub adjusted_end_ms: i64,
    pub transition_type: String,
    pub transition_duration_ms: i64,
}

/// Render result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RenderResult {
    pub output_path: String,
    pub duration_sec: f64,
    pub clips_count: usize,
}

/// Render a highlight reel from an edit sequence
#[tauri::command]
pub async fn render_highlight_reel(
    state: State<'_, AppState>,
    clips: Vec<RenderClipInput>,
    output_path: String,
    use_source: bool,
) -> Result<RenderResult, String> {
    let db_guard = state.db.lock().await;
    let db = db_guard.as_ref().ok_or("Database not initialized")?;

    // Build ConcatClip list by resolving segment IDs to paths
    let mut concat_clips = Vec::new();
    let mut total_duration_ms: i64 = 0;

    for clip_input in &clips {
        let segment = db
            .get_segment(&clip_input.segment_id)
            .await
            .map_err(|e| e.to_string())?
            .ok_or_else(|| format!("Segment not found: {}", clip_input.segment_id))?;

        let source_clip = db
            .get_clip(&segment.source_clip_id)
            .await
            .map_err(|e| e.to_string())?
            .ok_or_else(|| format!("Clip not found: {}", segment.source_clip_id))?;

        // Use source for quality, proxy for speed
        let input_path = if use_source {
            source_clip.source_path.clone()
        } else {
            source_clip
                .proxy_path
                .clone()
                .unwrap_or(source_clip.source_path.clone())
        };

        // Use adjusted times from edit sequence
        let start_sec = clip_input.adjusted_start_ms as f64 / 1000.0;
        let end_sec = clip_input.adjusted_end_ms as f64 / 1000.0;

        concat_clips.push(ConcatClip {
            input_path,
            start_sec,
            end_sec,
            transition_type: clip_input.transition_type.clone(),
            transition_duration_ms: clip_input.transition_duration_ms,
        });

        total_duration_ms += clip_input.adjusted_end_ms - clip_input.adjusted_start_ms;
    }

    drop(db_guard);

    // Render using FFmpeg
    let ffmpeg = FFmpeg::new().map_err(|e| e.to_string())?;
    ffmpeg
        .concat_with_transitions(concat_clips.clone(), &output_path, true)
        .map_err(|e| e.to_string())?;

    Ok(RenderResult {
        output_path,
        duration_sec: total_duration_ms as f64 / 1000.0,
        clips_count: clips.len(),
    })
}
```

## src-tauri/src/lib.rs

```rust
mod commands;
mod models;
mod services;

use commands::{
    analyze_clip, analyze_flight, delete_flight, export_segment, get_clip_segments,
    get_flight_clips, get_segment_with_clip, get_top_segments, ingest_folder, init_database,
    list_flights, list_profiles, parse_srt, scan_folder, AppState,
    // Visual analysis commands
    check_python_available, install_python_deps, analyze_segment_visual,
    generate_edit_sequence, suggest_transition, render_highlight_reel,
    // AI Director commands
    save_api_key, get_api_key, clear_api_key, director_generate_edit,
};
use tauri::Manager;
use tokio::sync::Mutex;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .setup(|app| {
            let app_data_dir = app
                .path()
                .app_data_dir()
                .expect("Failed to get app data directory");

            app.manage(AppState {
                db: Mutex::new(None),
                app_data_dir: Mutex::new(app_data_dir),
            });

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            init_database,
            scan_folder,
            ingest_folder,
            list_flights,
            delete_flight,
            get_flight_clips,
            parse_srt,
            analyze_clip,
            analyze_flight,
            get_clip_segments,
            get_top_segments,
            get_segment_with_clip,
            export_segment,
            list_profiles,
            // Visual analysis
            check_python_available,
            install_python_deps,
            analyze_segment_visual,
            generate_edit_sequence,
            suggest_transition,
            render_highlight_reel,
            // AI Director
            save_api_key,
            get_api_key,
            clear_api_key,
            director_generate_edit,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
```

## src-tauri/src/main.rs

```rust
// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    skyclip_lib::run()
}
```

## src-tauri/src/models/clip.rs

```rust
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceClip {
    pub id: String,
    pub flight_id: String,
    pub filename: String,
    pub source_path: String,
    pub proxy_path: Option<String>,
    pub proxy_source: Option<String>, // "lrf" or "generated"
    pub srt_path: Option<String>,
    pub duration_sec: Option<f64>,
    pub resolution_width: Option<i32>,
    pub resolution_height: Option<i32>,
    pub framerate: Option<f64>,
    pub recorded_at: Option<DateTime<Utc>>,
}

impl SourceClip {
    pub fn new(flight_id: String, filename: String, source_path: String) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            flight_id,
            filename,
            source_path,
            proxy_path: None,
            proxy_source: None,
            srt_path: None,
            duration_sec: None,
            resolution_width: None,
            resolution_height: None,
            framerate: None,
            recorded_at: None,
        }
    }
}
```

## src-tauri/src/models/flight.rs

```rust
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Flight {
    pub id: String,
    pub name: String,
    pub import_date: DateTime<Utc>,
    pub source_path: String,
    pub location_name: Option<String>,
    pub gps_center_lat: Option<f64>,
    pub gps_center_lon: Option<f64>,
    pub total_duration_sec: Option<f64>,
    pub total_clips: Option<i32>,
}

impl Flight {
    pub fn new(name: String, source_path: String) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            name,
            import_date: Utc::now(),
            source_path,
            location_name: None,
            gps_center_lat: None,
            gps_center_lon: None,
            total_duration_sec: None,
            total_clips: None,
        }
    }
}
```

## src-tauri/src/models/mod.rs

```rust
mod flight;
mod clip;
mod segment;
mod telemetry;

pub use flight::Flight;
pub use clip::SourceClip;
pub use segment::Segment;
pub use telemetry::TelemetryFrame;
```

## src-tauri/src/models/segment.rs

```rust
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Segment {
    pub id: String,
    pub source_clip_id: String,
    pub start_time_ms: i64,
    pub end_time_ms: i64,
    pub duration_ms: i64,
    pub thumbnail_path: Option<String>,

    // Raw signals (computed once)
    pub motion_magnitude: Option<f64>,
    pub gimbal_pitch_delta_avg: Option<f64>,
    pub gimbal_yaw_delta_avg: Option<f64>,
    pub gimbal_smoothness: Option<f64>,
    pub altitude_delta: Option<f64>,
    pub gps_speed_avg: Option<f64>,
    pub iso_avg: Option<f64>,
    pub visual_quality: Option<f64>,
    pub has_scene_change: Option<bool>,

    // User state
    pub is_selected: bool,
    pub user_adjusted_start_ms: Option<i64>,
    pub user_adjusted_end_ms: Option<i64>,
}

impl Segment {
    pub fn new(source_clip_id: String, start_time_ms: i64, end_time_ms: i64) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            source_clip_id,
            start_time_ms,
            end_time_ms,
            duration_ms: end_time_ms - start_time_ms,
            thumbnail_path: None,
            motion_magnitude: None,
            gimbal_pitch_delta_avg: None,
            gimbal_yaw_delta_avg: None,
            gimbal_smoothness: None,
            altitude_delta: None,
            gps_speed_avg: None,
            iso_avg: None,
            visual_quality: None,
            has_scene_change: None,
            is_selected: false,
            user_adjusted_start_ms: None,
            user_adjusted_end_ms: None,
        }
    }
}
```

## src-tauri/src/models/telemetry.rs

```rust
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// A single frame of telemetry data parsed from DJI SRT files.
/// Each frame represents one second of data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryFrame {
    /// Frame index (1-based, matches SrtCnt)
    pub index: u32,
    /// Start time in milliseconds
    pub start_time_ms: i64,
    /// End time in milliseconds
    pub end_time_ms: i64,
    /// Timestamp from the drone
    pub timestamp: Option<DateTime<Utc>>,

    // Camera settings
    pub iso: Option<i32>,
    pub shutter: Option<String>, // e.g., "1/500.0"
    pub fnum: Option<i32>,       // f-number * 100 (e.g., 280 = f/2.8)
    pub ev: Option<f64>,         // exposure value
    pub color_temp: Option<i32>, // color temperature
    pub color_mode: Option<String>,
    pub focal_len: Option<f64>,

    // GPS data
    pub latitude: Option<f64>,
    pub longitude: Option<f64>,
    pub altitude: Option<f64>,

    // Gimbal orientation
    pub gimbal_yaw: Option<f64>,
    pub gimbal_pitch: Option<f64>,
    pub gimbal_roll: Option<f64>,
}

impl TelemetryFrame {
    pub fn new(index: u32, start_time_ms: i64, end_time_ms: i64) -> Self {
        Self {
            index,
            start_time_ms,
            end_time_ms,
            timestamp: None,
            iso: None,
            shutter: None,
            fnum: None,
            ev: None,
            color_temp: None,
            color_mode: None,
            focal_len: None,
            latitude: None,
            longitude: None,
            altitude: None,
            gimbal_yaw: None,
            gimbal_pitch: None,
            gimbal_roll: None,
        }
    }
}
```

## src-tauri/src/services/analyzer.rs

```rust
use crate::models::TelemetryFrame;

/// Computed signal values for a segment of video
#[derive(Debug, Clone, Default)]
pub struct SegmentSignals {
    /// Average rate of gimbal pitch change (degrees/sec)
    pub gimbal_pitch_delta_avg: f64,
    /// Average rate of gimbal yaw change (degrees/sec)
    pub gimbal_yaw_delta_avg: f64,
    /// Gimbal smoothness score (0-1, higher = smoother)
    pub gimbal_smoothness: f64,
    /// Average GPS horizontal speed (m/s)
    pub gps_speed_avg: f64,
    /// Total altitude change over segment (meters)
    pub altitude_delta: f64,
    /// Average ISO value
    pub iso_avg: f64,
    /// Motion magnitude combining all movement signals
    pub motion_magnitude: f64,
}

pub struct TelemetryAnalyzer;

impl TelemetryAnalyzer {
    pub fn new() -> Self {
        Self
    }

    /// Analyze a slice of telemetry frames and compute signal values
    pub fn analyze_frames(&self, frames: &[TelemetryFrame]) -> SegmentSignals {
        if frames.is_empty() {
            return SegmentSignals::default();
        }

        if frames.len() == 1 {
            return SegmentSignals {
                iso_avg: frames[0].iso.map(|i| i as f64).unwrap_or(0.0),
                gimbal_smoothness: 1.0, // Single frame = perfectly smooth
                ..Default::default()
            };
        }

        // Compute gimbal deltas
        let (pitch_deltas, yaw_deltas) = self.compute_gimbal_deltas(frames);
        let gimbal_pitch_delta_avg = average(&pitch_deltas);
        let gimbal_yaw_delta_avg = average(&yaw_deltas);

        // Gimbal smoothness: inverse of jitter (standard deviation of deltas)
        // If no gimbal data available, default to 1.0 (perfectly smooth - no jitter detected)
        let gimbal_smoothness = if pitch_deltas.is_empty() && yaw_deltas.is_empty() {
            1.0 // No gimbal data = assume smooth (neutral value that passes thresholds)
        } else {
            let pitch_jitter = std_dev(&pitch_deltas);
            let yaw_jitter = std_dev(&yaw_deltas);
            let combined_jitter = (pitch_jitter + yaw_jitter) / 2.0;
            // Map jitter to 0-1 smoothness (lower jitter = higher smoothness)
            // Using sigmoid-like transform: smoothness = 1 / (1 + jitter/10)
            1.0 / (1.0 + combined_jitter / 10.0)
        };

        // Compute GPS speed
        let gps_speeds = self.compute_gps_speeds(frames);
        let gps_speed_avg = average(&gps_speeds);

        // Compute altitude delta (first to last)
        let altitude_delta = self.compute_altitude_delta(frames);

        // Compute average ISO
        let iso_values: Vec<f64> = frames
            .iter()
            .filter_map(|f| f.iso.map(|i| i as f64))
            .collect();
        let iso_avg = if iso_values.is_empty() {
            0.0
        } else {
            average(&iso_values)
        };

        // Motion magnitude: combined normalized score
        // Adapt weights based on available data
        let gimbal_motion = (gimbal_pitch_delta_avg.abs() + gimbal_yaw_delta_avg.abs()) / 2.0;
        let has_gimbal_data = !pitch_deltas.is_empty() || !yaw_deltas.is_empty();

        // If no gimbal data, rely entirely on GPS; otherwise use weighted combination
        let motion_magnitude = if has_gimbal_data {
            (gimbal_motion * 0.6) + (gps_speed_avg * 0.4)
        } else {
            // GPS-only mode: use GPS speed + altitude change as motion indicator
            gps_speed_avg + (altitude_delta.abs() / 10.0) // altitude contributes up to 5 units per 50m
        };

        SegmentSignals {
            gimbal_pitch_delta_avg,
            gimbal_yaw_delta_avg,
            gimbal_smoothness,
            gps_speed_avg,
            altitude_delta,
            iso_avg,
            motion_magnitude,
        }
    }

    /// Compute frame-to-frame gimbal pitch and yaw deltas
    fn compute_gimbal_deltas(&self, frames: &[TelemetryFrame]) -> (Vec<f64>, Vec<f64>) {
        let mut pitch_deltas = Vec::new();
        let mut yaw_deltas = Vec::new();

        for window in frames.windows(2) {
            let prev = &window[0];
            let curr = &window[1];

            // Time delta in seconds
            let time_delta_sec = (curr.start_time_ms - prev.start_time_ms) as f64 / 1000.0;
            if time_delta_sec <= 0.0 {
                continue;
            }

            // Pitch delta (degrees per second)
            if let (Some(prev_pitch), Some(curr_pitch)) = (prev.gimbal_pitch, curr.gimbal_pitch) {
                let delta = (curr_pitch - prev_pitch) / time_delta_sec;
                pitch_deltas.push(delta);
            }

            // Yaw delta - handle wraparound at 180/-180
            if let (Some(prev_yaw), Some(curr_yaw)) = (prev.gimbal_yaw, curr.gimbal_yaw) {
                let mut delta = curr_yaw - prev_yaw;
                // Handle wraparound
                if delta > 180.0 {
                    delta -= 360.0;
                } else if delta < -180.0 {
                    delta += 360.0;
                }
                let delta_per_sec = delta / time_delta_sec;
                yaw_deltas.push(delta_per_sec);
            }
        }

        (pitch_deltas, yaw_deltas)
    }

    /// Compute GPS horizontal speed between consecutive frames
    fn compute_gps_speeds(&self, frames: &[TelemetryFrame]) -> Vec<f64> {
        let mut speeds = Vec::new();

        for window in frames.windows(2) {
            let prev = &window[0];
            let curr = &window[1];

            let time_delta_sec = (curr.start_time_ms - prev.start_time_ms) as f64 / 1000.0;
            if time_delta_sec <= 0.0 {
                continue;
            }

            if let (Some(lat1), Some(lon1), Some(lat2), Some(lon2)) = (
                prev.latitude,
                prev.longitude,
                curr.latitude,
                curr.longitude,
            ) {
                // Skip invalid coordinates (0,0 or very small values indicate no GPS fix)
                if lat1.abs() < 0.1 || lon1.abs() < 0.1 || lat2.abs() < 0.1 || lon2.abs() < 0.1 {
                    continue;
                }

                let distance = haversine_distance(lat1, lon1, lat2, lon2);
                let speed = distance / time_delta_sec;

                // Sanity check: max drone speed is ~30 m/s (108 km/h) for consumer drones
                // Allow up to 50 m/s to account for wind/diving, but filter GPS glitches
                if speed <= 50.0 {
                    speeds.push(speed);
                }
            }
        }

        speeds
    }

    /// Compute total altitude change from first to last frame
    fn compute_altitude_delta(&self, frames: &[TelemetryFrame]) -> f64 {
        let first_alt = frames.iter().find_map(|f| f.altitude);
        let last_alt = frames.iter().rev().find_map(|f| f.altitude);

        match (first_alt, last_alt) {
            (Some(first), Some(last)) => last - first,
            _ => 0.0,
        }
    }

    /// Detect segments from telemetry based on activity thresholds
    /// Returns (start_index, end_index) pairs
    pub fn detect_segments(
        &self,
        frames: &[TelemetryFrame],
        min_segment_sec: f64,
        max_segment_sec: f64,
    ) -> Vec<(usize, usize)> {
        if frames.is_empty() {
            return vec![];
        }

        // Calculate actual frame rate from timestamps
        let total_duration_ms = frames.last().map(|f| f.end_time_ms).unwrap_or(0)
            - frames.first().map(|f| f.start_time_ms).unwrap_or(0);
        let total_duration_sec = total_duration_ms as f64 / 1000.0;

        // Frames per second (could be 1fps for old format, 60fps for new Mavic 3)
        let fps = if total_duration_sec > 0.0 {
            frames.len() as f64 / total_duration_sec
        } else {
            1.0
        };

        let mut segments = Vec::new();
        let min_frames = (min_segment_sec * fps) as usize;
        let max_frames = (max_segment_sec * fps) as usize;

        // If clip is shorter than min segment, create one segment for the whole clip
        if frames.len() < min_frames {
            // Still create a segment if we have at least some data
            if frames.len() >= 2 {
                segments.push((0, frames.len()));
            }
            return segments;
        }

        let mut start = 0;
        while start < frames.len() {
            let end = (start + max_frames).min(frames.len());
            // Accept segment if it meets minimum OR if it's the last chunk and has some content
            if end - start >= min_frames || (end == frames.len() && end - start >= min_frames / 2) {
                segments.push((start, end));
            }
            start = end;
        }

        segments
    }
}

/// Calculate haversine distance between two GPS points in meters
fn haversine_distance(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    const EARTH_RADIUS_M: f64 = 6_371_000.0;

    let lat1_rad = lat1.to_radians();
    let lat2_rad = lat2.to_radians();
    let delta_lat = (lat2 - lat1).to_radians();
    let delta_lon = (lon2 - lon1).to_radians();

    let a = (delta_lat / 2.0).sin().powi(2)
        + lat1_rad.cos() * lat2_rad.cos() * (delta_lon / 2.0).sin().powi(2);
    let c = 2.0 * a.sqrt().asin();

    EARTH_RADIUS_M * c
}

/// Calculate average of values
fn average(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

/// Calculate standard deviation
fn std_dev(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let avg = average(values);
    let variance = values.iter().map(|v| (v - avg).powi(2)).sum::<f64>() / values.len() as f64;
    variance.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_frame(index: u32, start_ms: i64, pitch: f64, yaw: f64, lat: f64, lon: f64) -> TelemetryFrame {
        TelemetryFrame {
            index,
            start_time_ms: start_ms,
            end_time_ms: start_ms + 1000,
            timestamp: None,
            iso: Some(100),
            shutter: None,
            fnum: None,
            ev: None,
            color_temp: None,
            color_mode: None,
            focal_len: None,
            latitude: Some(lat),
            longitude: Some(lon),
            altitude: Some(100.0),
            gimbal_yaw: Some(yaw),
            gimbal_pitch: Some(pitch),
            gimbal_roll: Some(0.0),
        }

        
    }

    #[test]
    fn test_gimbal_deltas() {
        let analyzer = TelemetryAnalyzer::new();
        let frames = vec![
            make_frame(1, 0, -15.0, 0.0, 40.0, -74.0),
            make_frame(2, 1000, -20.0, 10.0, 40.0, -74.0),
            make_frame(3, 2000, -25.0, 20.0, 40.0, -74.0),
        ];

        let signals = analyzer.analyze_frames(&frames);

        // Pitch goes from -15 to -25 over 2 seconds = -5 deg/sec average
        assert!((signals.gimbal_pitch_delta_avg - (-5.0)).abs() < 0.1);
        // Yaw goes from 0 to 20 over 2 seconds = 10 deg/sec average
        assert!((signals.gimbal_yaw_delta_avg - 10.0).abs() < 0.1);
    }

    #[test]
    fn test_gps_speed() {
        let analyzer = TelemetryAnalyzer::new();
        // Two points roughly 111 meters apart (0.001 degree latitude)
        let frames = vec![
            make_frame(1, 0, 0.0, 0.0, 40.0, -74.0),
            make_frame(2, 1000, 0.0, 0.0, 40.001, -74.0),
        ];

        let signals = analyzer.analyze_frames(&frames);

        // Should be approximately 111 m/s (very fast, but validates calculation)
        assert!(signals.gps_speed_avg > 100.0 && signals.gps_speed_avg < 120.0);
    }

    #[test]
    fn test_altitude_delta() {
        let analyzer = TelemetryAnalyzer::new();
        let mut frames = vec![
            make_frame(1, 0, 0.0, 0.0, 40.0, -74.0),
            make_frame(2, 1000, 0.0, 0.0, 40.0, -74.0),
        ];
        frames[0].altitude = Some(100.0);
        frames[1].altitude = Some(150.0);

        let signals = analyzer.analyze_frames(&frames);

        assert!((signals.altitude_delta - 50.0).abs() < 0.1);
    }

    #[test]
    fn test_smoothness() {
        let analyzer = TelemetryAnalyzer::new();

        // Smooth movement: consistent gimbal deltas
        let smooth_frames = vec![
            make_frame(1, 0, 0.0, 0.0, 40.0, -74.0),
            make_frame(2, 1000, 5.0, 5.0, 40.0, -74.0),
            make_frame(3, 2000, 10.0, 10.0, 40.0, -74.0),
            make_frame(4, 3000, 15.0, 15.0, 40.0, -74.0),
        ];

        // Jerky movement: inconsistent gimbal deltas
        let jerky_frames = vec![
            make_frame(1, 0, 0.0, 0.0, 40.0, -74.0),
            make_frame(2, 1000, 20.0, -10.0, 40.0, -74.0),
            make_frame(3, 2000, -5.0, 30.0, 40.0, -74.0),
            make_frame(4, 3000, 15.0, 5.0, 40.0, -74.0),
        ];

        let smooth_signals = analyzer.analyze_frames(&smooth_frames);
        let jerky_signals = analyzer.analyze_frames(&jerky_frames);

        // Smooth should have higher smoothness score
        assert!(smooth_signals.gimbal_smoothness > jerky_signals.gimbal_smoothness);
    }
}
```

## src-tauri/src/services/database.rs

```rust
use anyhow::Result;
use sqlx::{sqlite::SqlitePoolOptions, Pool, Row, Sqlite};
use std::path::Path;

use crate::models::{Flight, SourceClip, Segment};

pub struct Database {
    pool: Pool<Sqlite>,
}

impl Database {
    /// Create a new database connection
    pub async fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let db_url = format!("sqlite:{}?mode=rwc", path.as_ref().display());

        let pool = SqlitePoolOptions::new()
            .max_connections(5)
            .connect(&db_url)
            .await?;

        let db = Self { pool };
        db.run_migrations().await?;

        Ok(db)
    }

    /// Run database migrations
    async fn run_migrations(&self) -> Result<()> {
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS flights (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                import_date TEXT NOT NULL,
                source_path TEXT NOT NULL,
                location_name TEXT,
                gps_center_lat REAL,
                gps_center_lon REAL,
                total_duration_sec REAL,
                total_clips INTEGER
            )
            "#,
        )
        .execute(&self.pool)
        .await?;

        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS source_clips (
                id TEXT PRIMARY KEY,
                flight_id TEXT NOT NULL REFERENCES flights(id),
                filename TEXT NOT NULL,
                source_path TEXT NOT NULL,
                proxy_path TEXT,
                proxy_source TEXT,
                srt_path TEXT,
                duration_sec REAL,
                resolution_width INTEGER,
                resolution_height INTEGER,
                framerate REAL,
                recorded_at TEXT
            )
            "#,
        )
        .execute(&self.pool)
        .await?;

        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS segments (
                id TEXT PRIMARY KEY,
                source_clip_id TEXT NOT NULL REFERENCES source_clips(id),
                start_time_ms INTEGER NOT NULL,
                end_time_ms INTEGER NOT NULL,
                duration_ms INTEGER NOT NULL,
                thumbnail_path TEXT,
                motion_magnitude REAL,
                gimbal_pitch_delta_avg REAL,
                gimbal_yaw_delta_avg REAL,
                gimbal_smoothness REAL,
                altitude_delta REAL,
                gps_speed_avg REAL,
                iso_avg REAL,
                visual_quality REAL,
                has_scene_change INTEGER,
                is_selected INTEGER DEFAULT 0,
                user_adjusted_start_ms INTEGER,
                user_adjusted_end_ms INTEGER
            )
            "#,
        )
        .execute(&self.pool)
        .await?;

        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS selections (
                id TEXT PRIMARY KEY,
                flight_id TEXT NOT NULL REFERENCES flights(id),
                segment_id TEXT NOT NULL REFERENCES segments(id),
                sequence_order INTEGER NOT NULL,
                added_at TEXT NOT NULL
            )
            "#,
        )
        .execute(&self.pool)
        .await?;

        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS segment_scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                segment_id TEXT NOT NULL REFERENCES segments(id),
                profile_id TEXT NOT NULL,
                score REAL NOT NULL,
                UNIQUE(segment_id, profile_id)
            )
            "#,
        )
        .execute(&self.pool)
        .await?;

        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_segment_scores_segment ON segment_scores(segment_id)",
        )
        .execute(&self.pool)
        .await?;

        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_segment_scores_profile ON segment_scores(profile_id, score DESC)",
        )
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    // Flight operations
    pub async fn insert_flight(&self, flight: &Flight) -> Result<()> {
        sqlx::query(
            r#"
            INSERT INTO flights (id, name, import_date, source_path, location_name, gps_center_lat, gps_center_lon, total_duration_sec, total_clips)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            "#,
        )
        .bind(&flight.id)
        .bind(&flight.name)
        .bind(flight.import_date.to_rfc3339())
        .bind(&flight.source_path)
        .bind(&flight.location_name)
        .bind(flight.gps_center_lat)
        .bind(flight.gps_center_lon)
        .bind(flight.total_duration_sec)
        .bind(flight.total_clips)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    pub async fn get_flight(&self, id: &str) -> Result<Option<Flight>> {
        let row = sqlx::query(
            "SELECT id, name, import_date, source_path, location_name, gps_center_lat, gps_center_lon, total_duration_sec, total_clips FROM flights WHERE id = ?"
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await?;

        match row {
            Some(row) => {
                let import_date_str: String = row.get("import_date");
                Ok(Some(Flight {
                    id: row.get("id"),
                    name: row.get("name"),
                    import_date: chrono::DateTime::parse_from_rfc3339(&import_date_str)?.with_timezone(&chrono::Utc),
                    source_path: row.get("source_path"),
                    location_name: row.get("location_name"),
                    gps_center_lat: row.get("gps_center_lat"),
                    gps_center_lon: row.get("gps_center_lon"),
                    total_duration_sec: row.get("total_duration_sec"),
                    total_clips: row.get("total_clips"),
                }))
            }
            None => Ok(None),
        }
    }

    pub async fn list_flights(&self) -> Result<Vec<Flight>> {
        let rows = sqlx::query(
            "SELECT id, name, import_date, source_path, location_name, gps_center_lat, gps_center_lon, total_duration_sec, total_clips FROM flights ORDER BY import_date DESC"
        )
        .fetch_all(&self.pool)
        .await?;

        let mut flights = Vec::new();
        for row in rows {
            let import_date_str: String = row.get("import_date");
            flights.push(Flight {
                id: row.get("id"),
                name: row.get("name"),
                import_date: chrono::DateTime::parse_from_rfc3339(&import_date_str)?.with_timezone(&chrono::Utc),
                source_path: row.get("source_path"),
                location_name: row.get("location_name"),
                gps_center_lat: row.get("gps_center_lat"),
                gps_center_lon: row.get("gps_center_lon"),
                total_duration_sec: row.get("total_duration_sec"),
                total_clips: row.get("total_clips"),
            });
        }

        Ok(flights)
    }

    // SourceClip operations
    pub async fn insert_clip(&self, clip: &SourceClip) -> Result<()> {
        sqlx::query(
            r#"
            INSERT INTO source_clips (id, flight_id, filename, source_path, proxy_path, proxy_source, srt_path, duration_sec, resolution_width, resolution_height, framerate, recorded_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            "#,
        )
        .bind(&clip.id)
        .bind(&clip.flight_id)
        .bind(&clip.filename)
        .bind(&clip.source_path)
        .bind(&clip.proxy_path)
        .bind(&clip.proxy_source)
        .bind(&clip.srt_path)
        .bind(clip.duration_sec)
        .bind(clip.resolution_width)
        .bind(clip.resolution_height)
        .bind(clip.framerate)
        .bind(clip.recorded_at.map(|dt| dt.to_rfc3339()))
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    pub async fn get_clips_for_flight(&self, flight_id: &str) -> Result<Vec<SourceClip>> {
        let rows = sqlx::query(
            "SELECT id, flight_id, filename, source_path, proxy_path, proxy_source, srt_path, duration_sec, resolution_width, resolution_height, framerate, recorded_at FROM source_clips WHERE flight_id = ?"
        )
        .bind(flight_id)
        .fetch_all(&self.pool)
        .await?;

        let mut clips = Vec::new();
        for row in rows {
            let recorded_at_str: Option<String> = row.get("recorded_at");
            clips.push(SourceClip {
                id: row.get("id"),
                flight_id: row.get("flight_id"),
                filename: row.get("filename"),
                source_path: row.get("source_path"),
                proxy_path: row.get("proxy_path"),
                proxy_source: row.get("proxy_source"),
                srt_path: row.get("srt_path"),
                duration_sec: row.get("duration_sec"),
                resolution_width: row.get("resolution_width"),
                resolution_height: row.get("resolution_height"),
                framerate: row.get("framerate"),
                recorded_at: recorded_at_str.and_then(|s| chrono::DateTime::parse_from_rfc3339(&s).ok().map(|dt| dt.with_timezone(&chrono::Utc))),
            });
        }

        Ok(clips)
    }

    pub async fn update_clip_proxy(&self, clip_id: &str, proxy_path: &str, proxy_source: &str) -> Result<()> {
        sqlx::query("UPDATE source_clips SET proxy_path = ?, proxy_source = ? WHERE id = ?")
            .bind(proxy_path)
            .bind(proxy_source)
            .bind(clip_id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    /// Delete a flight and all associated data (clips, segments, scores)
    pub async fn delete_flight(&self, flight_id: &str) -> Result<()> {
        // Get clip IDs for this flight to clean up segments and scores
        let clip_ids: Vec<String> = sqlx::query_scalar(
            "SELECT id FROM source_clips WHERE flight_id = ?"
        )
        .bind(flight_id)
        .fetch_all(&self.pool)
        .await?;

        // Delete segment scores for all clips in this flight
        for clip_id in &clip_ids {
            sqlx::query("DELETE FROM segment_scores WHERE segment_id IN (SELECT id FROM segments WHERE source_clip_id = ?)")
                .bind(clip_id)
                .execute(&self.pool)
                .await?;

            sqlx::query("DELETE FROM segments WHERE source_clip_id = ?")
                .bind(clip_id)
                .execute(&self.pool)
                .await?;
        }

        // Delete clips
        sqlx::query("DELETE FROM source_clips WHERE flight_id = ?")
            .bind(flight_id)
            .execute(&self.pool)
            .await?;

        // Delete flight
        sqlx::query("DELETE FROM flights WHERE id = ?")
            .bind(flight_id)
            .execute(&self.pool)
            .await?;

        Ok(())
    }

    // Segment operations
    pub async fn insert_segment(&self, segment: &Segment) -> Result<()> {
        sqlx::query(
            r#"
            INSERT INTO segments (id, source_clip_id, start_time_ms, end_time_ms, duration_ms, thumbnail_path, motion_magnitude, gimbal_pitch_delta_avg, gimbal_yaw_delta_avg, gimbal_smoothness, altitude_delta, gps_speed_avg, iso_avg, visual_quality, has_scene_change, is_selected, user_adjusted_start_ms, user_adjusted_end_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            "#,
        )
        .bind(&segment.id)
        .bind(&segment.source_clip_id)
        .bind(segment.start_time_ms)
        .bind(segment.end_time_ms)
        .bind(segment.duration_ms)
        .bind(&segment.thumbnail_path)
        .bind(segment.motion_magnitude)
        .bind(segment.gimbal_pitch_delta_avg)
        .bind(segment.gimbal_yaw_delta_avg)
        .bind(segment.gimbal_smoothness)
        .bind(segment.altitude_delta)
        .bind(segment.gps_speed_avg)
        .bind(segment.iso_avg)
        .bind(segment.visual_quality)
        .bind(segment.has_scene_change)
        .bind(segment.is_selected)
        .bind(segment.user_adjusted_start_ms)
        .bind(segment.user_adjusted_end_ms)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    pub async fn get_segments_for_clip(&self, clip_id: &str) -> Result<Vec<Segment>> {
        let rows = sqlx::query(
            "SELECT id, source_clip_id, start_time_ms, end_time_ms, duration_ms, thumbnail_path, motion_magnitude, gimbal_pitch_delta_avg, gimbal_yaw_delta_avg, gimbal_smoothness, altitude_delta, gps_speed_avg, iso_avg, visual_quality, has_scene_change, is_selected, user_adjusted_start_ms, user_adjusted_end_ms FROM segments WHERE source_clip_id = ? ORDER BY start_time_ms"
        )
        .bind(clip_id)
        .fetch_all(&self.pool)
        .await?;

        Ok(Self::rows_to_segments(rows))
    }

    pub async fn get_segment(&self, segment_id: &str) -> Result<Option<Segment>> {
        let row = sqlx::query(
            "SELECT id, source_clip_id, start_time_ms, end_time_ms, duration_ms, thumbnail_path, motion_magnitude, gimbal_pitch_delta_avg, gimbal_yaw_delta_avg, gimbal_smoothness, altitude_delta, gps_speed_avg, iso_avg, visual_quality, has_scene_change, is_selected, user_adjusted_start_ms, user_adjusted_end_ms FROM segments WHERE id = ?"
        )
        .bind(segment_id)
        .fetch_optional(&self.pool)
        .await?;

        match row {
            Some(row) => {
                let segments = Self::rows_to_segments(vec![row]);
                Ok(segments.into_iter().next())
            }
            None => Ok(None),
        }
    }

    pub async fn get_clip(&self, clip_id: &str) -> Result<Option<SourceClip>> {
        let row = sqlx::query(
            "SELECT id, flight_id, filename, source_path, proxy_path, proxy_source, srt_path, duration_sec, resolution_width, resolution_height, framerate, recorded_at FROM source_clips WHERE id = ?"
        )
        .bind(clip_id)
        .fetch_optional(&self.pool)
        .await?;

        match row {
            Some(row) => {
                let recorded_at_str: Option<String> = row.get("recorded_at");
                Ok(Some(SourceClip {
                    id: row.get("id"),
                    flight_id: row.get("flight_id"),
                    filename: row.get("filename"),
                    source_path: row.get("source_path"),
                    proxy_path: row.get("proxy_path"),
                    proxy_source: row.get("proxy_source"),
                    srt_path: row.get("srt_path"),
                    duration_sec: row.get("duration_sec"),
                    resolution_width: row.get("resolution_width"),
                    resolution_height: row.get("resolution_height"),
                    framerate: row.get("framerate"),
                    recorded_at: recorded_at_str.and_then(|s| chrono::DateTime::parse_from_rfc3339(&s).ok().map(|dt| dt.with_timezone(&chrono::Utc))),
                }))
            }
            None => Ok(None),
        }
    }

    pub async fn delete_segments_for_clip(&self, clip_id: &str) -> Result<()> {
        // First delete scores for these segments
        sqlx::query(
            "DELETE FROM segment_scores WHERE segment_id IN (SELECT id FROM segments WHERE source_clip_id = ?)"
        )
        .bind(clip_id)
        .execute(&self.pool)
        .await?;

        // Then delete the segments
        sqlx::query("DELETE FROM segments WHERE source_clip_id = ?")
            .bind(clip_id)
            .execute(&self.pool)
            .await?;

        Ok(())
    }

    pub async fn insert_segment_score(&self, segment_id: &str, profile_id: &str, score: f64) -> Result<()> {
        sqlx::query(
            "INSERT OR REPLACE INTO segment_scores (segment_id, profile_id, score) VALUES (?, ?, ?)"
        )
        .bind(segment_id)
        .bind(profile_id)
        .bind(score)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    pub async fn get_segment_scores(&self, segment_id: &str) -> Result<std::collections::HashMap<String, f64>> {
        let rows = sqlx::query(
            "SELECT profile_id, score FROM segment_scores WHERE segment_id = ?"
        )
        .bind(segment_id)
        .fetch_all(&self.pool)
        .await?;

        let mut scores = std::collections::HashMap::new();
        for row in rows {
            let profile_id: String = row.get("profile_id");
            let score: f64 = row.get("score");
            scores.insert(profile_id, score);
        }

        Ok(scores)
    }

    pub async fn get_top_segments_for_flight(&self, flight_id: &str, profile_id: &str, limit: u32) -> Result<Vec<Segment>> {
        let rows = sqlx::query(
            r#"
            SELECT s.id, s.source_clip_id, s.start_time_ms, s.end_time_ms, s.duration_ms, s.thumbnail_path,
                   s.motion_magnitude, s.gimbal_pitch_delta_avg, s.gimbal_yaw_delta_avg, s.gimbal_smoothness,
                   s.altitude_delta, s.gps_speed_avg, s.iso_avg, s.visual_quality, s.has_scene_change,
                   s.is_selected, s.user_adjusted_start_ms, s.user_adjusted_end_ms
            FROM segments s
            JOIN source_clips c ON s.source_clip_id = c.id
            JOIN segment_scores sc ON s.id = sc.segment_id
            WHERE c.flight_id = ? AND sc.profile_id = ?
            ORDER BY sc.score DESC
            LIMIT ?
            "#
        )
        .bind(flight_id)
        .bind(profile_id)
        .bind(limit)
        .fetch_all(&self.pool)
        .await?;

        Ok(Self::rows_to_segments(rows))
    }

    fn rows_to_segments(rows: Vec<sqlx::sqlite::SqliteRow>) -> Vec<Segment> {
        rows.into_iter()
            .map(|row| {
                let has_scene_change: Option<i32> = row.get("has_scene_change");
                let is_selected: i32 = row.get("is_selected");
                Segment {
                    id: row.get("id"),
                    source_clip_id: row.get("source_clip_id"),
                    start_time_ms: row.get("start_time_ms"),
                    end_time_ms: row.get("end_time_ms"),
                    duration_ms: row.get("duration_ms"),
                    thumbnail_path: row.get("thumbnail_path"),
                    motion_magnitude: row.get("motion_magnitude"),
                    gimbal_pitch_delta_avg: row.get("gimbal_pitch_delta_avg"),
                    gimbal_yaw_delta_avg: row.get("gimbal_yaw_delta_avg"),
                    gimbal_smoothness: row.get("gimbal_smoothness"),
                    altitude_delta: row.get("altitude_delta"),
                    gps_speed_avg: row.get("gps_speed_avg"),
                    iso_avg: row.get("iso_avg"),
                    visual_quality: row.get("visual_quality"),
                    has_scene_change: has_scene_change.map(|v| v != 0),
                    is_selected: is_selected != 0,
                    user_adjusted_start_ms: row.get("user_adjusted_start_ms"),
                    user_adjusted_end_ms: row.get("user_adjusted_end_ms"),
                }
            })
            .collect()
    }
}
```

## src-tauri/src/services/director.rs

```rust
use anyhow::{anyhow, Result};
use base64::{engine::general_purpose::STANDARD as BASE64, Engine};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

/// Segment info passed to Claude for editing decisions
#[derive(Debug, Clone, Serialize)]
pub struct SegmentContext {
    pub id: String,
    pub duration_sec: f64,
    pub start_ms: i64,
    pub end_ms: i64,
    pub gimbal_pitch_delta: f64,
    pub gimbal_yaw_delta: f64,
    pub gimbal_smoothness: f64,
    pub gps_speed: f64,
    pub altitude_delta: f64,
    pub score: f64,
}

/// Claude's response for a single clip decision
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DirectorClipDecision {
    pub segment_id: String,
    pub sequence_order: i32,
    pub in_point_ms: i64,
    pub out_point_ms: i64,
    pub transition_to_next: String,
    pub transition_duration_ms: i64,
    pub reasoning: String,
}

/// Claude's full response
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DirectorResponse {
    pub edit_sequence: Vec<DirectorClipDecision>,
    pub total_duration_sec: f64,
    pub style_notes: String,
}

/// Request body for Claude API
#[derive(Debug, Serialize)]
struct ClaudeRequest {
    model: String,
    max_tokens: u32,
    messages: Vec<ClaudeMessage>,
}

#[derive(Debug, Serialize)]
struct ClaudeMessage {
    role: String,
    content: Vec<ClaudeContent>,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type")]
enum ClaudeContent {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image")]
    Image { source: ImageSource },
}

#[derive(Debug, Serialize)]
struct ImageSource {
    #[serde(rename = "type")]
    source_type: String,
    media_type: String,
    data: String,
}

/// Response from Claude API
#[derive(Debug, Deserialize)]
struct ClaudeApiResponse {
    content: Vec<ClaudeResponseContent>,
}

#[derive(Debug, Deserialize)]
struct ClaudeResponseContent {
    text: Option<String>,
}

pub struct Director {
    api_key: String,
}

impl Director {
    pub fn new(api_key: String) -> Self {
        Self { api_key }
    }

    /// Generate an edit sequence based on the director's natural language instructions
    pub async fn generate_edit(
        &self,
        prompt: &str,
        segments: Vec<SegmentContext>,
        thumbnail_paths: Vec<String>,
        target_duration_sec: Option<f64>,
    ) -> Result<DirectorResponse> {
        // Build the content array
        let mut content: Vec<ClaudeContent> = Vec::new();

        // System context and user prompt
        let system_text = self.build_system_prompt(&segments, target_duration_sec);
        let user_text = format!(
            "{}\n\nDIRECTOR'S INSTRUCTIONS:\n\"{}\"",
            system_text, prompt
        );
        content.push(ClaudeContent::Text { text: user_text });

        // Add thumbnails as base64 images (limit to first 20 to control costs)
        let max_thumbnails = 20.min(thumbnail_paths.len());
        for (i, thumb_path) in thumbnail_paths.iter().take(max_thumbnails).enumerate() {
            if let Ok(image_data) = self.load_thumbnail_as_base64(thumb_path) {
                content.push(ClaudeContent::Text {
                    text: format!("Segment {} thumbnail (ID: {}):", i + 1, segments.get(i).map(|s| s.id.as_str()).unwrap_or("unknown")),
                });
                content.push(ClaudeContent::Image {
                    source: ImageSource {
                        source_type: "base64".to_string(),
                        media_type: "image/jpeg".to_string(),
                        data: image_data,
                    },
                });
            }
        }

        // Add response format instructions
        content.push(ClaudeContent::Text {
            text: self.get_response_format_instructions(),
        });

        let request = ClaudeRequest {
            model: "claude-sonnet-4-20250514".to_string(),
            max_tokens: 4096,
            messages: vec![ClaudeMessage {
                role: "user".to_string(),
                content,
            }],
        };

        // Make API call
        let client = reqwest::Client::new();
        let response = client
            .post("https://api.anthropic.com/v1/messages")
            .header("Content-Type", "application/json")
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .json(&request)
            .send()
            .await
            .map_err(|e| anyhow!("Failed to call Claude API: {}", e))?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(anyhow!("Claude API error ({}): {}", status, error_text));
        }

        let api_response: ClaudeApiResponse = response
            .json()
            .await
            .map_err(|e| anyhow!("Failed to parse Claude response: {}", e))?;

        // Extract text response
        let response_text = api_response
            .content
            .iter()
            .find_map(|c| c.text.clone())
            .ok_or_else(|| anyhow!("No text in Claude response"))?;

        // Parse JSON from response
        self.parse_director_response(&response_text)
    }

    fn build_system_prompt(&self, segments: &[SegmentContext], target_duration: Option<f64>) -> String {
        let segments_json = serde_json::to_string_pretty(segments).unwrap_or_default();

        let duration_instruction = if let Some(dur) = target_duration {
            format!("Target duration: {} seconds. Select and trim clips to achieve this.", dur)
        } else {
            "No specific duration target - include what best matches the vision.".to_string()
        };

        format!(
            r#"You are a professional drone footage editor creating highlight reels. You'll receive:
1. Segment metadata (telemetry data from the drone)
2. Thumbnail images showing what each segment looks like

Your job is to select clips, determine their order, choose transitions, and set in/out points to create a cohesive edit.

AVAILABLE SEGMENTS:
{}

{}

EDITING GUIDELINES:
- gimbal_pitch_delta: Negative = tilting down (reveals), Positive = tilting up
- gimbal_yaw_delta: Positive = panning right, Negative = panning left
- gimbal_smoothness: 0-1, higher = smoother camera movement (better for cinematic)
- gps_speed: m/s, higher = faster drone movement (good for action)
- altitude_delta: Positive = ascending, Negative = descending
- score: Pre-computed quality score (0-100)

TRANSITION TYPES:
- "cut" - Instant cut (good for matching motion, action sequences)
- "dissolve" - Crossfade (good for different scenes, time passing, cinematic feel)
- "dip_black" - Fade through black (good for major scene changes, dramatic moments)

GENERAL PRINCIPLES:
- Match motion direction between clips when using cuts
- Use dissolves when scene content is very different
- Start with establishing shots, end with resolution
- Build energy through the edit
- Smoother gimbal = longer holds; jerky = quicker cuts"#,
            segments_json, duration_instruction
        )
    }

    fn get_response_format_instructions(&self) -> String {
        r#"

Respond with ONLY valid JSON in this exact format (no markdown, no explanation outside JSON):
{
  "edit_sequence": [
    {
      "segment_id": "the segment id",
      "sequence_order": 1,
      "in_point_ms": 0,
      "out_point_ms": 5000,
      "transition_to_next": "dissolve",
      "transition_duration_ms": 500,
      "reasoning": "Brief explanation of why this clip and edit choice"
    }
  ],
  "total_duration_sec": 30.0,
  "style_notes": "Overall notes about the edit"
}"#.to_string()
    }

    fn load_thumbnail_as_base64(&self, path: &str) -> Result<String> {
        let path = Path::new(path);
        if !path.exists() {
            return Err(anyhow!("Thumbnail not found: {}", path.display()));
        }

        let bytes = fs::read(path)?;
        Ok(BASE64.encode(&bytes))
    }

    fn parse_director_response(&self, response: &str) -> Result<DirectorResponse> {
        // Try to find JSON in the response (Claude sometimes adds text around it)
        let json_start = response.find('{');
        let json_end = response.rfind('}');

        match (json_start, json_end) {
            (Some(start), Some(end)) if end > start => {
                let json_str = &response[start..=end];
                serde_json::from_str(json_str)
                    .map_err(|e| anyhow!("Failed to parse director response JSON: {}\n\nRaw response:\n{}", e, json_str))
            }
            _ => Err(anyhow!("No valid JSON found in response:\n{}", response)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_response() {
        let director = Director::new("test".to_string());
        let response = r#"{
            "edit_sequence": [
                {
                    "segment_id": "seg_001",
                    "sequence_order": 1,
                    "in_point_ms": 0,
                    "out_point_ms": 5000,
                    "transition_to_next": "dissolve",
                    "transition_duration_ms": 500,
                    "reasoning": "Test"
                }
            ],
            "total_duration_sec": 5.0,
            "style_notes": "Test notes"
        }"#;

        let result = director.parse_director_response(response);
        assert!(result.is_ok());
        let parsed = result.unwrap();
        assert_eq!(parsed.edit_sequence.len(), 1);
        assert_eq!(parsed.edit_sequence[0].segment_id, "seg_001");
    }
}
```

## src-tauri/src/services/ffmpeg.rs

```rust
use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::process::Command;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoInfo {
    pub duration_sec: f64,
    pub width: i32,
    pub height: i32,
    pub framerate: f64,
    pub codec: String,
}

pub struct FFmpeg {
    ffmpeg_path: String,
    ffprobe_path: String,
}

impl FFmpeg {
    pub fn new() -> Result<Self> {
        // Try to find ffmpeg and ffprobe in PATH
        let ffmpeg_path = which_command("ffmpeg")?;
        let ffprobe_path = which_command("ffprobe")?;

        Ok(Self {
            ffmpeg_path,
            ffprobe_path,
        })
    }

    /// Get video information using ffprobe
    pub fn probe<P: AsRef<Path>>(&self, input: P) -> Result<VideoInfo> {
        let output = Command::new(&self.ffprobe_path)
            .args([
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                "-show_streams",
                "-select_streams", "v:0",
            ])
            .arg(input.as_ref())
            .output()
            .context("Failed to execute ffprobe")?;

        if !output.status.success() {
            return Err(anyhow!(
                "ffprobe failed: {}",
                String::from_utf8_lossy(&output.stderr)
            ));
        }

        let json: serde_json::Value = serde_json::from_slice(&output.stdout)?;

        // Extract video stream info
        let stream = json["streams"]
            .as_array()
            .and_then(|s| s.first())
            .ok_or_else(|| anyhow!("No video stream found"))?;

        let width = stream["width"].as_i64().unwrap_or(0) as i32;
        let height = stream["height"].as_i64().unwrap_or(0) as i32;
        let codec = stream["codec_name"]
            .as_str()
            .unwrap_or("unknown")
            .to_string();

        // Parse framerate (e.g., "30000/1001" or "30")
        let framerate = self.parse_framerate(
            stream["r_frame_rate"]
                .as_str()
                .or_else(|| stream["avg_frame_rate"].as_str())
                .unwrap_or("0"),
        );

        // Get duration from format
        let duration_sec = json["format"]["duration"]
            .as_str()
            .and_then(|d| d.parse::<f64>().ok())
            .unwrap_or(0.0);

        Ok(VideoInfo {
            duration_sec,
            width,
            height,
            framerate,
            codec,
        })
    }

    fn parse_framerate(&self, fps_str: &str) -> f64 {
        if fps_str.contains('/') {
            let parts: Vec<&str> = fps_str.split('/').collect();
            if parts.len() == 2 {
                let num: f64 = parts[0].parse().unwrap_or(0.0);
                let den: f64 = parts[1].parse().unwrap_or(1.0);
                if den != 0.0 {
                    return num / den;
                }
            }
        }
        fps_str.parse().unwrap_or(0.0)
    }

    /// Generate a proxy video with hardware acceleration
    pub fn generate_proxy<P: AsRef<Path>>(&self, input: P, output: P) -> Result<()> {
        let status = Command::new(&self.ffmpeg_path)
            .args([
                "-hwaccel", "videotoolbox",
                "-i",
            ])
            .arg(input.as_ref())
            .args([
                "-vf", "scale=1280:720",
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-c:a", "aac",
                "-b:a", "128k",
                "-movflags", "+faststart",
                "-y", // Overwrite output
            ])
            .arg(output.as_ref())
            .status()
            .context("Failed to execute ffmpeg")?;

        if !status.success() {
            return Err(anyhow!("FFmpeg proxy generation failed"));
        }

        Ok(())
    }

    /// Extract thumbnails at 1fps
    pub fn extract_thumbnails<P: AsRef<Path>>(
        &self,
        input: P,
        output_dir: P,
        prefix: &str,
    ) -> Result<Vec<String>> {
        let output_pattern = output_dir
            .as_ref()
            .join(format!("{prefix}_%04d.jpg"));

        let status = Command::new(&self.ffmpeg_path)
            .args([
                "-hwaccel", "videotoolbox",
                "-i",
            ])
            .arg(input.as_ref())
            .args([
                "-vf", "fps=1,scale=320:180",
                "-q:v", "3",
                "-y",
            ])
            .arg(&output_pattern)
            .status()
            .context("Failed to execute ffmpeg for thumbnails")?;

        if !status.success() {
            return Err(anyhow!("FFmpeg thumbnail extraction failed"));
        }

        // List generated thumbnails
        let mut thumbnails = Vec::new();
        let output_dir = output_dir.as_ref();
        if output_dir.is_dir() {
            for entry in std::fs::read_dir(output_dir)? {
                let entry = entry?;
                let filename = entry.file_name().to_string_lossy().to_string();
                if filename.starts_with(prefix) && filename.ends_with(".jpg") {
                    thumbnails.push(entry.path().to_string_lossy().to_string());
                }
            }
        }
        thumbnails.sort();

        Ok(thumbnails)
    }

    /// Fast export using stream copy (keyframe-aligned cuts)
    pub fn export_fast<P: AsRef<Path>>(
        &self,
        input: P,
        output: P,
        start_sec: f64,
        end_sec: f64,
    ) -> Result<()> {
        let start_time = format_time(start_sec);
        let end_time = format_time(end_sec);

        let status = Command::new(&self.ffmpeg_path)
            .args(["-ss", &start_time, "-to", &end_time, "-i"])
            .arg(input.as_ref())
            .args([
                "-c", "copy",
                "-avoid_negative_ts", "make_zero",
                "-y",
            ])
            .arg(output.as_ref())
            .status()
            .context("Failed to execute ffmpeg for export")?;

        if !status.success() {
            return Err(anyhow!("FFmpeg fast export failed"));
        }

        Ok(())
    }

    /// Check if a video file has an audio stream
    fn check_has_audio(&self, path: &str) -> bool {
        let output = Command::new(&self.ffprobe_path)
            .args([
                "-v", "quiet",
                "-select_streams", "a",
                "-show_entries", "stream=codec_type",
                "-of", "csv=p=0",
            ])
            .arg(path)
            .output();

        match output {
            Ok(out) => !out.stdout.is_empty(),
            Err(_) => false, // Assume no audio if probe fails
        }
    }

    /// Precise export with re-encode (frame-exact cuts)
    pub fn export_precise<P: AsRef<Path>>(
        &self,
        input: P,
        output: P,
        start_sec: f64,
        end_sec: f64,
    ) -> Result<()> {
        let start_time = format_time(start_sec);
        let end_time = format_time(end_sec);

        let status = Command::new(&self.ffmpeg_path)
            .args([
                "-hwaccel", "videotoolbox",
                "-ss", &start_time,
                "-to", &end_time,
                "-i",
            ])
            .arg(input.as_ref())
            .args([
                "-c:v", "libx264",
                "-crf", "18",
                "-preset", "medium",
                "-c:a", "aac",
                "-b:a", "192k",
                "-y",
            ])
            .arg(output.as_ref())
            .status()
            .context("Failed to execute ffmpeg for precise export")?;

        if !status.success() {
            return Err(anyhow!("FFmpeg precise export failed"));
        }

        Ok(())
    }
}

/// Clip definition for concatenation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConcatClip {
    pub input_path: String,
    pub start_sec: f64,
    pub end_sec: f64,
    pub transition_type: String,      // "cut", "dissolve", "dip_black"
    pub transition_duration_ms: i64,
}

impl FFmpeg {
    /// Concatenate multiple clips with transitions into a single output
    pub fn concat_with_transitions(
        &self,
        clips: Vec<ConcatClip>,
        output_path: &str,
        use_hw_accel: bool,
    ) -> Result<()> {
        if clips.is_empty() {
            return Err(anyhow!("No clips to concatenate"));
        }

        // For a single clip or all "cut" transitions, use simple concat
        let has_transitions = clips.iter().skip(1).any(|c| c.transition_type != "cut");

        if clips.len() == 1 || !has_transitions {
            return self.concat_simple(&clips, output_path, use_hw_accel);
        }

        // Use filter_complex with xfade for transitions
        self.concat_with_xfade(&clips, output_path, use_hw_accel)
    }

    /// Simple concatenation without transitions (or all cuts)
    fn concat_simple(&self, clips: &[ConcatClip], output_path: &str, use_hw_accel: bool) -> Result<()> {
        // Create a temp file for the concat list
        let temp_dir = std::env::temp_dir();
        let concat_list_path = temp_dir.join(format!("skyclip_concat_{}.txt", uuid::Uuid::new_v4()));
        let temp_clips_dir = temp_dir.join(format!("skyclip_clips_{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&temp_clips_dir)?;

        // First, extract each segment to a temp file
        let mut temp_files = Vec::new();
        for (i, clip) in clips.iter().enumerate() {
            let temp_output = temp_clips_dir.join(format!("clip_{:04}.mp4", i));
            let start_time = format_time(clip.start_sec);
            let end_time = format_time(clip.end_sec);

            let mut args = vec![];
            if use_hw_accel {
                args.extend(["-hwaccel", "videotoolbox"]);
            }
            args.extend([
                "-ss", &start_time,
                "-to", &end_time,
                "-i", &clip.input_path,
                "-c:v", "libx264",
                "-crf", "18",
                "-preset", "fast",
                "-c:a", "aac",
                "-b:a", "192k",
                "-y",
            ]);

            let status = Command::new(&self.ffmpeg_path)
                .args(&args)
                .arg(&temp_output)
                .status()
                .context("Failed to extract clip segment")?;

            if !status.success() {
                // Clean up
                let _ = std::fs::remove_dir_all(&temp_clips_dir);
                return Err(anyhow!("Failed to extract clip segment {}", i));
            }

            temp_files.push(temp_output);
        }

        // Write concat list
        let concat_content: String = temp_files
            .iter()
            .map(|p| format!("file '{}'\n", p.to_string_lossy()))
            .collect();
        std::fs::write(&concat_list_path, &concat_content)?;

        // Concatenate
        let status = Command::new(&self.ffmpeg_path)
            .args([
                "-f", "concat",
                "-safe", "0",
                "-i",
            ])
            .arg(&concat_list_path)
            .args(["-c", "copy", "-y"])
            .arg(output_path)
            .status()
            .context("Failed to concatenate clips")?;

        // Clean up temp files
        let _ = std::fs::remove_file(&concat_list_path);
        let _ = std::fs::remove_dir_all(&temp_clips_dir);

        if !status.success() {
            return Err(anyhow!("FFmpeg concatenation failed"));
        }

        Ok(())
    }

    /// Concatenation with xfade transitions
    fn concat_with_xfade(&self, clips: &[ConcatClip], output_path: &str, use_hw_accel: bool) -> Result<()> {
        // First extract all clips to temp files with consistent encoding
        let temp_dir = std::env::temp_dir();
        let temp_clips_dir = temp_dir.join(format!("skyclip_xfade_{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&temp_clips_dir)?;

        let mut temp_files = Vec::new();
        let mut clip_durations = Vec::new();

        for (i, clip) in clips.iter().enumerate() {
            let temp_output = temp_clips_dir.join(format!("clip_{:04}.mp4", i));
            let duration = clip.end_sec - clip.start_sec;
            clip_durations.push(duration);
            let start_time = format_time(clip.start_sec);
            let end_time = format_time(clip.end_sec);

            // Check if source has audio
            let has_audio = self.check_has_audio(&clip.input_path);

            // Build command with or without audio handling
            let status = if has_audio {
                // Source has audio - normal encoding
                let mut args = vec![];
                if use_hw_accel {
                    args.push("-hwaccel");
                    args.push("videotoolbox");
                }
                Command::new(&self.ffmpeg_path)
                    .args(&args)
                    .args(["-ss", &start_time, "-to", &end_time, "-i", &clip.input_path])
                    .args(["-vf", "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,fps=30,format=yuv420p"])
                    .args(["-c:v", "libx264", "-crf", "18", "-preset", "fast", "-pix_fmt", "yuv420p"])
                    .args(["-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2"])
                    .args(["-y"])
                    .arg(&temp_output)
                    .status()
                    .context("Failed to extract clip segment for xfade")?
            } else {
                // No audio - generate silent audio track
                let mut args = vec![];
                if use_hw_accel {
                    args.push("-hwaccel");
                    args.push("videotoolbox");
                }
                let duration_str = format!("{}", duration);
                let anullsrc = format!("anullsrc=channel_layout=stereo:sample_rate=48000");
                Command::new(&self.ffmpeg_path)
                    .args(&args)
                    .args(["-ss", &start_time, "-to", &end_time, "-i", &clip.input_path])
                    .args(["-f", "lavfi", "-t", &duration_str, "-i", &anullsrc])
                    .args(["-vf", "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,fps=30,format=yuv420p"])
                    .args(["-map", "0:v:0", "-map", "1:a:0"])
                    .args(["-c:v", "libx264", "-crf", "18", "-preset", "fast", "-pix_fmt", "yuv420p"])
                    .args(["-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2"])
                    .args(["-shortest", "-y"])
                    .arg(&temp_output)
                    .status()
                    .context("Failed to extract clip segment for xfade")?
            };

            if !status.success() {
                let _ = std::fs::remove_dir_all(&temp_clips_dir);
                return Err(anyhow!("Failed to extract clip {} for xfade", i));
            }

            temp_files.push(temp_output);
        }

        // Build the filter_complex string
        let mut filter_parts = Vec::new();
        let mut video_labels = Vec::new();
        let mut audio_labels = Vec::new();

        // Input labels
        for i in 0..temp_files.len() {
            video_labels.push(format!("[{}:v]", i));
            audio_labels.push(format!("[{}:a]", i));
        }

        // Build xfade chain for video
        if clips.len() >= 2 {
            let mut current_video = video_labels[0].clone();
            let mut cumulative_duration = clip_durations[0];

            for i in 1..clips.len() {
                let transition = &clips[i].transition_type;
                let trans_duration_sec = clips[i].transition_duration_ms as f64 / 1000.0;
                let trans_duration_sec = trans_duration_sec.max(0.1).min(2.0); // Clamp between 0.1 and 2 seconds

                // Calculate offset (when transition starts)
                let offset = (cumulative_duration - trans_duration_sec).max(0.0);

                let xfade_transition = match transition.as_str() {
                    "dissolve" => "fade",
                    "dip_black" => "fadeblack",
                    _ => "fade", // Default to fade for unknown transitions
                };

                let out_label = format!("[v{}]", i);
                filter_parts.push(format!(
                    "{}{}xfade=transition={}:duration={}:offset={}{}",
                    current_video,
                    video_labels[i],
                    xfade_transition,
                    trans_duration_sec,
                    offset,
                    out_label
                ));

                current_video = out_label;
                cumulative_duration = offset + clip_durations[i];
            }

            // Audio crossfade
            let mut current_audio = audio_labels[0].clone();
            cumulative_duration = clip_durations[0];

            for i in 1..clips.len() {
                let trans_duration_sec = clips[i].transition_duration_ms as f64 / 1000.0;
                let trans_duration_sec = trans_duration_sec.max(0.1).min(2.0);
                let offset = (cumulative_duration - trans_duration_sec).max(0.0);

                let out_label = format!("[a{}]", i);
                filter_parts.push(format!(
                    "{}{}acrossfade=d={}:c1=tri:c2=tri{}",
                    current_audio,
                    audio_labels[i],
                    trans_duration_sec,
                    out_label
                ));

                current_audio = out_label;
                cumulative_duration = offset + clip_durations[i];
            }

            let filter_complex = filter_parts.join(";");
            let final_video = format!("[v{}]", clips.len() - 1);
            let final_audio = format!("[a{}]", clips.len() - 1);

            // Build ffmpeg command
            let mut cmd = Command::new(&self.ffmpeg_path);

            for temp_file in &temp_files {
                cmd.arg("-i").arg(temp_file);
            }

            let status = cmd
                .args([
                    "-filter_complex", &filter_complex,
                    "-map", &final_video,
                    "-map", &final_audio,
                    "-c:v", "libx264",
                    "-crf", "18",
                    "-preset", "fast",
                    "-pix_fmt", "yuv420p",
                    "-c:a", "aac",
                    "-b:a", "192k",
                    "-movflags", "+faststart",
                    "-y",
                ])
                .arg(output_path)
                .status()
                .context("Failed to execute ffmpeg with xfade")?;

            // Clean up
            let _ = std::fs::remove_dir_all(&temp_clips_dir);

            if !status.success() {
                return Err(anyhow!("FFmpeg xfade concatenation failed"));
            }
        }

        Ok(())
    }
}

impl Default for FFmpeg {
    fn default() -> Self {
        Self::new().expect("FFmpeg not found in PATH")
    }
}

fn which_command(name: &str) -> Result<String> {
    // Common installation paths for ffmpeg on macOS
    let common_paths = [
        format!("/opt/homebrew/bin/{}", name),      // Apple Silicon Homebrew
        format!("/usr/local/bin/{}", name),          // Intel Homebrew / manual install
        format!("/usr/bin/{}", name),                // System install
        format!("/opt/local/bin/{}", name),          // MacPorts
    ];

    // First check common paths directly (works when launched from Finder)
    for path in &common_paths {
        if std::path::Path::new(path).exists() {
            return Ok(path.clone());
        }
    }

    // Fallback to which command (works when launched from terminal)
    let output = Command::new("which")
        .arg(name)
        .output()
        .context(format!("Failed to find {name}"))?;

    if output.status.success() {
        let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if !path.is_empty() {
            return Ok(path);
        }
    }

    Err(anyhow!(
        "{} not found. Please install FFmpeg via: brew install ffmpeg",
        name
    ))
}

fn format_time(seconds: f64) -> String {
    let hours = (seconds / 3600.0) as u32;
    let minutes = ((seconds % 3600.0) / 60.0) as u32;
    let secs = seconds % 60.0;
    format!("{:02}:{:02}:{:06.3}", hours, minutes, secs)
}
```

## src-tauri/src/services/mod.rs

```rust
pub mod srt_parser;
pub mod database;
pub mod ffmpeg;
pub mod analyzer;
pub mod scoring;
pub mod python_sidecar;
pub mod director;

pub use srt_parser::SrtParser;
pub use database::Database;
pub use ffmpeg::{FFmpeg, ConcatClip};
pub use analyzer::{TelemetryAnalyzer, SegmentSignals};
pub use scoring::{ScoreCalculator, Profile};
pub use python_sidecar::{PythonSidecar, VisualAnalysis, EditSequence, EditDecision, ClipInfo};
pub use director::{Director, SegmentContext};
```

## src-tauri/src/services/python_sidecar.rs

```rust
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::process::{Command, Stdio};
use std::io::Write;

/// Visual analysis results from Python sidecar
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualAnalysis {
    pub motion: Option<MotionAnalysis>,
    pub scene: Option<SceneAnalysis>,
    pub color: Option<ColorAnalysis>,
    pub objects: Option<ObjectAnalysis>,
    pub semantic: Option<SemanticAnalysis>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotionAnalysis {
    pub avg_magnitude: f64,
    pub peak_magnitude: f64,
    pub peak_frame: i64,
    pub dominant_direction: f64,
    pub motion_consistency: f64,
    pub action_peaks: Vec<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneChange {
    pub frame: i64,
    pub timestamp_ms: i64,
    #[serde(rename = "type")]
    pub transition_type: String,
    pub confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneAnalysis {
    pub scene_changes: Vec<SceneChange>,
    pub avg_scene_duration_ms: f64,
    pub is_single_shot: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColorAnalysis {
    pub dominant_colors: Vec<(u8, u8, u8)>,
    pub color_weights: Vec<f64>,
    pub avg_brightness: f64,
    pub avg_saturation: f64,
    pub is_low_light: bool,
    pub is_golden_hour: bool,
    pub color_consistency: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectAnalysis {
    pub primary_subject: Option<String>,
    pub subject_entry_direction: Option<String>,
    pub subject_exit_direction: Option<String>,
    pub avg_subjects_per_frame: f64,
    pub has_consistent_subject: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticAnalysis {
    pub scene_type: String,
    pub top_descriptions: Vec<(String, f64)>,
    pub embedding_size: usize,
}

/// Edit decision from the suggestion engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EditDecision {
    pub clip_id: String,
    pub sequence_order: i32,
    pub adjusted_start_ms: i64,
    pub adjusted_end_ms: i64,
    pub transition_type: String,
    pub transition_duration_ms: i64,
    pub confidence: f64,
    pub reasoning: String,
}

/// Complete edit sequence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EditSequence {
    pub decisions: Vec<EditDecision>,
    pub total_duration_ms: i64,
    pub style: String,
    pub was_reordered: bool,
}

/// Clip info for edit sequence generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClipInfo {
    pub clip_id: String,
    pub video_path: String,
    pub start_ms: i64,
    pub end_ms: i64,
}

/// Python sidecar for visual analysis
pub struct PythonSidecar {
    python_path: String,
    script_dir: String,
}

impl PythonSidecar {
    pub fn new() -> Result<Self, String> {
        // Find Python executable
        let python_path = Self::find_python()?;

        // Find the script directory (relative to the executable or in dev mode)
        let script_dir = Self::find_script_dir()?;

        Ok(Self {
            python_path,
            script_dir,
        })
    }

    fn find_python() -> Result<String, String> {
        // Try common Python paths
        let candidates = [
            "python3",
            "python",
            "/usr/bin/python3",
            "/usr/local/bin/python3",
            "/opt/homebrew/bin/python3",
        ];

        for candidate in candidates {
            let output = Command::new(candidate)
                .arg("--version")
                .output();

            if let Ok(output) = output {
                if output.status.success() {
                    return Ok(candidate.to_string());
                }
            }
        }

        Err("Python 3 not found. Please install Python 3.11+".to_string())
    }

    fn find_script_dir() -> Result<String, String> {
        // In development, look for python/ directory relative to project root
        let dev_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .map(|p| p.join("python"));

        if let Some(path) = dev_path {
            if path.exists() {
                return Ok(path.to_string_lossy().to_string());
            }
        }

        // In production, look in Resources
        if let Ok(exe_path) = std::env::current_exe() {
            let resources = exe_path
                .parent()
                .and_then(|p| p.parent())
                .map(|p| p.join("Resources").join("python"));

            if let Some(path) = resources {
                if path.exists() {
                    return Ok(path.to_string_lossy().to_string());
                }
            }
        }

        Err("Python scripts not found".to_string())
    }

    fn run_command(&self, command: &str, args: serde_json::Value) -> Result<serde_json::Value, String> {
        let request = serde_json::json!({
            "command": command,
            "args": args
        });

        let mut child = Command::new(&self.python_path)
            .arg("-m")
            .arg("skyclip_analyzer.cli")
            .current_dir(&self.script_dir)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("Failed to spawn Python: {}", e))?;

        // Write request to stdin
        if let Some(ref mut stdin) = child.stdin {
            stdin
                .write_all(request.to_string().as_bytes())
                .map_err(|e| format!("Failed to write to Python stdin: {}", e))?;
        }

        let output = child
            .wait_with_output()
            .map_err(|e| format!("Failed to wait for Python: {}", e))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(format!("Python error: {}", stderr));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let response: serde_json::Value = serde_json::from_str(&stdout)
            .map_err(|e| format!("Failed to parse Python response: {} - {}", e, stdout))?;

        if response.get("success").and_then(|v| v.as_bool()) == Some(true) {
            Ok(response["result"].clone())
        } else {
            Err(response["error"]
                .as_str()
                .unwrap_or("Unknown error")
                .to_string())
        }
    }

    /// Analyze a video clip for motion, scene, and color
    pub fn analyze_clip(
        &self,
        video_path: &str,
        start_ms: i64,
        end_ms: Option<i64>,
        include_objects: bool,
        include_semantic: bool,
    ) -> Result<VisualAnalysis, String> {
        let args = serde_json::json!({
            "video_path": video_path,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "include_objects": include_objects,
            "include_semantic": include_semantic
        });

        let result = self.run_command("analyze_clip", args)?;

        // Parse individual components
        let motion: Option<MotionAnalysis> = result
            .get("motion")
            .and_then(|v| serde_json::from_value(v.clone()).ok());

        let scene: Option<SceneAnalysis> = result
            .get("scene")
            .and_then(|v| serde_json::from_value(v.clone()).ok());

        let color: Option<ColorAnalysis> = result
            .get("color")
            .and_then(|v| serde_json::from_value(v.clone()).ok());

        let objects: Option<ObjectAnalysis> = result
            .get("objects")
            .and_then(|v| serde_json::from_value(v.clone()).ok());

        let semantic: Option<SemanticAnalysis> = result
            .get("semantic")
            .and_then(|v| serde_json::from_value(v.clone()).ok());

        Ok(VisualAnalysis {
            motion,
            scene,
            color,
            objects,
            semantic,
        })
    }

    /// Generate edit sequence for multiple clips
    pub fn generate_edit_sequence(
        &self,
        clips: Vec<ClipInfo>,
        style: &str,
        reorder: bool,
        full_analysis: bool,
    ) -> Result<EditSequence, String> {
        let args = serde_json::json!({
            "clips": clips,
            "style": style,
            "reorder": reorder,
            "full_analysis": full_analysis
        });

        let result = self.run_command("generate_edit_sequence", args)?;
        serde_json::from_value(result).map_err(|e| format!("Failed to parse edit sequence: {}", e))
    }

    /// Suggest transition between two clips
    pub fn suggest_transition(
        &self,
        clip_a: ClipInfo,
        clip_b: ClipInfo,
        style: &str,
    ) -> Result<(String, i64, f64, String), String> {
        let args = serde_json::json!({
            "clip_a": clip_a,
            "clip_b": clip_b,
            "style": style
        });

        let result = self.run_command("suggest_transition", args)?;

        Ok((
            result["transition_type"]
                .as_str()
                .unwrap_or("cut")
                .to_string(),
            result["transition_duration_ms"].as_i64().unwrap_or(0),
            result["confidence"].as_f64().unwrap_or(0.5),
            result["reasoning"]
                .as_str()
                .unwrap_or("")
                .to_string(),
        ))
    }

    /// Check if Python sidecar is available
    pub fn is_available(&self) -> bool {
        let result = Command::new(&self.python_path)
            .arg("-c")
            .arg("import cv2; print('ok')")
            .output();

        matches!(result, Ok(output) if output.status.success())
    }

    /// Install Python dependencies
    pub fn install_dependencies(&self) -> Result<(), String> {
        let requirements_path = Path::new(&self.script_dir).join("requirements.txt");

        let output = Command::new(&self.python_path)
            .arg("-m")
            .arg("pip")
            .arg("install")
            .arg("-r")
            .arg(&requirements_path)
            .output()
            .map_err(|e| format!("Failed to run pip: {}", e))?;

        if output.status.success() {
            Ok(())
        } else {
            Err(String::from_utf8_lossy(&output.stderr).to_string())
        }
    }
}

impl Default for PythonSidecar {
    fn default() -> Self {
        Self::new().expect("Failed to initialize Python sidecar")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_python() {
        let result = PythonSidecar::find_python();
        // This test will pass if Python is installed
        if result.is_ok() {
            println!("Found Python at: {}", result.unwrap());
        }
    }
}
```

## src-tauri/src/services/scoring.rs

```rust
use crate::services::SegmentSignals;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileWeights {
    #[serde(default)]
    pub gimbal_smoothness: f64,
    #[serde(default)]
    pub gimbal_pitch_delta: f64,
    #[serde(default)]
    pub gimbal_yaw_delta: f64,
    #[serde(default)]
    pub gps_speed: f64,
    #[serde(default)]
    pub altitude_delta: f64,
    #[serde(default)]
    pub iso_penalty: f64,
    #[serde(default)]
    pub motion_magnitude: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileThresholds {
    #[serde(default)]
    pub min_gimbal_smoothness: Option<f64>,
    #[serde(default)]
    pub max_iso: Option<f64>,
    #[serde(default)]
    pub min_gps_speed: Option<f64>,
    #[serde(default)]
    pub min_motion_magnitude: Option<f64>,
    #[serde(default)]
    pub min_duration_sec: Option<f64>,
    #[serde(default)]
    pub max_duration_sec: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Profile {
    pub id: String,
    pub name: String,
    pub description: String,
    pub weights: ProfileWeights,
    #[serde(default)]
    pub thresholds: ProfileThresholds,
    #[serde(default)]
    pub preferences: HashMap<String, bool>,
}

impl Default for ProfileThresholds {
    fn default() -> Self {
        Self {
            min_gimbal_smoothness: None,
            max_iso: None,
            min_gps_speed: None,
            min_motion_magnitude: None,
            min_duration_sec: Some(3.0),
            max_duration_sec: Some(30.0),
        }
    }
}

pub struct ScoreCalculator {
    profiles: HashMap<String, Profile>,
}

impl ScoreCalculator {
    pub fn new() -> Self {
        Self {
            profiles: HashMap::new(),
        }
    }

    /// Load profiles from a directory of JSON files
    pub fn load_profiles_from_dir<P: AsRef<Path>>(&mut self, dir: P) -> Result<(), String> {
        let dir = dir.as_ref();
        if !dir.exists() {
            return Err(format!("Profiles directory does not exist: {:?}", dir));
        }

        for entry in std::fs::read_dir(dir).map_err(|e| e.to_string())? {
            let entry = entry.map_err(|e| e.to_string())?;
            let path = entry.path();

            if path.extension().map(|e| e == "json").unwrap_or(false) {
                let content = std::fs::read_to_string(&path).map_err(|e| e.to_string())?;
                let profile: Profile = serde_json::from_str(&content)
                    .map_err(|e| format!("Failed to parse {:?}: {}", path, e))?;
                self.profiles.insert(profile.id.clone(), profile);
            }
        }

        Ok(())
    }

    /// Load a single profile from JSON string
    pub fn load_profile(&mut self, json: &str) -> Result<(), String> {
        let profile: Profile =
            serde_json::from_str(json).map_err(|e| format!("Failed to parse profile: {}", e))?;
        self.profiles.insert(profile.id.clone(), profile);
        Ok(())
    }

    /// Get all loaded profiles
    pub fn get_profiles(&self) -> Vec<&Profile> {
        self.profiles.values().collect()
    }

    /// Get a specific profile by ID
    pub fn get_profile(&self, id: &str) -> Option<&Profile> {
        self.profiles.get(id)
    }

    /// Calculate score for a segment using a specific profile
    pub fn calculate_score(&self, profile_id: &str, signals: &SegmentSignals) -> Option<f64> {
        let profile = self.profiles.get(profile_id)?;
        Some(self.score_with_profile(profile, signals))
    }

    /// Calculate scores for a segment across all profiles
    pub fn calculate_all_scores(&self, signals: &SegmentSignals) -> HashMap<String, f64> {
        self.profiles
            .iter()
            .map(|(id, profile)| (id.clone(), self.score_with_profile(profile, signals)))
            .collect()
    }

    fn score_with_profile(&self, profile: &Profile, signals: &SegmentSignals) -> f64 {
        let weights = &profile.weights;

        // Normalize signals to 0-1 range for consistent scoring
        // These normalizations are based on typical drone telemetry ranges

        // Gimbal smoothness is already 0-1
        let smoothness_score = signals.gimbal_smoothness * weights.gimbal_smoothness;

        // Gimbal pitch delta: 0-30 deg/sec is typical range for intentional movement
        let pitch_normalized = (signals.gimbal_pitch_delta_avg.abs() / 30.0).min(1.0);
        let pitch_score = pitch_normalized * weights.gimbal_pitch_delta;

        // Gimbal yaw delta: 0-45 deg/sec for pans
        let yaw_normalized = (signals.gimbal_yaw_delta_avg.abs() / 45.0).min(1.0);
        let yaw_score = yaw_normalized * weights.gimbal_yaw_delta;

        // GPS speed: 0-20 m/s (0-72 km/h) covers most drone movement
        let speed_normalized = (signals.gps_speed_avg / 20.0).min(1.0);
        let speed_score = speed_normalized * weights.gps_speed;

        // Altitude delta: 0-50m change is significant
        let alt_normalized = (signals.altitude_delta.abs() / 50.0).min(1.0);
        let alt_score = alt_normalized * weights.altitude_delta;

        // ISO penalty: lower is better, 100-3200 range
        // Score decreases as ISO increases
        let iso_normalized = if signals.iso_avg > 0.0 {
            1.0 - ((signals.iso_avg - 100.0) / 3100.0).clamp(0.0, 1.0)
        } else {
            1.0 // No ISO data = assume good
        };
        let iso_score = iso_normalized * weights.iso_penalty;

        // Motion magnitude: 0-20 combined score
        let motion_normalized = (signals.motion_magnitude / 20.0).min(1.0);
        let motion_score = motion_normalized * weights.motion_magnitude;

        // Sum all weighted scores (should sum to ~1.0 if weights are normalized)
        let total = smoothness_score
            + pitch_score
            + yaw_score
            + speed_score
            + alt_score
            + iso_score
            + motion_score;

        // Clamp to 0-100 range for display
        (total * 100.0).clamp(0.0, 100.0)
    }

    /// Check if a segment passes the threshold requirements for a profile
    pub fn passes_thresholds(
        &self,
        profile_id: &str,
        signals: &SegmentSignals,
        duration_sec: f64,
    ) -> bool {
        let profile = match self.profiles.get(profile_id) {
            Some(p) => p,
            None => return false,
        };

        let thresholds = &profile.thresholds;

        // Check duration bounds
        if let Some(min_dur) = thresholds.min_duration_sec {
            if duration_sec < min_dur {
                return false;
            }
        }
        if let Some(max_dur) = thresholds.max_duration_sec {
            if duration_sec > max_dur {
                return false;
            }
        }

        // Check gimbal smoothness
        if let Some(min_smooth) = thresholds.min_gimbal_smoothness {
            if signals.gimbal_smoothness < min_smooth {
                return false;
            }
        }

        // Check ISO
        if let Some(max_iso) = thresholds.max_iso {
            if signals.iso_avg > max_iso {
                return false;
            }
        }

        // Check GPS speed
        if let Some(min_speed) = thresholds.min_gps_speed {
            if signals.gps_speed_avg < min_speed {
                return false;
            }
        }

        // Check motion magnitude
        if let Some(min_motion) = thresholds.min_motion_magnitude {
            if signals.motion_magnitude < min_motion {
                return false;
            }
        }

        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_profile() -> Profile {
        Profile {
            id: "test".to_string(),
            name: "Test Profile".to_string(),
            description: "For testing".to_string(),
            weights: ProfileWeights {
                gimbal_smoothness: 0.3,
                gimbal_pitch_delta: 0.1,
                gimbal_yaw_delta: 0.1,
                gps_speed: 0.2,
                altitude_delta: 0.1,
                iso_penalty: 0.1,
                motion_magnitude: 0.1,
            },
            thresholds: ProfileThresholds::default(),
            preferences: HashMap::new(),
        }
    }

    #[test]
    fn test_score_calculation() {
        let mut calc = ScoreCalculator::new();
        calc.profiles.insert("test".to_string(), test_profile());

        let signals = SegmentSignals {
            gimbal_smoothness: 0.8,
            gimbal_pitch_delta_avg: 10.0,
            gimbal_yaw_delta_avg: 15.0,
            gps_speed_avg: 5.0,
            altitude_delta: 20.0,
            iso_avg: 200.0,
            motion_magnitude: 8.0,
        };

        let score = calc.calculate_score("test", &signals).unwrap();
        assert!(score > 0.0 && score <= 100.0);
    }

    #[test]
    fn test_thresholds() {
        let mut calc = ScoreCalculator::new();
        let mut profile = test_profile();
        profile.thresholds.min_gimbal_smoothness = Some(0.7);
        profile.thresholds.max_iso = Some(800.0);
        calc.profiles.insert("test".to_string(), profile);

        let good_signals = SegmentSignals {
            gimbal_smoothness: 0.8,
            iso_avg: 400.0,
            ..Default::default()
        };

        let bad_signals = SegmentSignals {
            gimbal_smoothness: 0.5, // Below threshold
            iso_avg: 1600.0,        // Above threshold
            ..Default::default()
        };

        assert!(calc.passes_thresholds("test", &good_signals, 10.0));
        assert!(!calc.passes_thresholds("test", &bad_signals, 10.0));
    }
}
```

## src-tauri/src/services/srt_parser.rs

```rust
use crate::models::TelemetryFrame;
use anyhow::{Context, Result};
use chrono::{NaiveDateTime, TimeZone, Utc};
use regex::Regex;
use std::fs;
use std::path::Path;

pub struct SrtParser {
    // Regex patterns for parsing SRT content
    time_pattern: Regex,
    iso_pattern: Regex,
    shutter_pattern: Regex,
    fnum_pattern: Regex,
    ev_pattern: Regex,
    ct_pattern: Regex,
    color_md_pattern: Regex,
    focal_len_pattern: Regex,
    latitude_pattern: Regex,
    longitude_pattern: Regex,
    altitude_pattern: Regex,
    gb_yaw_pattern: Regex,
    gb_pitch_pattern: Regex,
    gb_roll_pattern: Regex,
    timestamp_pattern: Regex,
    srt_cnt_pattern: Regex,
}

impl SrtParser {
    pub fn new() -> Self {
        Self {
            time_pattern: Regex::new(r"(\d{2}):(\d{2}):(\d{2}),(\d{3}) --> (\d{2}):(\d{2}):(\d{2}),(\d{3})").unwrap(),
            iso_pattern: Regex::new(r"\[iso\s*:\s*(\d+)\]").unwrap(),
            shutter_pattern: Regex::new(r"\[shutter\s*:\s*([^\]]+)\]").unwrap(),
            fnum_pattern: Regex::new(r"\[fnum\s*:\s*(\d+)\]").unwrap(),
            ev_pattern: Regex::new(r"\[ev\s*:\s*([^\]]+)\]").unwrap(),
            ct_pattern: Regex::new(r"\[ct\s*:\s*(\d+)\]").unwrap(),
            color_md_pattern: Regex::new(r"\[color_md\s*:\s*([^\]]+)\]").unwrap(),
            focal_len_pattern: Regex::new(r"\[focal_len\s*:\s*([^\]]+)\]").unwrap(),
            latitude_pattern: Regex::new(r"\[latitude\s*:\s*([^\]]+)\]").unwrap(),
            longitude_pattern: Regex::new(r"\[longitude\s*:\s*([^\]]+)\]").unwrap(),
            // Match both "altitude: X" and "rel_alt: X abs_alt: Y" formats
            altitude_pattern: Regex::new(r"\[(?:altitude|rel_alt)\s*:\s*([^\]\s]+)").unwrap(),
            gb_yaw_pattern: Regex::new(r"\[gb_yaw\s*:\s*([^\]]+)\]").unwrap(),
            gb_pitch_pattern: Regex::new(r"\[gb_pitch\s*:\s*([^\]]+)\]").unwrap(),
            gb_roll_pattern: Regex::new(r"\[gb_roll\s*:\s*([^\]]+)\]").unwrap(),
            timestamp_pattern: Regex::new(r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d{3})").unwrap(),
            srt_cnt_pattern: Regex::new(r"SrtCnt\s*:\s*(\d+)").unwrap(),
        }
    }

    /// Parse an SRT file and return a vector of telemetry frames
    pub fn parse_file<P: AsRef<Path>>(&self, path: P) -> Result<Vec<TelemetryFrame>> {
        let content = fs::read_to_string(path.as_ref())
            .with_context(|| format!("Failed to read SRT file: {:?}", path.as_ref()))?;

        self.parse_content(&content)
    }

    /// Parse SRT content string and return telemetry frames
    pub fn parse_content(&self, content: &str) -> Result<Vec<TelemetryFrame>> {
        let mut frames = Vec::new();

        // Split by blank lines to get individual subtitle blocks
        let blocks: Vec<&str> = content.split("\n\n").collect();

        for block in blocks {
            if block.trim().is_empty() {
                continue;
            }

            if let Some(frame) = self.parse_block(block) {
                frames.push(frame);
            }
        }

        Ok(frames)
    }

    fn parse_block(&self, block: &str) -> Option<TelemetryFrame> {
        let lines: Vec<&str> = block.lines().collect();
        if lines.len() < 3 {
            return None;
        }

        // First line is the index
        let index: u32 = lines[0].trim().parse().ok()?;

        // Second line is the timecode
        let (start_ms, end_ms) = self.parse_timecode(lines[1])?;

        let mut frame = TelemetryFrame::new(index, start_ms, end_ms);

        // Rest is the content with telemetry data
        let content = lines[2..].join("\n");

        // Extract SrtCnt if present (validation)
        if let Some(caps) = self.srt_cnt_pattern.captures(&content) {
            let _srt_cnt: u32 = caps[1].parse().unwrap_or(0);
        }

        // Extract timestamp
        if let Some(caps) = self.timestamp_pattern.captures(&content) {
            if let Ok(dt) = NaiveDateTime::parse_from_str(&caps[1], "%Y-%m-%d %H:%M:%S%.3f") {
                frame.timestamp = Some(Utc.from_utc_datetime(&dt));
            }
        }

        // Extract camera settings
        if let Some(caps) = self.iso_pattern.captures(&content) {
            frame.iso = caps[1].parse().ok();
        }
        if let Some(caps) = self.shutter_pattern.captures(&content) {
            frame.shutter = Some(caps[1].trim().to_string());
        }
        if let Some(caps) = self.fnum_pattern.captures(&content) {
            frame.fnum = caps[1].parse().ok();
        }
        if let Some(caps) = self.ev_pattern.captures(&content) {
            frame.ev = caps[1].trim().parse().ok();
        }
        if let Some(caps) = self.ct_pattern.captures(&content) {
            frame.color_temp = caps[1].parse().ok();
        }
        if let Some(caps) = self.color_md_pattern.captures(&content) {
            frame.color_mode = Some(caps[1].trim().to_string());
        }
        if let Some(caps) = self.focal_len_pattern.captures(&content) {
            frame.focal_len = caps[1].trim().parse().ok();
        }

        // Extract GPS data
        if let Some(caps) = self.latitude_pattern.captures(&content) {
            frame.latitude = caps[1].trim().parse().ok();
        }
        if let Some(caps) = self.longitude_pattern.captures(&content) {
            frame.longitude = caps[1].trim().parse().ok();
        }
        if let Some(caps) = self.altitude_pattern.captures(&content) {
            frame.altitude = caps[1].trim().parse().ok();
        }

        // Extract gimbal orientation
        if let Some(caps) = self.gb_yaw_pattern.captures(&content) {
            frame.gimbal_yaw = caps[1].trim().parse().ok();
        }
        if let Some(caps) = self.gb_pitch_pattern.captures(&content) {
            frame.gimbal_pitch = caps[1].trim().parse().ok();
        }
        if let Some(caps) = self.gb_roll_pattern.captures(&content) {
            frame.gimbal_roll = caps[1].trim().parse().ok();
        }

        Some(frame)
    }

    fn parse_timecode(&self, line: &str) -> Option<(i64, i64)> {
        let caps = self.time_pattern.captures(line)?;

        let start_h: i64 = caps[1].parse().ok()?;
        let start_m: i64 = caps[2].parse().ok()?;
        let start_s: i64 = caps[3].parse().ok()?;
        let start_ms: i64 = caps[4].parse().ok()?;

        let end_h: i64 = caps[5].parse().ok()?;
        let end_m: i64 = caps[6].parse().ok()?;
        let end_s: i64 = caps[7].parse().ok()?;
        let end_ms: i64 = caps[8].parse().ok()?;

        let start = start_h * 3600000 + start_m * 60000 + start_s * 1000 + start_ms;
        let end = end_h * 3600000 + end_m * 60000 + end_s * 1000 + end_ms;

        Some((start, end))
    }
}

impl Default for SrtParser {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_srt_block() {
        let content = r#"1
00:00:00,000 --> 00:00:01,000
<font size="28">SrtCnt : 1, DiffTime : 1000ms
2025-12-23 14:32:15.123
[iso : 100] [shutter : 1/500.0] [fnum : 280] [ev : 0]
[ct : 5500] [color_md : default] [focal_len : 24.00]
[latitude : 40.7128] [longitude : -74.0060] [altitude: 150.0]
[gb_yaw : 45.2] [gb_pitch : -15.3] [gb_roll : 0.1]</font>"#;

        let parser = SrtParser::new();
        let frames = parser.parse_content(content).unwrap();

        assert_eq!(frames.len(), 1);
        let frame = &frames[0];

        assert_eq!(frame.index, 1);
        assert_eq!(frame.start_time_ms, 0);
        assert_eq!(frame.end_time_ms, 1000);
        assert_eq!(frame.iso, Some(100));
        assert_eq!(frame.shutter, Some("1/500.0".to_string()));
        assert_eq!(frame.fnum, Some(280));
        assert_eq!(frame.latitude, Some(40.7128));
        assert_eq!(frame.longitude, Some(-74.0060));
        assert_eq!(frame.altitude, Some(150.0));
        assert_eq!(frame.gimbal_yaw, Some(45.2));
        assert_eq!(frame.gimbal_pitch, Some(-15.3));
        assert_eq!(frame.gimbal_roll, Some(0.1));
    }
}
```

## vite.config.ts

```typescript
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// @ts-expect-error process is a nodejs global
const host = process.env.TAURI_DEV_HOST;

// https://vite.dev/config/
export default defineConfig(async () => ({
  plugins: [react()],

  // Vite options tailored for Tauri development and only applied in `tauri dev` or `tauri build`
  //
  // 1. prevent Vite from obscuring rust errors
  clearScreen: false,
  // 2. tauri expects a fixed port, fail if that port is not available
  server: {
    port: 1420,
    strictPort: true,
    host: host || false,
    hmr: host
      ? {
          protocol: "ws",
          host,
          port: 1421,
        }
      : undefined,
    watch: {
      // 3. tell Vite to ignore watching `src-tauri`
      ignored: ["**/src-tauri/**"],
    },
  },
}));
```
