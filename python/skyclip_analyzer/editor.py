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
