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
