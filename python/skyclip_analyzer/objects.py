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
