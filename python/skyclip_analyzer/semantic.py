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
