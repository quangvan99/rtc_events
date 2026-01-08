"""Face database - manages registered face features"""

import json
import os
import time
import numpy as np


class FaceDatabase:
    """Manages registered face features for matching"""

    def __init__(self, features_path: str):
        self.names: list[str] = []
        self.features_matrix: np.ndarray | None = None
        self.avatars: dict[str, str] = {}
        self._load(features_path)

    def _load(self, path: str) -> None:
        """Load pre-registered face features from JSON"""
        if not os.path.exists(path):
            print(f"Warning: Features file not found: {path}")
            return

        start = time.time()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        features = []
        for name, info in data.items():
            if "feature" not in info:
                continue

            feat = info["feature"]
            if isinstance(feat[0], list):
                feat = feat[0]
            vec = np.array(feat, dtype=np.float32)

            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm

            self.names.append(name)
            features.append(vec)
            if "avatar" in info:
                self.avatars[name] = info["avatar"]

        if features:
            self.features_matrix = np.vstack(features).astype(np.float32)

        print(f"Loaded {len(self.names)} registered faces in {(time.time() - start) * 1000:.1f}ms")
        if self.features_matrix is not None:
            print(f"Feature matrix shape: {self.features_matrix.shape}")

    def match(self, embedding: np.ndarray) -> tuple[int, float]:
        """Match embedding against database. Returns (person_idx, distance)"""
        if self.features_matrix is None:
            return -1, float("inf")

        distances = np.linalg.norm(self.features_matrix - embedding, axis=1)
        idx = int(np.argmin(distances))
        return idx, float(distances[idx])
