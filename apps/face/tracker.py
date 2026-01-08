"""Face tracker - streak-based identity confirmation"""

from dataclasses import dataclass, field


@dataclass
class TrackedFace:
    """
    Track a face and confirm identity via consecutive matches.

    - Pending (no label): run SGIE every skip_reid frames
    - Confirmed (has label): run SGIE every reid_interval frames
    """
    object_id: int

    # Config values
    l2_threshold: float = 1.0
    min_streak: int = 3
    skip_reid: int = 3
    reid_interval: int = 30

    # Identity
    label: str | None = None
    score: float = 0.0

    # Streak: consecutive matches of same person
    _person: int = -1
    _streak: int = 0
    _distances: list[float] = field(default_factory=list)

    # Timing
    last_sgie: int = 0
    age: int = 0

    def should_run_sgie(self, frame: int) -> bool:
        """Check if SGIE should run based on frame interval"""
        elapsed = frame - self.last_sgie
        interval = self.reid_interval if self.label else self.skip_reid
        return elapsed >= interval

    def add_match(self, person: int, distance: float) -> bool:
        """
        Add match result. Returns True if identity confirmed.

        - Bad distance (> threshold): ignore
        - Different person: restart streak
        - Same person: continue streak → confirm when streak >= min_streak
        """
        if distance > self.l2_threshold:
            return False

        if person != self._person:
            self._person = person
            self._distances = [distance]
            self._streak = 1
            return False

        self._streak += 1
        self._distances.append(distance)
        return self._streak >= self.min_streak

    def confirm(self, name: str) -> bool:
        """Confirm identity. Returns True if this is first confirmation."""
        is_new = self.label is None
        self.label = name
        self.score = sum(self._distances) / len(self._distances) if self._distances else 0.0

        # Reset streak
        self._person = -1
        self._streak = 0
        self._distances = []

        if is_new:
            print(f"[CONFIRMED] id={self.object_id} -> {name} (score={self.score:.3f})")
        return is_new


class TrackerManager:
    """Manage tracked faces across frames, namespaced by source_id for multi-camera

    Tracks faces per camera to avoid object_id collision across cameras.
    """

    def __init__(self, config: dict, max_age: int = 30):
        self.config = config
        self.max_age = max_age
        # Nested: {source_id: {object_id: TrackedFace}}
        self._trackers: dict[int, dict[int, TrackedFace]] = {}

    def _get_camera_dict(self, source_id: int) -> dict[int, TrackedFace]:
        """Get or create tracker dict for camera"""
        if source_id not in self._trackers:
            self._trackers[source_id] = {}
        return self._trackers[source_id]

    def get(self, source_id: int, oid: int) -> TrackedFace | None:
        """Get tracker by source_id and object_id"""
        cam_dict = self._trackers.get(source_id, {})
        return cam_dict.get(oid)

    def get_or_create(self, source_id: int, oid: int, frame: int) -> TrackedFace:
        """Get or create tracker for (source_id, object_id)"""
        cam_dict = self._get_camera_dict(source_id)

        if oid not in cam_dict:
            cam_dict[oid] = TrackedFace(
                object_id=oid,
                l2_threshold=self.config.get("l2_threshold", 1.0),
                min_streak=self.config.get("min_streak", 3),
                skip_reid=self.config.get("skip_reid", 3),
                reid_interval=self.config.get("reid_interval", 30),
                last_sgie=frame,
            )
        return cam_dict[oid]

    def cleanup(self) -> list[tuple[int, int]]:
        """Increment age and remove stale trackers.

        Returns list of (source_id, object_id) removed.
        """
        removed = []

        for source_id, cam_dict in self._trackers.items():
            to_remove = []
            for oid, t in cam_dict.items():
                t.age += 1
                if t.age > self.max_age:
                    to_remove.append(oid)

            for oid in to_remove:
                del cam_dict[oid]
                removed.append((source_id, oid))

        return removed

    def remove_camera(self, source_id: int) -> list[int]:
        """Remove all trackers for a camera.

        Returns list of object_ids that were removed.
        """
        cam_dict = self._trackers.pop(source_id, {})
        return list(cam_dict.keys())

    def stats(self) -> tuple[int, int, int]:
        """Returns (total, confirmed, pending) across all cameras"""
        total = 0
        confirmed = 0

        for cam_dict in self._trackers.values():
            for t in cam_dict.values():
                total += 1
                if t.label:
                    confirmed += 1

        return total, confirmed, total - confirmed

    def stats_by_camera(self) -> dict[int, tuple[int, int, int]]:
        """Returns {source_id: (total, confirmed, pending)}"""
        result = {}
        for source_id, cam_dict in self._trackers.items():
            total = len(cam_dict)
            confirmed = sum(1 for t in cam_dict.values() if t.label)
            result[source_id] = (total, confirmed, total - confirmed)
        return result

    # Backward compatibility for single-camera mode (source_id=0)

    def get_single(self, oid: int) -> TrackedFace | None:
        """Get tracker for single-camera mode (source_id=0)"""
        return self.get(0, oid)

    def get_or_create_single(self, oid: int, frame: int) -> TrackedFace:
        """Get or create tracker for single-camera mode (source_id=0)"""
        return self.get_or_create(0, oid, frame)
