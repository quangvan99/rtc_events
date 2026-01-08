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
    """Manage tracked faces across frames"""

    def __init__(self, config: dict, max_age: int = 30):
        self.config = config
        self.max_age = max_age
        self._trackers: dict[int, TrackedFace] = {}

    def get(self, oid: int) -> TrackedFace | None:
        return self._trackers.get(oid)

    def get_or_create(self, oid: int, frame: int) -> TrackedFace:
        if oid not in self._trackers:
            self._trackers[oid] = TrackedFace(
                object_id=oid,
                l2_threshold=self.config.get("l2_threshold", 1.0),
                min_streak=self.config.get("min_streak", 3),
                skip_reid=self.config.get("skip_reid", 3),
                reid_interval=self.config.get("reid_interval", 30),
                last_sgie=frame,
            )
        return self._trackers[oid]

    def cleanup(self) -> list[int]:
        """Increment age and remove stale trackers. Returns removed IDs."""
        removed = []
        for oid, t in self._trackers.items():
            t.age += 1
            if t.age > self.max_age:
                removed.append(oid)
        for oid in removed:
            del self._trackers[oid]
        return removed

    def stats(self) -> tuple[int, int, int]:
        """Returns (total, confirmed, pending)"""
        confirmed = sum(1 for t in self._trackers.values() if t.label)
        return len(self._trackers), confirmed, len(self._trackers) - confirmed
