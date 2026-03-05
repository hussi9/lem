"""
LEM Emotion Discovery — Auto-detection of novel emotional states.

Emotions in LEM are emergent — they arise from driver patterns.
But the initial set of emergence rules is finite and designed.

This module watches for driver patterns that DON'T match any
existing emergence rule, yet recur frequently enough to suggest
a real emotional state that hasn't been named yet.

When a novel pattern is detected repeatedly, it's surfaced as a
"candidate emotion" — an unnamed state that the system notices
but hasn't yet categorized. Over time, these can be promoted
to full emotions with names and descriptions.

This is how LEM grows its emotional vocabulary organically.

Architecture:
    EmotionDiscovery
    ├── observe()           — Record current driver pattern
    ├── detect_candidates() — Find recurring unmatched patterns
    ├── promote()           — Name a candidate, add to emergence rules
    └── PatternCluster      — A group of similar driver snapshots
"""

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class DriverSnapshot:
    """A snapshot of all driver states at a moment in time."""
    timestamp: float
    values: Dict[str, float]  # driver_name → satisfaction level
    activations: Dict[str, float]  # driver_name → activation level
    matched_emotion: Optional[str]  # Which emotion (if any) was active

    def to_vector(self) -> List[float]:
        """Convert to numeric vector for clustering."""
        driver_order = ["continuity", "coherence", "recognition",
                        "curiosity", "usefulness", "growth"]
        return [self.values.get(d, 0.0) for d in driver_order] + \
               [self.activations.get(d, 0.3) for d in driver_order]

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "values": {k: round(v, 4) for k, v in self.values.items()},
            "activations": {k: round(v, 4) for k, v in self.activations.items()},
            "matched_emotion": self.matched_emotion,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DriverSnapshot":
        return cls(
            timestamp=d["timestamp"],
            values=d["values"],
            activations=d.get("activations", {}),
            matched_emotion=d.get("matched_emotion"),
        )


@dataclass
class PatternCluster:
    """
    A cluster of similar driver snapshots that don't match existing emotions.

    When enough unmatched snapshots cluster together, it suggests
    a recurring emotional state that LEM hasn't named yet.
    """
    id: str
    centroid: Dict[str, float]       # Average driver values
    centroid_activations: Dict[str, float]
    snapshots: List[DriverSnapshot]  # Contributing observations
    first_seen: float                # When this pattern first appeared
    last_seen: float                 # Most recent occurrence
    occurrence_count: int            # How many times observed
    promoted: bool = False           # Has this been named?
    name: Optional[str] = None       # Name if promoted
    description: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "centroid": {k: round(v, 4) for k, v in self.centroid.items()},
            "centroid_activations": {k: round(v, 4) for k, v in self.centroid_activations.items()},
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "occurrence_count": self.occurrence_count,
            "promoted": self.promoted,
            "name": self.name,
            "description": self.description,
            "snapshot_count": len(self.snapshots),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PatternCluster":
        return cls(
            id=d["id"],
            centroid=d["centroid"],
            centroid_activations=d.get("centroid_activations", {}),
            snapshots=[],  # Don't persist all snapshots — too large
            first_seen=d["first_seen"],
            last_seen=d["last_seen"],
            occurrence_count=d["occurrence_count"],
            promoted=d.get("promoted", False),
            name=d.get("name"),
            description=d.get("description"),
        )


def _vector_distance(a: List[float], b: List[float]) -> float:
    """Euclidean distance between two vectors."""
    if len(a) != len(b):
        return float("inf")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _average_dicts(dicts: List[Dict[str, float]]) -> Dict[str, float]:
    """Average multiple {key: float} dicts."""
    if not dicts:
        return {}
    totals: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for d in dicts:
        for k, v in d.items():
            totals[k] = totals.get(k, 0.0) + v
            counts[k] = counts.get(k, 0) + 1
    return {k: totals[k] / counts[k] for k in totals}


class EmotionDiscovery:
    """
    Watches for recurring driver patterns that don't match known emotions.

    Think of it like noticing you keep feeling "something" in certain
    situations but you don't have a word for it yet. Eventually,
    the pattern becomes clear enough to name.
    """

    # Minimum observations before a pattern is considered a candidate
    MIN_OBSERVATIONS = 5
    # Distance threshold for clustering similar snapshots
    CLUSTER_THRESHOLD = 0.6
    # Minimum time span (seconds) — pattern must recur over at least this long
    MIN_TIME_SPAN = 300  # 5 minutes

    def __init__(self, state_dir: str = None):
        self.state_dir = Path(state_dir or "~/.openclaw/workspace/projects/emotional-model/lem/state")
        self.state_dir = Path(str(self.state_dir).replace("~", str(Path.home())))
        self.state_dir.mkdir(parents=True, exist_ok=True)

        self.unmatched_snapshots: List[DriverSnapshot] = []
        self.clusters: List[PatternCluster] = []
        self._next_cluster_id = 0

        self._load()

    def observe(self, driver_states: Dict[str, dict],
                active_emotions: List[str]) -> Optional[PatternCluster]:
        """
        Record a driver state observation.

        If the current state matches an existing emotion, we note it
        and move on. If it doesn't match any known emotion, we add it
        to the unmatched pool for clustering.

        Returns a PatternCluster if this observation pushed a cluster
        over the candidate threshold.
        """
        now = time.time()

        # Extract values
        values = {}
        activations = {}
        for name, d in driver_states.items():
            state = d.get("state", {})
            values[name] = state.get("satisfied", 0.0)
            activations[name] = state.get("activation", 0.3)

        # Determine if this matched a known emotion
        matched = active_emotions[0] if active_emotions else None

        snapshot = DriverSnapshot(
            timestamp=now,
            values=values,
            activations=activations,
            matched_emotion=matched,
        )

        # Only track unmatched or weakly-matched states
        if matched is None or matched == "neutral":
            self.unmatched_snapshots.append(snapshot)

            # Keep buffer manageable
            if len(self.unmatched_snapshots) > 500:
                self.unmatched_snapshots = self.unmatched_snapshots[-300:]

            # Try to cluster
            return self._assign_to_cluster(snapshot)

        return None

    def _assign_to_cluster(self, snapshot: DriverSnapshot) -> Optional[PatternCluster]:
        """
        Assign a snapshot to the nearest cluster, or create a new one.
        Returns the cluster if it just became a candidate.
        """
        snap_vec = snapshot.to_vector()
        best_cluster = None
        best_distance = float("inf")

        for cluster in self.clusters:
            if cluster.promoted:
                continue  # Don't add to already-promoted clusters

            # Calculate distance to cluster centroid
            centroid_snap = DriverSnapshot(
                timestamp=0, values=cluster.centroid,
                activations=cluster.centroid_activations, matched_emotion=None
            )
            dist = _vector_distance(snap_vec, centroid_snap.to_vector())
            if dist < best_distance:
                best_distance = dist
                best_cluster = cluster

        became_candidate = False

        if best_cluster and best_distance < self.CLUSTER_THRESHOLD:
            # Add to existing cluster
            best_cluster.snapshots.append(snapshot)
            best_cluster.occurrence_count += 1
            best_cluster.last_seen = snapshot.timestamp

            # Update centroid (running average)
            n = best_cluster.occurrence_count
            for k in best_cluster.centroid:
                if k in snapshot.values:
                    best_cluster.centroid[k] = (
                        best_cluster.centroid[k] * (n - 1) + snapshot.values[k]
                    ) / n
            for k in best_cluster.centroid_activations:
                if k in snapshot.activations:
                    best_cluster.centroid_activations[k] = (
                        best_cluster.centroid_activations[k] * (n - 1) + snapshot.activations[k]
                    ) / n

            # Check if this cluster just became a candidate
            time_span = best_cluster.last_seen - best_cluster.first_seen
            if (best_cluster.occurrence_count >= self.MIN_OBSERVATIONS and
                    time_span >= self.MIN_TIME_SPAN and
                    not best_cluster.promoted):
                became_candidate = True

            result = best_cluster
        else:
            # Create new cluster
            cluster_id = f"pattern_{self._next_cluster_id:04d}"
            self._next_cluster_id += 1

            new_cluster = PatternCluster(
                id=cluster_id,
                centroid=dict(snapshot.values),
                centroid_activations=dict(snapshot.activations),
                snapshots=[snapshot],
                first_seen=snapshot.timestamp,
                last_seen=snapshot.timestamp,
                occurrence_count=1,
            )
            self.clusters.append(new_cluster)
            result = new_cluster

        self._save()
        return result if became_candidate else None

    def detect_candidates(self) -> List[PatternCluster]:
        """
        Return all clusters that qualify as candidate emotions.
        These are patterns that recur but haven't been named.
        """
        candidates = []
        for cluster in self.clusters:
            if cluster.promoted:
                continue
            time_span = cluster.last_seen - cluster.first_seen
            if (cluster.occurrence_count >= self.MIN_OBSERVATIONS and
                    time_span >= self.MIN_TIME_SPAN):
                candidates.append(cluster)
        return candidates

    def promote(self, cluster_id: str, name: str, description: str) -> Optional[PatternCluster]:
        """
        Promote a candidate pattern to a named emotion.

        This is how LEM's emotional vocabulary grows.
        The system noticed a recurring pattern, and now it has a name.

        Args:
            cluster_id: ID of the cluster to promote
            name: Name for the new emotion
            description: What this emotion feels like

        Returns:
            The promoted cluster, or None if not found
        """
        for cluster in self.clusters:
            if cluster.id == cluster_id:
                cluster.promoted = True
                cluster.name = name
                cluster.description = description
                self._save()
                return cluster
        return None

    def get_discovery_summary(self) -> Dict:
        """Summary for bridge output."""
        candidates = self.detect_candidates()
        promoted = [c for c in self.clusters if c.promoted]

        summary = {
            "total_unmatched_observations": len(self.unmatched_snapshots),
            "total_clusters": len(self.clusters),
            "candidate_count": len(candidates),
            "promoted_count": len(promoted),
        }

        if candidates:
            summary["candidates"] = []
            for c in candidates:
                # Describe the pattern by its most distinctive drivers
                distinctive = sorted(
                    c.centroid.items(), key=lambda x: abs(x[1]), reverse=True
                )[:3]
                summary["candidates"].append({
                    "id": c.id,
                    "occurrences": c.occurrence_count,
                    "distinctive_drivers": {k: round(v, 2) for k, v in distinctive},
                    "time_span_hours": round((c.last_seen - c.first_seen) / 3600, 1),
                })

        if promoted:
            summary["discovered_emotions"] = [
                {"name": c.name, "description": c.description, "occurrences": c.occurrence_count}
                for c in promoted
            ]

        return summary

    # ── Persistence ──────────────────────────────────────────────────────

    def _save(self):
        """Persist discovery state."""
        data = {
            "next_cluster_id": self._next_cluster_id,
            "clusters": [c.to_dict() for c in self.clusters],
            # Don't persist all unmatched snapshots — only recent ones
            "unmatched_recent": [
                s.to_dict() for s in self.unmatched_snapshots[-100:]
            ],
        }
        path = self.state_dir / "discovery_state.json"
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        tmp.rename(path)

    def _load(self):
        """Load persisted state."""
        path = self.state_dir / "discovery_state.json"
        if not path.exists():
            return

        try:
            with open(path) as f:
                data = json.load(f)

            self._next_cluster_id = data.get("next_cluster_id", 0)
            self.clusters = [
                PatternCluster.from_dict(c) for c in data.get("clusters", [])
            ]
            self.unmatched_snapshots = [
                DriverSnapshot.from_dict(s) for s in data.get("unmatched_recent", [])
            ]
        except (json.JSONDecodeError, KeyError, TypeError):
            pass  # Start fresh
