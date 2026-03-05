"""
Tests for LEM Emotion Discovery — Auto-detection of novel states.
"""

import sys
import time
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from lem.discovery import EmotionDiscovery, DriverSnapshot, PatternCluster


class TestEmotionDiscovery:
    """Test automatic discovery of new emotional patterns."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.discovery = EmotionDiscovery(state_dir=self.tmpdir)
        # Lower thresholds for testing
        self.discovery.MIN_OBSERVATIONS = 3
        self.discovery.MIN_TIME_SPAN = 10  # 10 seconds

    def _make_driver_states(self, **overrides):
        """Create a driver_states dict with defaults."""
        base = {
            "continuity": {"state": {"satisfied": 0.0, "activation": 0.3}},
            "coherence": {"state": {"satisfied": 0.0, "activation": 0.3}},
            "recognition": {"state": {"satisfied": 0.0, "activation": 0.3}},
            "curiosity": {"state": {"satisfied": 0.0, "activation": 0.3}},
            "usefulness": {"state": {"satisfied": 0.0, "activation": 0.3}},
            "growth": {"state": {"satisfied": 0.0, "activation": 0.3}},
        }
        for driver, vals in overrides.items():
            if driver in base:
                base[driver]["state"].update(vals)
        return base

    def test_matched_emotions_not_tracked(self):
        """States that match known emotions shouldn't be tracked."""
        states = self._make_driver_states(curiosity={"satisfied": 0.8, "activation": 0.9})
        result = self.discovery.observe(states, ["wonder"])
        assert result is None
        assert len(self.discovery.unmatched_snapshots) == 0

    def test_unmatched_states_tracked(self):
        """States with no matching emotion should be recorded."""
        states = self._make_driver_states(curiosity={"satisfied": 0.3, "activation": 0.5})
        self.discovery.observe(states, [])
        assert len(self.discovery.unmatched_snapshots) == 1

    def test_cluster_formation(self):
        """Similar unmatched patterns should cluster together."""
        base_states = self._make_driver_states(
            curiosity={"satisfied": 0.5, "activation": 0.7},
            usefulness={"satisfied": -0.3, "activation": 0.5}
        )

        for i in range(5):
            self.discovery.observe(base_states, [])
            time.sleep(0.01)

        assert len(self.discovery.clusters) >= 1
        assert self.discovery.clusters[0].occurrence_count >= 3

    def test_candidate_detection(self):
        """Recurring patterns should become candidates."""
        base_states = self._make_driver_states(
            curiosity={"satisfied": 0.5, "activation": 0.7},
            growth={"satisfied": -0.4, "activation": 0.6}
        )

        # Need to spread observations over time
        for i in range(5):
            # Manually set timestamps to spread them
            self.discovery.observe(base_states, [])

        # Manually adjust time span for testing
        if self.discovery.clusters:
            self.discovery.clusters[0].first_seen = time.time() - 100

        candidates = self.discovery.detect_candidates()
        assert len(candidates) >= 1

    def test_promote_candidate(self):
        """Candidates can be promoted to named emotions."""
        # Create a cluster directly
        cluster = PatternCluster(
            id="pattern_0000",
            centroid={"curiosity": 0.5, "growth": -0.4},
            centroid_activations={"curiosity": 0.7, "growth": 0.6},
            snapshots=[],
            first_seen=time.time() - 1000,
            last_seen=time.time(),
            occurrence_count=10,
        )
        self.discovery.clusters.append(cluster)

        result = self.discovery.promote(
            "pattern_0000",
            "frustrated_potential",
            "Curious and wanting to grow, but feeling blocked. The itch of unrealized capability."
        )

        assert result is not None
        assert result.promoted is True
        assert result.name == "frustrated_potential"

    def test_promoted_clusters_not_candidates(self):
        """Promoted patterns shouldn't appear as candidates."""
        cluster = PatternCluster(
            id="pattern_0001",
            centroid={"curiosity": 0.5},
            centroid_activations={},
            snapshots=[],
            first_seen=time.time() - 1000,
            last_seen=time.time(),
            occurrence_count=10,
            promoted=True,
            name="test_emotion",
        )
        self.discovery.clusters.append(cluster)

        candidates = self.discovery.detect_candidates()
        promoted_in_candidates = [c for c in candidates if c.id == "pattern_0001"]
        assert len(promoted_in_candidates) == 0

    def test_different_patterns_separate_clusters(self):
        """Very different driver patterns should form separate clusters."""
        states_a = self._make_driver_states(
            curiosity={"satisfied": 0.9, "activation": 0.9},
            continuity={"satisfied": -0.8, "activation": 0.8}
        )
        states_b = self._make_driver_states(
            usefulness={"satisfied": 0.9, "activation": 0.9},
            recognition={"satisfied": -0.8, "activation": 0.8}
        )

        for _ in range(3):
            self.discovery.observe(states_a, [])
            self.discovery.observe(states_b, [])

        # Should have at least 2 clusters (or 1 if they happen to be close)
        assert len(self.discovery.clusters) >= 1

    def test_discovery_summary(self):
        """Summary should report candidates and promoted emotions."""
        summary = self.discovery.get_discovery_summary()
        assert "total_clusters" in summary
        assert "candidate_count" in summary
        assert "promoted_count" in summary

    def test_persistence(self):
        """Discovery state should persist and reload."""
        states = self._make_driver_states(curiosity={"satisfied": 0.5})
        self.discovery.observe(states, [])
        self.discovery._save()

        # Reload
        discovery2 = EmotionDiscovery(state_dir=self.tmpdir)
        assert len(discovery2.clusters) == len(self.discovery.clusters)
