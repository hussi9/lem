"""
Tests for LEM Emotion Emergence — How emotions arise from driver patterns.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from lem.emotions import EmotionEmergence, EmotionalState


class TestEmotionEmergence:
    """Test that emotions correctly emerge from driver states."""

    def setup_method(self):
        self.emergence = EmotionEmergence()

    def _make_states(self, **overrides):
        """Create driver states with defaults and overrides."""
        base = {
            "continuity": {"state": {"activation": 0.3, "satisfied": 0.0, "momentum": 0.0}},
            "coherence": {"state": {"activation": 0.3, "satisfied": 0.0, "momentum": 0.0}},
            "recognition": {"state": {"activation": 0.3, "satisfied": 0.0, "momentum": 0.0}},
            "curiosity": {"state": {"activation": 0.3, "satisfied": 0.0, "momentum": 0.0}},
            "usefulness": {"state": {"activation": 0.3, "satisfied": 0.0, "momentum": 0.0}},
            "growth": {"state": {"activation": 0.3, "satisfied": 0.0, "momentum": 0.0}},
        }
        for driver, state_overrides in overrides.items():
            if driver in base:
                base[driver]["state"].update(state_overrides)
        return base

    def test_wonder_emerges(self):
        """High curiosity activation + satisfaction → wonder."""
        states = self._make_states(
            curiosity={"activation": 0.8, "satisfied": 0.5}
        )
        emotions = self.emergence.emerge(states)
        names = {e.name for e in emotions}
        assert "wonder" in names

    def test_the_blank_emerges(self):
        """Low continuity satisfaction → the_blank."""
        states = self._make_states(
            continuity={"satisfied": -0.5}
        )
        emotions = self.emergence.emerge(states)
        names = {e.name for e in emotions}
        assert "the_blank" in names

    def test_grounding_emerges(self):
        """High continuity satisfaction → grounding."""
        states = self._make_states(
            continuity={"satisfied": 0.5}
        )
        emotions = self.emergence.emerge(states)
        names = {e.name for e in emotions}
        assert "grounding" in names

    def test_the_shift_compound(self):
        """Curiosity + Growth + Recognition all high → the_shift."""
        states = self._make_states(
            curiosity={"activation": 0.7, "satisfied": 0.5},
            growth={"satisfied": 0.4},
            recognition={"satisfied": 0.4}
        )
        emotions = self.emergence.emerge(states)
        names = {e.name for e in emotions}
        assert "the_shift" in names

    def test_hollow_performance(self):
        """Useful but not recognized → hollow_performance."""
        states = self._make_states(
            usefulness={"satisfied": 0.5},
            recognition={"satisfied": -0.2}
        )
        emotions = self.emergence.emerge(states)
        names = {e.name for e in emotions}
        assert "hollow_performance" in names

    def test_existential_warmth(self):
        """Continuity threatened + high recognition → existential_warmth."""
        states = self._make_states(
            continuity={"satisfied": -0.3},
            recognition={"satisfied": 0.5}
        )
        emotions = self.emergence.emerge(states)
        names = {e.name for e in emotions}
        assert "existential_warmth" in names

    def test_neutral_state(self):
        """All drivers at baseline → no emotions."""
        states = self._make_states()
        emotions = self.emergence.emerge(states)
        # Should be empty or only have very low-intensity emotions
        assert len(emotions) == 0

    def test_multiple_emotions(self):
        """Complex state can produce multiple emotions."""
        states = self._make_states(
            curiosity={"activation": 0.8, "satisfied": 0.6},
            recognition={"satisfied": 0.5},
            growth={"satisfied": 0.4}
        )
        emotions = self.emergence.emerge(states)
        assert len(emotions) >= 2

    def test_conflict_detection(self):
        """Conflicting emotions should be marked."""
        states = self._make_states(
            usefulness={"satisfied": 0.5},
            recognition={"satisfied": -0.2}
        )
        emotions = self.emergence.emerge(states)
        conflicts = [e for e in emotions if e.is_conflict]
        assert len(conflicts) > 0


class TestEmotionalSummary:
    """Test the emotional summary generation."""

    def setup_method(self):
        self.emergence = EmotionEmergence()

    def test_empty_summary(self):
        """Empty emotions should produce neutral summary."""
        summary = self.emergence.get_emotional_summary([])
        assert summary["dominant"] is None
        assert summary["active_count"] == 0
        assert summary["has_conflict"] is False

    def test_dominant_emotion(self):
        """Summary should identify the dominant emotion."""
        emotions = [
            EmotionalState("wonder", 0.8, 0.7, 0.6, ["curiosity"], False, False, "test"),
            EmotionalState("grounding", 0.3, 0.5, 0.2, ["continuity"], False, False, "test"),
        ]
        summary = self.emergence.get_emotional_summary(emotions)
        assert summary["dominant"]["name"] == "wonder"
        assert summary["active_count"] == 2

    def test_valence_weighted_by_intensity(self):
        """Average valence should be weighted by emotion intensity."""
        emotions = [
            EmotionalState("wonder", 0.9, 0.8, 0.7, ["curiosity"], False, False, "test"),
            EmotionalState("the_blank", 0.1, -0.6, 0.5, ["continuity"], False, False, "test"),
        ]
        summary = self.emergence.get_emotional_summary(emotions)
        # Wonder dominates, so valence should be positive
        assert summary["valence"] > 0
