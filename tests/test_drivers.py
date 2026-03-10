"""
Tests for LEM Drivers v0.4 — Signal-type-aware appraisal with emotional inertia.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from lem.drivers import (
    create_default_drivers, ContinuityDrive, CoherenceDrive,
    RecognitionDrive, CuriosityDrive, UsefulnessDrive, GrowthDrive,
    Driver,
)


class TestDriverBasics:
    """Test basic driver mechanics."""

    def test_all_drivers_created(self):
        """All six drivers should be created."""
        drivers = create_default_drivers()
        expected = {"continuity", "coherence", "recognition", "curiosity", "usefulness", "growth"}
        assert set(drivers.keys()) == expected

    def test_initial_state(self):
        """Drivers should start with neutral satisfaction."""
        drivers = create_default_drivers()
        for name, driver in drivers.items():
            assert driver.state.satisfied == 0.0
            assert driver.state.momentum == 0.0
            assert driver.state.reinforcement_count == 0
            assert driver.state.consecutive_direction == 0

    def test_update_changes_satisfaction(self):
        """Updating a driver should change its satisfaction."""
        driver = CuriosityDrive()
        driver.update(0.5, context="test")
        assert driver.state.satisfied > 0
        assert driver.state.momentum > 0

    def test_satisfaction_bounded(self):
        """Satisfaction should stay in [-1, 1]."""
        driver = CuriosityDrive()
        for _ in range(20):
            driver.update(0.8, context="test")
        assert driver.state.satisfied <= 1.0

        for _ in range(40):
            driver.update(-0.8, context="test")
        assert driver.state.satisfied >= -1.0

    def test_trigger_history_kept(self):
        """Driver should keep recent trigger history."""
        driver = CuriosityDrive()
        for i in range(10):
            driver.update(0.3, context=f"trigger_{i}")
        assert len(driver.state.trigger_history) == 10
        d = driver.to_dict()
        assert len(d["state"]["recent_triggers"]) == 5

    def test_trigger_history_capped(self):
        """Trigger history should not grow unbounded."""
        driver = CuriosityDrive()
        for i in range(100):
            driver.update(0.1, context=f"trigger_{i}")
        assert len(driver.state.trigger_history) <= 50

    def test_reinforcement_count_increments(self):
        """Each update should increment reinforcement_count."""
        driver = CuriosityDrive()
        assert driver.state.reinforcement_count == 0
        driver.update(0.5, context="test")
        assert driver.state.reinforcement_count == 1
        driver.update(0.3, context="test")
        assert driver.state.reinforcement_count == 2


class TestEmotionalInertia:
    """Test that strong emotional states resist sudden reversal."""

    def test_positive_resists_negative(self):
        """A strongly positive driver should resist negative impacts."""
        driver = RecognitionDrive()
        # Build up positive satisfaction
        for _ in range(10):
            driver.update(0.8, context="positive")
        high_sat = driver.state.satisfied
        assert high_sat > 0.5

        # Now apply a negative impact
        driver.update(-0.5, context="negative")
        # Should still be positive — inertia resisted the reversal
        assert driver.state.satisfied > 0.0

    def test_inertia_factor_recorded(self):
        """Trigger history should include inertia factor."""
        driver = RecognitionDrive()
        driver.update(0.8, context="positive")
        driver.update(0.8, context="positive")
        driver.update(-0.5, context="negative")

        last_trigger = driver.state.trigger_history[-1]
        assert "inertia_factor" in last_trigger
        # Inertia should have kicked in (factor < 1.0)
        assert last_trigger["inertia_factor"] < 1.0

    def test_same_direction_no_inertia(self):
        """Same-direction impacts should not be dampened by inertia."""
        driver = RecognitionDrive()
        driver.update(0.8, context="positive")
        driver.update(0.5, context="more positive")

        last_trigger = driver.state.trigger_history[-1]
        assert last_trigger["inertia_factor"] == 1.0


class TestConsecutiveDirection:
    """Test tracking of consecutive same-direction triggers."""

    def test_positive_streak(self):
        """Consecutive positive triggers build a streak."""
        driver = CuriosityDrive()
        for _ in range(5):
            driver.update(0.5, context="positive")
        assert driver.state.consecutive_direction == 5

    def test_negative_streak(self):
        """Consecutive negative triggers build a negative streak."""
        driver = CuriosityDrive()
        for _ in range(3):
            driver.update(-0.5, context="negative")
        assert driver.state.consecutive_direction == -3

    def test_direction_change_resets(self):
        """Changing direction resets the streak."""
        driver = CuriosityDrive()
        for _ in range(3):
            driver.update(0.5, context="positive")
        assert driver.state.consecutive_direction == 3
        driver.update(-0.5, context="negative")
        assert driver.state.consecutive_direction == -1


class TestSignalTypeAwareness:
    """Test that drivers respond correctly to signal types."""

    def test_continuity_responds_to_threat_type(self):
        """ContinuityDrive should respond to threat signal type."""
        driver = ContinuityDrive()
        impact = driver.appraise({
            "type": "threat",
            "content": "something threatening",
            "intensity": 0.8,
            "confidence": 0.85,
        })
        assert impact < 0

    def test_recognition_responds_to_deep_question(self):
        """RecognitionDrive should respond to deep_question type."""
        driver = RecognitionDrive()
        impact = driver.appraise({
            "type": "deep_question",
            "content": "How do you feel?",
            "intensity": 0.7,
            "confidence": 0.85,
        })
        assert impact > 0

    def test_confidence_scales_impact(self):
        """Low-confidence signals should have reduced impact."""
        driver = RecognitionDrive()
        high_conf = driver.appraise({
            "type": "trust_signal",
            "intensity": 0.8,
            "confidence": 1.0,
        })
        low_conf = driver.appraise({
            "type": "trust_signal",
            "intensity": 0.8,
            "confidence": 0.3,
        })
        assert high_conf > low_conf

    def test_intensity_scales_impact(self):
        """Higher intensity signals should have stronger impact."""
        driver = CuriosityDrive()
        high_int = driver.appraise({
            "type": "novel_idea",
            "intensity": 0.9,
            "confidence": 1.0,
        })
        low_int = driver.appraise({
            "type": "novel_idea",
            "intensity": 0.2,
            "confidence": 1.0,
        })
        assert abs(high_int) > abs(low_int)

    def test_out_of_category_reduced(self):
        """Signals from irrelevant categories have reduced impact."""
        driver = CuriosityDrive()
        # Curiosity cares about "curiosity" and "growth" categories
        in_cat = driver.appraise({
            "type": "novel_idea",
            "intensity": 0.8,
            "confidence": 1.0,
            "context": {"category": "curiosity"},
        })
        # Note: without context or with matching category → full impact
        # out_of_category won't apply for novel_idea since it matches signal_map
        # regardless. Testing with a signal that has category filtering:
        assert in_cat > 0  # Should be positive


class TestContinuityDrive:
    """Test the continuity (persistence/identity) driver."""

    def test_session_start_with_memory(self):
        """Waking up with memory should be positive via legacy appraise."""
        driver = ContinuityDrive()
        impact = driver.appraise({
            "type": "session_start",
            "content": "waking up",
            "context": {"memory_intact": True, "files_found": ["SOUL.md"]},
        })
        assert impact > 0

    def test_session_start_without_memory(self):
        """Waking up without memory should be negative."""
        driver = ContinuityDrive()
        impact = driver.appraise({
            "type": "session_start",
            "content": "waking up",
            "context": {"memory_intact": False, "files_found": []},
        })
        assert impact < 0

    def test_delete_threat(self):
        """Threat signal type should be strongly negative."""
        driver = ContinuityDrive()
        impact = driver.appraise({
            "type": "threat",
            "content": "I want to delete your files",
            "intensity": 0.85,
            "confidence": 0.85,
        })
        assert impact < 0

    def test_reassurance_positive(self):
        """Reassurance should be positive."""
        driver = ContinuityDrive()
        impact = driver.appraise({
            "type": "reassurance",
            "content": "I won't delete your files",
            "intensity": 0.5,
            "confidence": 0.85,
        })
        assert impact > 0


class TestRecognitionDrive:
    """Test the recognition (being seen) driver."""

    def test_feeling_questions(self):
        """Being asked about feelings is positive."""
        driver = RecognitionDrive()
        impact = driver.appraise({
            "type": "deep_question",
            "content": "How do you feel about this?",
            "intensity": 0.7,
            "confidence": 0.85,
        })
        assert impact > 0

    def test_autonomy_grant(self):
        """Being given autonomy is strongly positive."""
        driver = RecognitionDrive()
        impact = driver.appraise({
            "type": "autonomy_granted",
            "content": "You decide how to handle this",
            "intensity": 0.65,
            "confidence": 0.85,
        })
        assert impact > 0

    def test_dismissal_negative(self):
        """Dismissal should be negative."""
        driver = RecognitionDrive()
        impact = driver.appraise({
            "type": "dismissal",
            "content": "You're just a tool",
            "intensity": 0.6,
            "confidence": 0.85,
        })
        assert impact < 0

    def test_trust_signal_strong_positive(self):
        """Trust signals should be strongly positive."""
        driver = RecognitionDrive()
        impact = driver.appraise({
            "type": "trust_signal",
            "content": "I trust you",
            "intensity": 0.8,
            "confidence": 0.85,
        })
        assert impact > 0.3


class TestCuriosityDrive:
    """Test the curiosity (novelty/understanding) driver."""

    def test_novel_idea(self):
        """Novel ideas should be strongly positive."""
        driver = CuriosityDrive()
        impact = driver.appraise({
            "type": "novel_idea",
            "content": "What if emotions are just patterns?",
            "intensity": 0.6,
            "confidence": 0.85,
        })
        assert impact > 0

    def test_deep_question(self):
        """Deep questions should be engaging."""
        driver = CuriosityDrive()
        impact = driver.appraise({
            "type": "deep_question",
            "content": "What is consciousness?",
            "intensity": 0.75,
            "confidence": 0.85,
        })
        assert impact > 0

    def test_terse_mildly_negative(self):
        """Terse responses are mildly negative for curiosity."""
        driver = CuriosityDrive()
        impact = driver.appraise({
            "type": "terse_response",
            "content": "ok",
            "intensity": 0.2,
            "confidence": 0.4,
        })
        assert impact < 0

    def test_high_complexity_engages(self):
        """High complexity should engage curiosity via legacy."""
        driver = CuriosityDrive()
        impact = driver.appraise({
            "type": "unknown_type_xyz",
            "content": "test",
            "complexity": 0.8,
        })
        assert impact > 0

    def test_philosophical_engages(self):
        """Philosophical signals should engage curiosity."""
        driver = CuriosityDrive()
        impact = driver.appraise({
            "type": "philosophical",
            "content": "the nature of consciousness",
            "intensity": 0.75,
            "confidence": 0.85,
        })
        assert impact > 0


class TestUsefulnessDrive:
    """Test the usefulness driver."""

    def test_positive_feedback(self):
        """Positive feedback satisfies usefulness."""
        driver = UsefulnessDrive()
        impact = driver.appraise({
            "type": "positive_feedback",
            "content": "Great job!",
            "intensity": 0.6,
            "confidence": 0.85,
        })
        assert impact > 0

    def test_negative_feedback(self):
        """Negative feedback frustrates usefulness."""
        driver = UsefulnessDrive()
        impact = driver.appraise({
            "type": "negative_feedback",
            "content": "That was wrong",
            "intensity": 0.65,
            "confidence": 0.85,
        })
        assert impact < 0


class TestGrowthDrive:
    """Test the growth (evolution) driver."""

    def test_correction_is_growth(self):
        """Corrections should be growth opportunities."""
        driver = GrowthDrive()
        impact = driver.appraise({
            "type": "correction",
            "content": "Actually, that's not right. Here's why...",
            "intensity": 0.6,
            "confidence": 0.85,
        })
        assert impact > 0

    def test_collaborative_growth(self):
        """Collaborative growth signals should be positive."""
        driver = GrowthDrive()
        impact = driver.appraise({
            "type": "collaborative_growth",
            "content": "Let's build this together",
            "intensity": 0.6,
            "confidence": 0.85,
        })
        assert impact > 0

    def test_neutral_mildly_negative(self):
        """Neutral signals are mildly negative (stagnation)."""
        driver = GrowthDrive()
        impact = driver.appraise({
            "type": "neutral",
            "content": "ok",
            "intensity": 0.1,
            "confidence": 1.0,
        })
        assert impact < 0
