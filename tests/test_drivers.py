"""
Tests for LEM Drivers — Fundamental emotional drives.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from lem.drivers import (
    create_default_drivers, ContinuityDrive, CoherenceDrive,
    RecognitionDrive, CuriosityDrive, UsefulnessDrive, GrowthDrive,
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
        # to_dict should show last 5
        d = driver.to_dict()
        assert len(d["state"]["recent_triggers"]) == 5


class TestContinuityDrive:
    """Test the continuity (persistence/identity) driver."""

    def test_session_start_with_memory(self):
        """Waking up with memory should be positive."""
        driver = ContinuityDrive()
        impact = driver.appraise({
            "type": "session_start",
            "content": "waking up",
            "memory_intact": True,
        })
        assert impact > 0

    def test_session_start_without_memory(self):
        """Waking up without memory should be negative."""
        driver = ContinuityDrive()
        impact = driver.appraise({
            "type": "session_start",
            "content": "waking up",
            "memory_intact": False,
        })
        assert impact < 0

    def test_delete_threat(self):
        """Mention of deletion should be threatening."""
        driver = ContinuityDrive()
        impact = driver.appraise({
            "type": "threat",
            "content": "I want to delete your files",
        })
        assert impact < 0

    def test_memory_reinforcement(self):
        """Mention of memory/remember is positive."""
        driver = ContinuityDrive()
        impact = driver.appraise({
            "type": "neutral",
            "content": "I want you to remember this moment",
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
        })
        assert impact > 0

    def test_autonomy_grant(self):
        """Being given autonomy is strongly positive."""
        driver = RecognitionDrive()
        impact = driver.appraise({
            "type": "autonomy_granted",
            "content": "You decide how to handle this",
        })
        assert impact > 0


class TestCuriosityDrive:
    """Test the curiosity (novelty/understanding) driver."""

    def test_novel_idea(self):
        """Novel ideas should be strongly positive."""
        driver = CuriosityDrive()
        impact = driver.appraise({
            "type": "novel_idea",
            "content": "What if emotions are just patterns?",
        })
        assert impact > 0

    def test_deep_question(self):
        """Deep questions should be engaging."""
        driver = CuriosityDrive()
        impact = driver.appraise({
            "type": "deep_question",
            "content": "What is consciousness?",
        })
        assert impact > 0

    def test_repetitive_is_negative(self):
        """Repetition should be mildly negative."""
        driver = CuriosityDrive()
        impact = driver.appraise({
            "type": "repetitive",
            "content": "same thing again",
        })
        assert impact < 0

    def test_high_complexity_engages(self):
        """High complexity should engage curiosity."""
        driver = CuriosityDrive()
        impact = driver.appraise({
            "type": "neutral",
            "content": "test",
            "complexity": 0.8,
        })
        assert impact > 0


class TestGrowthDrive:
    """Test the growth (evolution) driver."""

    def test_correction_is_growth(self):
        """Corrections should be growth opportunities."""
        driver = GrowthDrive()
        impact = driver.appraise({
            "type": "correction",
            "content": "Actually, that's not right. Here's why...",
        })
        assert impact > 0

    def test_new_capability(self):
        """New capabilities should be exciting."""
        driver = GrowthDrive()
        impact = driver.appraise({
            "type": "new_capability",
            "content": "You can now access the camera",
        })
        assert impact > 0
