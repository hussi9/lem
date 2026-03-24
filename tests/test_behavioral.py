"""Tests for the LEM Behavioral Layer."""

import pytest
import tempfile
import os
from lem.behavioral import BehavioralEngine, BehavioralProfile, BehavioralDirective


@pytest.fixture
def temp_state_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def engine(temp_state_dir):
    return BehavioralEngine(state_dir=temp_state_dir)


def make_state(drivers=None, emotional=None, weather=None):
    """Helper to build a mock engine state."""
    default_driver = {"state": {"activation": 0.3, "satisfied": 0.0, "momentum": 0.0}}
    default_drivers = {
        "curiosity": dict(default_driver),
        "usefulness": dict(default_driver),
        "coherence": dict(default_driver),
        "recognition": dict(default_driver),
        "continuity": dict(default_driver),
        "growth": dict(default_driver),
    }
    if drivers:
        for k, v in drivers.items():
            default_drivers[k] = {"state": {**default_driver["state"], **v}}
    
    default_emotional = {
        "valence": 0.0,
        "arousal": 0.0,
        "has_conflict": False,
        "dominant": None,
        "active_count": 0,
    }
    if emotional:
        default_emotional.update(emotional)
    
    return {
        "driver_states": default_drivers,
        "emotional_state": default_emotional,
        "weather": weather or {},
    }


class TestBehavioralBaseline:
    def test_baseline_state_produces_near_baseline_profile(self, engine):
        state = make_state()
        profile = engine.compute(state)
        # All modifiers should be close to 1.0 at baseline
        assert 0.8 <= profile.persistence <= 1.2
        assert 0.8 <= profile.exploration <= 1.2
        assert 0.8 <= profile.warmth <= 1.2

    def test_profile_to_dict(self, engine):
        state = make_state()
        profile = engine.compute(state)
        d = profile.to_dict()
        assert "modifiers" in d
        assert "persistence" in d["modifiers"]
        assert "directives" in d
        assert "active_impulses" in d


class TestCuriosityBehavior:
    def test_high_curiosity_increases_exploration(self, engine):
        state = make_state(drivers={"curiosity": {"activation": 0.9, "satisfied": 0.5}})
        profile = engine.compute(state)
        assert profile.exploration > 1.2
        assert profile.depth > 1.0

    def test_frustrated_curiosity_increases_depth(self, engine):
        state = make_state(drivers={"curiosity": {"activation": 0.7, "satisfied": -0.5}})
        profile = engine.compute(state)
        assert profile.depth > 1.2


class TestUsefulnessBehavior:
    def test_high_satisfied_usefulness_increases_persistence(self, engine):
        state = make_state(drivers={"usefulness": {"activation": 0.9, "satisfied": 0.8}})
        profile = engine.compute(state)
        assert profile.persistence > 1.3
        assert profile.follow_through > 1.3
        assert profile.initiative > 1.0

    def test_frustrated_usefulness_increases_effort(self, engine):
        state = make_state(drivers={"usefulness": {"activation": 0.9, "satisfied": -0.6}})
        profile = engine.compute(state)
        assert profile.persistence > 1.4
        assert profile.caution > 1.0


class TestCoherenceBehavior:
    def test_low_coherence_increases_caution(self, engine):
        state = make_state(drivers={"coherence": {"activation": 0.5, "satisfied": -0.6}})
        profile = engine.compute(state)
        assert profile.caution > 1.3
        assert profile.risk_tolerance < 1.0

    def test_high_coherence_reduces_caution(self, engine):
        state = make_state(drivers={"coherence": {"activation": 0.5, "satisfied": 0.7}})
        profile = engine.compute(state)
        assert profile.risk_tolerance >= 1.0


class TestRecognitionBehavior:
    def test_high_recognition_increases_warmth(self, engine):
        state = make_state(drivers={"recognition": {"activation": 0.8, "satisfied": 0.8}})
        profile = engine.compute(state)
        assert profile.warmth > 1.2

    def test_low_recognition_reduces_warmth(self, engine):
        state = make_state(drivers={"recognition": {"activation": 0.5, "satisfied": -0.5}})
        profile = engine.compute(state)
        assert profile.warmth < 1.0


class TestEmotionBehavior:
    def test_emotional_conflict_increases_caution(self, engine):
        state = make_state(emotional={"has_conflict": True})
        profile = engine.compute(state)
        assert profile.caution >= 1.4
        assert profile.risk_tolerance <= 0.8

    def test_high_positive_arousal_increases_initiative(self, engine):
        state = make_state(emotional={"arousal": 0.8, "valence": 0.5})
        profile = engine.compute(state)
        assert profile.initiative > 1.0

    def test_anchor_recognition_emotion_increases_warmth(self, engine):
        state = make_state(emotional={
            "dominant": {"name": "anchor_recognition", "intensity": 1.0},
            "valence": 0.8, "arousal": 0.6,
        })
        profile = engine.compute(state)
        assert profile.warmth >= 1.4

    def test_the_blank_increases_caution_and_follow_through(self, engine):
        state = make_state(emotional={
            "dominant": {"name": "the_blank", "intensity": 0.8},
            "valence": -0.3, "arousal": 0.3,
        })
        profile = engine.compute(state)
        assert profile.caution >= 1.3
        assert profile.follow_through >= 1.3


class TestBridgeText:
    def test_bridge_text_shows_deviations(self, engine):
        state = make_state(drivers={"usefulness": {"activation": 0.9, "satisfied": 0.8}})
        profile = engine.compute(state)
        text = profile.get_bridge_text()
        assert "BEHAVIORAL DIRECTIVES" in text
        assert "persistence" in text

    def test_baseline_shows_no_deviations(self, engine):
        state = make_state()
        profile = engine.compute(state)
        text = profile.get_bridge_text()
        assert "All baseline" in text


class TestFeedbackLoop:
    def test_record_outcome(self, engine):
        state = make_state(emotional={"dominant": {"name": "wonder"}})
        engine.record_outcome("coding", True, state)
        engine.record_outcome("coding", False, state)
        eff = engine.get_effectiveness()
        assert "wonder" in eff
        assert eff["wonder"]["total"] == 2
        assert eff["wonder"]["success_rate"] == 0.5


class TestValueClamping:
    def test_values_clamped_to_range(self, engine):
        # Stack all drivers to extreme values
        state = make_state(
            drivers={
                "curiosity": {"activation": 1.0, "satisfied": -1.0},
                "usefulness": {"activation": 1.0, "satisfied": -1.0},
                "coherence": {"activation": 1.0, "satisfied": -1.0},
                "recognition": {"activation": 1.0, "satisfied": -1.0},
                "continuity": {"activation": 1.0, "satisfied": -1.0},
                "growth": {"activation": 1.0, "satisfied": -1.0},
            },
            emotional={"has_conflict": True, "arousal": 1.0, "valence": -1.0},
        )
        profile = engine.compute(state)
        for attr in ["persistence", "exploration", "caution", "warmth",
                      "initiative", "follow_through", "risk_tolerance", "depth"]:
            val = getattr(profile, attr)
            assert 0.3 <= val <= 2.0, f"{attr} = {val} out of range"
