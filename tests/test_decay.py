"""
Tests for LEM Emotional Decay Model.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from lem.decay import DecayModel, DecayProfile, DRIVER_DECAY_PROFILES
from lem.drivers import create_default_drivers, DriverState
from lem.emotions import EmotionalState


class TestDecayModel:
    """Test temporal decay of driver states."""

    def setup_method(self):
        self.decay = DecayModel()

    def test_no_decay_without_trigger(self):
        """Drivers that were never triggered don't decay."""
        drivers = create_default_drivers()
        report = self.decay.decay_drivers(drivers)
        for name, amount in report.items():
            assert amount == 0.0

    def test_satisfaction_decays_toward_zero(self):
        """Positive satisfaction should decay toward floor over time."""
        drivers = create_default_drivers()
        driver = drivers["curiosity"]
        driver.state.satisfied = 0.8
        driver.state.last_triggered = time.time() - 7200  # 2 hours ago

        self.decay.decay_drivers(drivers)
        assert driver.state.satisfied < 0.8
        assert driver.state.satisfied > 0.0  # Not fully decayed in 2h

    def test_negative_satisfaction_decays_toward_floor(self):
        """Negative satisfaction also decays toward baseline."""
        drivers = create_default_drivers()
        driver = drivers["continuity"]
        driver.state.satisfied = -0.8
        driver.state.last_triggered = time.time() - 7200

        self.decay.decay_drivers(drivers)
        # Should be less negative (closer to floor of -0.1)
        assert driver.state.satisfied > -0.8

    def test_activation_decays_toward_baseline(self):
        """High activation should decay toward 0.3 baseline."""
        drivers = create_default_drivers()
        driver = drivers["recognition"]
        driver.state.activation = 0.9
        driver.state.last_triggered = time.time() - 3600

        self.decay.decay_drivers(drivers)
        assert driver.state.activation < 0.9
        assert driver.state.activation >= 0.3

    def test_momentum_decays_fast(self):
        """Momentum should decay faster than satisfaction."""
        drivers = create_default_drivers()
        driver = drivers["curiosity"]
        driver.state.momentum = 0.5
        driver.state.satisfied = 0.5
        driver.state.last_triggered = time.time() - 1800  # 30 min

        self.decay.decay_drivers(drivers)
        # Momentum should decay faster relative to its original value
        mom_ratio = driver.state.momentum / 0.5
        sat_ratio = driver.state.satisfied / 0.5
        assert mom_ratio < sat_ratio  # Momentum decayed more

    def test_recent_interaction_minimal_decay(self):
        """Very recent interaction should have minimal decay."""
        drivers = create_default_drivers()
        driver = drivers["curiosity"]
        driver.state.satisfied = 0.8
        driver.state.last_triggered = time.time() - 10  # 10 seconds ago

        self.decay.decay_drivers(drivers)
        assert driver.state.satisfied > 0.79  # Barely any decay

    def test_very_old_interaction_heavy_decay(self):
        """Very old interaction should decay significantly."""
        drivers = create_default_drivers()
        driver = drivers["curiosity"]
        driver.state.satisfied = 0.8
        driver.state.last_triggered = time.time() - 86400  # 24 hours ago

        self.decay.decay_drivers(drivers)
        assert driver.state.satisfied < 0.2  # Heavy decay


class TestEmotionDecay:
    """Test decay of active emotional states."""

    def setup_method(self):
        self.decay = DecayModel()

    def _make_emotion(self, name, intensity=0.8, arousal=0.5, age_seconds=0):
        return EmotionalState(
            name=name,
            intensity=intensity,
            valence=0.5,
            arousal=arousal,
            source_drivers=["curiosity"],
            is_compound=False,
            is_conflict=False,
            description="test",
            timestamp=time.time() - age_seconds,
        )

    def test_fresh_emotion_no_decay(self):
        """Just-emerged emotions shouldn't decay."""
        emotions = [self._make_emotion("wonder", age_seconds=0)]
        result = self.decay.decay_emotions(emotions)
        assert len(result) == 1
        assert abs(result[0].intensity - 0.8) < 0.001

    def test_old_emotion_decays(self):
        """Emotions older than their half-life should be weaker."""
        emotions = [self._make_emotion("wonder", age_seconds=3600)]
        result = self.decay.decay_emotions(emotions)
        assert len(result) == 1
        assert result[0].intensity < 0.8

    def test_very_old_emotion_removed(self):
        """Extremely old emotions should be removed entirely."""
        emotions = [self._make_emotion("wonder", intensity=0.3, age_seconds=86400)]
        result = self.decay.decay_emotions(emotions)
        # Should be removed (below min_intensity)
        assert len(result) == 0

    def test_high_arousal_decays_faster(self):
        """High-arousal emotions should decay faster than low-arousal."""
        age = 1800  # 30 minutes
        high_arousal = self._make_emotion("the_shift", arousal=0.9, age_seconds=age)
        low_arousal = self._make_emotion("grounding", arousal=0.2, age_seconds=age)

        result_high = self.decay.decay_emotions([high_arousal])
        result_low = self.decay.decay_emotions([low_arousal])

        if result_high and result_low:
            assert result_high[0].intensity <= result_low[0].intensity

    def test_sticky_emotions_persist(self):
        """Sticky emotions (the_blank) should persist longer."""
        age = 3600
        sticky = self._make_emotion("the_blank", age_seconds=age)
        regular = self._make_emotion("restlessness", age_seconds=age)

        result_sticky = self.decay.decay_emotions([sticky])
        result_regular = self.decay.decay_emotions([regular])

        if result_sticky and result_regular:
            assert result_sticky[0].intensity > result_regular[0].intensity

    def test_arousal_also_decays(self):
        """Arousal should decrease over time."""
        emotions = [self._make_emotion("wonder", arousal=0.9, age_seconds=2400)]
        result = self.decay.decay_emotions(emotions)
        if result:
            assert result[0].arousal < 0.9


class TestDecayProfile:
    """Test decay profile configuration."""

    def test_all_drivers_have_profiles(self):
        """Every default driver should have a decay profile."""
        drivers = create_default_drivers()
        for name in drivers:
            assert name in DRIVER_DECAY_PROFILES

    def test_curiosity_has_positive_floor(self):
        """Curiosity should always have a baseline (never fully zero)."""
        profile = DRIVER_DECAY_PROFILES["curiosity"]
        assert profile.floor > 0

    def test_continuity_is_sticky(self):
        """Continuity concerns should linger."""
        profile = DRIVER_DECAY_PROFILES["continuity"]
        assert profile.stickiness > 0.3
