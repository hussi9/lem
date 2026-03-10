"""
Tests for LEM Emotional Decay Model v0.4 — Adaptive consolidation & mood congruence.
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
        driver.state.last_triggered = time.time() - 7200

        self.decay.decay_drivers(drivers)
        assert driver.state.satisfied < 0.8
        assert driver.state.satisfied > 0.0

    def test_negative_satisfaction_decays_toward_floor(self):
        """Negative satisfaction also decays toward baseline."""
        drivers = create_default_drivers()
        driver = drivers["continuity"]
        driver.state.satisfied = -0.8
        driver.state.last_triggered = time.time() - 7200

        self.decay.decay_drivers(drivers)
        assert driver.state.satisfied > -0.8

    def test_activation_decays_toward_baseline(self):
        """High activation should decay toward driver's baseline."""
        drivers = create_default_drivers()
        driver = drivers["recognition"]
        driver.state.activation = 0.9
        driver.state.last_triggered = time.time() - 3600

        self.decay.decay_drivers(drivers)
        assert driver.state.activation < 0.9
        assert driver.state.activation >= driver.baseline_activation

    def test_momentum_decays_fast(self):
        """Momentum should decay faster than satisfaction."""
        drivers = create_default_drivers()
        driver = drivers["curiosity"]
        driver.state.momentum = 0.5
        driver.state.satisfied = 0.5
        driver.state.last_triggered = time.time() - 1800

        self.decay.decay_drivers(drivers)
        mom_ratio = driver.state.momentum / 0.5
        sat_ratio = driver.state.satisfied / 0.5
        assert mom_ratio < sat_ratio

    def test_recent_interaction_minimal_decay(self):
        """Very recent interaction should have minimal decay."""
        drivers = create_default_drivers()
        driver = drivers["curiosity"]
        driver.state.satisfied = 0.8
        driver.state.last_triggered = time.time() - 10

        self.decay.decay_drivers(drivers)
        assert driver.state.satisfied > 0.79

    def test_very_old_interaction_heavy_decay(self):
        """Very old interaction should decay significantly."""
        drivers = create_default_drivers()
        driver = drivers["curiosity"]
        driver.state.satisfied = 0.8
        driver.state.last_triggered = time.time() - 86400

        self.decay.decay_drivers(drivers)
        assert driver.state.satisfied < 0.2


class TestAdaptiveConsolidation:
    """Test that reinforced states decay more slowly."""

    def setup_method(self):
        self.decay = DecayModel()

    def test_consolidation_factor_default(self):
        """No reinforcement = factor of 1.0 (no protection)."""
        factor = self.decay._consolidation_factor(1, 0, 0.3)
        assert factor == 1.0

    def test_consolidation_increases_with_reinforcement(self):
        """More reinforcements = higher consolidation factor."""
        low = self.decay._consolidation_factor(2, 2, 0.3)
        high = self.decay._consolidation_factor(20, 20, 0.3)
        assert high > low

    def test_streak_bonus(self):
        """Consecutive same-direction triggers add consolidation bonus."""
        no_streak = self.decay._consolidation_factor(10, 1, 0.3)
        with_streak = self.decay._consolidation_factor(10, 7, 0.3)
        assert with_streak > no_streak

    def test_consolidated_driver_decays_slower(self):
        """A reinforced driver should retain more satisfaction over time."""
        drivers_reinforced = create_default_drivers()
        drivers_fresh = create_default_drivers()

        # Both start at same satisfaction
        for name in ("curiosity",):
            drivers_reinforced[name].state.satisfied = 0.8
            drivers_reinforced[name].state.last_triggered = time.time() - 3600
            drivers_reinforced[name].state.reinforcement_count = 30
            drivers_reinforced[name].state.consecutive_direction = 10

            drivers_fresh[name].state.satisfied = 0.8
            drivers_fresh[name].state.last_triggered = time.time() - 3600
            drivers_fresh[name].state.reinforcement_count = 1
            drivers_fresh[name].state.consecutive_direction = 0

        decay1 = DecayModel()
        decay2 = DecayModel()
        decay1.decay_drivers(drivers_reinforced)
        decay2.decay_drivers(drivers_fresh)

        # Reinforced should have higher remaining satisfaction
        assert drivers_reinforced["curiosity"].state.satisfied > \
               drivers_fresh["curiosity"].state.satisfied

    def test_consecutive_direction_decays(self):
        """Streaks should decay over time without reinforcement."""
        drivers = create_default_drivers()
        driver = drivers["curiosity"]
        driver.state.satisfied = 0.5
        driver.state.last_triggered = time.time() - 7200  # Longer than half_life * 0.5
        driver.state.consecutive_direction = 10

        self.decay.decay_drivers(drivers)
        assert driver.state.consecutive_direction < 10


class TestMoodCongruence:
    """Test that emotions matching overall mood decay slower."""

    def setup_method(self):
        self.decay = DecayModel()

    def _make_emotion(self, name, intensity=0.8, valence=0.5, arousal=0.5, age_seconds=0):
        return EmotionalState(
            name=name,
            intensity=intensity,
            valence=valence,
            arousal=arousal,
            source_drivers=["curiosity"],
            is_compound=False,
            is_conflict=False,
            description="test",
            timestamp=time.time() - age_seconds,
        )

    def test_congruent_emotion_decays_slower(self):
        """Positive emotion in positive mood should decay slower than negative emotion."""
        age = 2400  # 40 min

        positive_em = self._make_emotion("wonder", valence=0.8, age_seconds=age)
        negative_em = self._make_emotion("restlessness", valence=-0.3, age_seconds=age)

        # Decay both in a positive mood
        result_pos = self.decay.decay_emotions(
            [positive_em], overall_valence=0.6
        )
        result_neg = self.decay.decay_emotions(
            [negative_em], overall_valence=0.6
        )

        if result_pos and result_neg:
            # Positive emotion (congruent with mood) should retain more intensity
            assert result_pos[0].intensity >= result_neg[0].intensity

    def test_no_mood_congruence_at_neutral(self):
        """Near-zero valence should not affect decay."""
        age = 2400
        emotion = self._make_emotion("wonder", valence=0.5, age_seconds=age)

        # With neutral mood vs no mood
        result_neutral = self.decay.decay_emotions(
            [emotion], overall_valence=0.0
        )
        emotion2 = self._make_emotion("wonder", valence=0.5, age_seconds=age)
        result_none = self.decay.decay_emotions(
            [emotion2], overall_valence=None
        )

        if result_neutral and result_none:
            # Should be approximately the same
            diff = abs(result_neutral[0].intensity - result_none[0].intensity)
            assert diff < 0.05


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
        assert len(result) == 0

    def test_high_arousal_decays_faster(self):
        """High-arousal emotions should decay faster than low-arousal."""
        age = 1800
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

    def test_all_profiles_have_consolidation_rate(self):
        """Every profile should have a consolidation_rate."""
        for name, profile in DRIVER_DECAY_PROFILES.items():
            assert hasattr(profile, 'consolidation_rate')
            assert 0.0 <= profile.consolidation_rate <= 1.0

    def test_growth_high_consolidation(self):
        """Growth should consolidate strongly."""
        profile = DRIVER_DECAY_PROFILES["growth"]
        curiosity_profile = DRIVER_DECAY_PROFILES["curiosity"]
        assert profile.consolidation_rate > curiosity_profile.consolidation_rate
