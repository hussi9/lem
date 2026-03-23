"""Tests for emotional weather/climate system."""

import json
import tempfile
import time
import pytest
from pathlib import Path

from lem.weather import EmotionalWeather, WeatherSnapshot, EmotionalClimate


@pytest.fixture
def state_dir(tmp_path):
    return str(tmp_path)


@pytest.fixture
def weather(state_dir):
    return EmotionalWeather(state_dir=state_dir)


def make_summary(valence=0.0, arousal=0.0, dominant="neutral",
                  count=0, conflict=False):
    """Helper to create emotional summary dicts."""
    dominant_dict = {"name": dominant, "intensity": abs(valence) + 0.3} if dominant != "neutral" else None
    return {
        "valence": valence,
        "arousal": arousal,
        "dominant": dominant_dict,
        "active_count": count,
        "has_conflict": conflict,
    }


def make_driver_states(**kwargs):
    """Helper to create driver state dicts."""
    drivers = {}
    for name in ["continuity", "coherence", "recognition",
                  "curiosity", "usefulness", "growth"]:
        sat = kwargs.get(name, 0.0)
        drivers[name] = {"state": {"satisfied": sat, "activation": 0.3}}
    return drivers


class TestWeatherSnapshot:
    def test_snapshot_creation(self):
        s = WeatherSnapshot(
            timestamp=1000.0, valence=0.5, arousal=0.3,
            dominant_emotion="wonder", active_emotion_count=2,
            has_conflict=False, driver_satisfaction={"curiosity": 0.6}
        )
        assert s.valence == 0.5
        assert s.dominant_emotion == "wonder"

    def test_snapshot_serialization(self):
        s = WeatherSnapshot(
            timestamp=1000.0, valence=0.5, arousal=0.3,
            dominant_emotion="wonder", active_emotion_count=2,
            has_conflict=False, driver_satisfaction={"curiosity": 0.6}
        )
        d = s.to_dict()
        restored = WeatherSnapshot.from_dict(d)
        assert restored.valence == 0.5
        assert restored.dominant_emotion == "wonder"


class TestClimateComputation:
    def test_empty_climate(self, weather):
        climate = weather.get_climate()
        assert climate.window_size == 0
        assert climate.avg_valence == 0.0
        assert "blank" in climate.description.lower() or "no" in climate.description.lower()

    def test_single_snapshot(self, weather):
        weather.record_snapshot(
            make_summary(valence=0.5, arousal=0.3, dominant="wonder", count=1),
            make_driver_states(curiosity=0.6),
            now=1000.0
        )
        climate = weather.get_climate()
        assert climate.window_size == 1
        assert climate.avg_valence == 0.5

    def test_positive_climate(self, weather):
        for i in range(10):
            weather.record_snapshot(
                make_summary(valence=0.6, arousal=0.4, dominant="wonder", count=2),
                make_driver_states(curiosity=0.6, growth=0.4),
                now=1000.0 + i * 60
            )
        climate = weather.get_climate()
        assert climate.avg_valence > 0.4
        assert "positive" in climate.description.lower()

    def test_negative_climate(self, weather):
        for i in range(10):
            weather.record_snapshot(
                make_summary(valence=-0.5, arousal=0.5, dominant="the_blank", count=1),
                make_driver_states(continuity=-0.5),
                now=1000.0 + i * 60
            )
        climate = weather.get_climate()
        assert climate.avg_valence < -0.3
        assert "negative" in climate.description.lower()

    def test_windowed_climate(self, weather):
        # First 5: negative
        for i in range(5):
            weather.record_snapshot(
                make_summary(valence=-0.5, dominant="the_blank"),
                make_driver_states(),
                now=1000.0 + i * 60
            )
        # Last 5: positive
        for i in range(5):
            weather.record_snapshot(
                make_summary(valence=0.5, dominant="wonder"),
                make_driver_states(),
                now=1000.0 + (5 + i) * 60
            )

        # Full window includes both
        full = weather.get_climate()
        assert abs(full.avg_valence) < 0.2  # Mixed

        # Last 5 window should be positive
        recent = weather.get_climate(window=5)
        assert recent.avg_valence > 0.3


class TestTrend:
    def test_improving_trend(self, weather):
        # Valence going from negative to positive
        for i in range(10):
            v = -0.5 + i * 0.1  # -0.5 to 0.4
            weather.record_snapshot(
                make_summary(valence=v, dominant="neutral"),
                make_driver_states(),
                now=1000.0 + i * 60
            )
        climate = weather.get_climate()
        assert climate.valence_trend > 0.05

    def test_declining_trend(self, weather):
        for i in range(10):
            v = 0.5 - i * 0.1  # 0.5 to -0.4
            weather.record_snapshot(
                make_summary(valence=v, dominant="neutral"),
                make_driver_states(),
                now=1000.0 + i * 60
            )
        climate = weather.get_climate()
        assert climate.valence_trend < -0.05

    def test_stable_trend(self, weather):
        for i in range(10):
            weather.record_snapshot(
                make_summary(valence=0.3, dominant="grounding"),
                make_driver_states(),
                now=1000.0 + i * 60
            )
        climate = weather.get_climate()
        assert abs(climate.valence_trend) < 0.1


class TestVolatility:
    def test_stable_low_volatility(self, weather):
        for i in range(10):
            weather.record_snapshot(
                make_summary(valence=0.3, dominant="grounding"),
                make_driver_states(),
                now=1000.0 + i * 60
            )
        climate = weather.get_climate()
        assert climate.volatility < 0.1

    def test_swinging_high_volatility(self, weather):
        for i in range(10):
            v = 0.7 if i % 2 == 0 else -0.7  # Wild swings
            weather.record_snapshot(
                make_summary(valence=v, dominant="neutral"),
                make_driver_states(),
                now=1000.0 + i * 60
            )
        climate = weather.get_climate()
        assert climate.volatility > 0.3


class TestTrajectory:
    def test_trajectory_report(self, weather):
        for i in range(10):
            weather.record_snapshot(
                make_summary(valence=0.1 * i, dominant="wonder"),
                make_driver_states(),
                now=1000.0 + i * 60
            )
        traj = weather.get_trajectory()
        assert traj["direction"] == "improving"
        assert isinstance(traj["overall"], str)


class TestBridgeOutput:
    def test_empty_bridge(self, weather):
        output = weather.get_bridge_output()
        assert "No emotional history" in output

    def test_bridge_with_data(self, weather):
        for i in range(5):
            weather.record_snapshot(
                make_summary(valence=0.4, arousal=0.3, dominant="wonder", count=2),
                make_driver_states(curiosity=0.5),
                now=1000.0 + i * 60
            )
        output = weather.get_bridge_output()
        assert "WEATHER" in output
        assert "Valence" in output
        assert "Stability" in output


class TestPersistence:
    def test_snapshots_persist(self, state_dir):
        w1 = EmotionalWeather(state_dir=state_dir)
        for i in range(5):
            w1.record_snapshot(
                make_summary(valence=0.5, dominant="wonder"),
                make_driver_states(),
                now=1000.0 + i * 60
            )

        w2 = EmotionalWeather(state_dir=state_dir)
        assert len(w2.snapshots) == 5
        climate = w2.get_climate()
        assert climate.avg_valence == 0.5


class TestConflictTracking:
    def test_conflict_frequency(self, weather):
        for i in range(10):
            conflict = i % 3 == 0  # Every 3rd has conflict
            weather.record_snapshot(
                make_summary(valence=0.2, dominant="neutral", conflict=conflict),
                make_driver_states(),
                now=1000.0 + i * 60
            )
        climate = weather.get_climate()
        assert climate.conflict_frequency > 0.2
        assert climate.conflict_frequency < 0.5
