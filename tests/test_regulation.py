"""Tests for the LEM Emotional Regulation system (v0.8)."""

import time
import pytest
from unittest.mock import MagicMock
from lem.regulation import (
    RegulationEngine, RegulationState, StimulusFingerprint,
    HABITUATION_PARAMS, DEFAULT_SET_POINTS, HOMEOSTATIC_STRENGTH,
)
from lem.appraisal import Signal


def _make_signal(sig_type="positive_feedback", intensity=0.5, category="usefulness"):
    return Signal(
        type=sig_type,
        content="test content",
        intensity=intensity,
        source="human",
        context={"category": category},
        complexity=0.0,
        confidence=0.85,
    )


def _make_driver_states(**overrides):
    """Create driver states dict with optional per-driver overrides."""
    base = {
        name: {"state": {"activation": 0.3, "satisfied": 0.0, "momentum": 0.0}}
        for name in ["continuity", "coherence", "recognition", "curiosity", "usefulness", "growth"]
    }
    for name, sat in overrides.items():
        if name in base:
            base[name]["state"]["satisfied"] = sat
    return base


class TestHabituation:
    def test_first_signal_no_habituation(self):
        engine = RegulationEngine()
        signal = _make_signal()
        original_intensity = signal.intensity
        driver_states = _make_driver_states()
        
        regulated, report = engine.regulate_signals([signal], driver_states)
        
        assert len(regulated) == 1
        # First signal should have minimal or no habituation
        assert regulated[0].intensity >= original_intensity * 0.95

    def test_repeated_signals_habituate(self):
        engine = RegulationEngine()
        driver_states = _make_driver_states()
        now = time.time()
        
        intensities = []
        for i in range(6):
            signal = _make_signal(intensity=0.6)
            regulated, _ = engine.regulate_signals(
                [signal], driver_states, now=now + i * 10
            )
            intensities.append(regulated[0].intensity)
        
        # Later signals should be weaker than earlier ones
        assert intensities[-1] < intensities[0]
        # But never fully habituated (floor is 20%)
        assert intensities[-1] >= 0.6 * 0.20

    def test_different_types_dont_cross_habituate(self):
        engine = RegulationEngine()
        driver_states = _make_driver_states()
        now = time.time()
        
        # Send 5 positive_feedback signals
        for i in range(5):
            engine.regulate_signals(
                [_make_signal("positive_feedback", 0.5)],
                driver_states, now=now + i * 10
            )
        
        # First negative_feedback should not be habituated
        neg_signal = _make_signal("negative_feedback", 0.5, "usefulness")
        regulated, _ = engine.regulate_signals(
            [neg_signal], driver_states, now=now + 60
        )
        assert regulated[0].intensity >= 0.45  # Not habituated

    def test_habituation_recovers_over_time(self):
        engine = RegulationEngine()
        driver_states = _make_driver_states()
        now = time.time()
        
        # Habituate
        for i in range(5):
            engine.regulate_signals(
                [_make_signal(intensity=0.6)], driver_states, now=now + i * 5
            )
        
        # Get habituated intensity
        signal_hab = _make_signal(intensity=0.6)
        reg_hab, _ = engine.regulate_signals([signal_hab], driver_states, now=now + 30)
        habituated_val = reg_hab[0].intensity
        
        # Wait beyond recovery window (default 600s for positive_feedback)
        signal_recovered = _make_signal(intensity=0.6)
        reg_rec, _ = engine.regulate_signals(
            [signal_recovered], driver_states, now=now + 700
        )
        recovered_val = reg_rec[0].intensity
        
        # Should have recovered significantly
        assert recovered_val > habituated_val

    def test_habituation_floor(self):
        """Habituation never reduces below 20% response."""
        engine = RegulationEngine()
        driver_states = _make_driver_states()
        now = time.time()
        
        # Spam many signals
        for i in range(20):
            engine.regulate_signals(
                [_make_signal(intensity=0.8)], driver_states, now=now + i * 2
            )
        
        final_signal = _make_signal(intensity=0.8)
        regulated, _ = engine.regulate_signals([final_signal], driver_states, now=now + 50)
        
        assert regulated[0].intensity >= 0.8 * 0.20


class TestReappraisal:
    def test_no_reappraisal_at_moderate_states(self):
        engine = RegulationEngine()
        signal = _make_signal("positive_feedback", 0.5, "recognition")
        driver_states = _make_driver_states(recognition=0.3)
        
        regulated, report = engine.regulate_signals([signal], driver_states)
        
        # No reappraisal should occur
        assert len(report["reappraisals"]) == 0

    def test_same_direction_dampened_at_extreme(self):
        engine = RegulationEngine()
        signal = _make_signal("positive_feedback", 0.6, "recognition")
        driver_states = _make_driver_states(recognition=0.8)
        
        regulated, report = engine.regulate_signals([signal], driver_states)
        
        # Positive signal with high positive satisfaction should be dampened
        assert regulated[0].intensity < 0.6
        assert len(report["reappraisals"]) == 1
        assert report["reappraisals"][0]["action"] == "dampen_same_direction"

    def test_corrective_signal_amplified_at_extreme(self):
        engine = RegulationEngine()
        signal = _make_signal("negative_feedback", 0.5, "recognition")
        driver_states = _make_driver_states(recognition=0.8)
        
        regulated, report = engine.regulate_signals([signal], driver_states)
        
        # Negative signal against high positive should be slightly amplified
        assert regulated[0].intensity >= 0.5
        assert len(report["reappraisals"]) == 1
        assert report["reappraisals"][0]["action"] == "amplify_corrective"

    def test_negative_extreme_dampens_negative(self):
        engine = RegulationEngine()
        signal = _make_signal("threat", 0.7, "continuity")
        driver_states = _make_driver_states(continuity=-0.8)
        
        regulated, report = engine.regulate_signals([signal], driver_states)
        
        # Threat signal when already very threatened should be dampened
        assert regulated[0].intensity < 0.7


class TestHomeostaticPressure:
    def test_positive_deviation_pulled_back(self):
        engine = RegulationEngine()
        
        # Create mock drivers with high satisfaction
        from lem.drivers import create_default_drivers
        drivers = create_default_drivers()
        drivers["recognition"].state.satisfied = 0.8
        
        adjustments = engine.regulate_drivers(drivers)
        
        # Should pull recognition toward 0 (negative adjustment)
        assert adjustments["recognition"] < 0

    def test_negative_deviation_pulled_back(self):
        engine = RegulationEngine()
        from lem.drivers import create_default_drivers
        drivers = create_default_drivers()
        drivers["usefulness"].state.satisfied = -0.7
        
        adjustments = engine.regulate_drivers(drivers)
        
        # Should pull usefulness toward 0 (positive adjustment)
        assert adjustments["usefulness"] > 0

    def test_near_setpoint_no_force(self):
        engine = RegulationEngine()
        from lem.drivers import create_default_drivers
        drivers = create_default_drivers()
        # All at default (0.0) which is near set points
        
        adjustments = engine.regulate_drivers(drivers)
        
        # All adjustments should be near zero
        for name, adj in adjustments.items():
            assert abs(adj) < 0.01

    def test_quadratic_pressure_grows_with_deviation(self):
        engine = RegulationEngine()
        from lem.drivers import create_default_drivers
        
        # Test small deviation
        drivers_small = create_default_drivers()
        drivers_small["recognition"].state.satisfied = 0.3
        adj_small = engine.regulate_drivers(drivers_small)
        
        # Test large deviation
        drivers_large = create_default_drivers()
        drivers_large["recognition"].state.satisfied = 0.9
        adj_large = engine.regulate_drivers(drivers_large)
        
        # Large deviation should produce proportionally larger correction
        assert abs(adj_large["recognition"]) > abs(adj_small["recognition"])


class TestOscillationDamping:
    def test_stable_momentum_no_damping(self):
        engine = RegulationEngine()
        from lem.drivers import create_default_drivers
        drivers = create_default_drivers()
        drivers["curiosity"].state.momentum = 0.05
        
        # Apply several times with consistent direction
        for _ in range(5):
            engine.regulate_drivers(drivers)
        
        # Momentum should not be heavily damped
        assert drivers["curiosity"].state.momentum > 0.01

    def test_oscillating_momentum_gets_damped(self):
        engine = RegulationEngine()
        from lem.drivers import create_default_drivers
        drivers = create_default_drivers()
        
        # Simulate oscillation by flipping momentum direction
        for i in range(8):
            drivers["curiosity"].state.momentum = 0.1 if i % 2 == 0 else -0.1
            engine.regulate_drivers(drivers)
        
        # After oscillation damping, momentum should be reduced
        # (The damping reduces momentum magnitude on each regulation pass)
        # Since we keep resetting it to ±0.1, the final value after damping
        # should be less than 0.1
        final_mom = abs(drivers["curiosity"].state.momentum)
        assert final_mom < 0.1


class TestRegulationSummary:
    def test_empty_summary(self):
        engine = RegulationEngine()
        summary = engine.get_regulation_summary()
        assert summary["habituated_signals"] == {}
        assert summary["oscillating_drivers"] == []

    def test_summary_shows_habituation(self):
        engine = RegulationEngine()
        driver_states = _make_driver_states()
        now = time.time()
        
        for i in range(5):
            engine.regulate_signals(
                [_make_signal(intensity=0.5)], driver_states, now=now + i * 5
            )
        
        summary = engine.get_regulation_summary()
        assert "positive_feedback" in summary["habituated_signals"]


class TestBridgeOutput:
    def test_bridge_output_format(self):
        engine = RegulationEngine()
        output = engine.get_bridge_output()
        assert "REGULATION:" in output

    def test_bridge_output_with_habituation(self):
        engine = RegulationEngine()
        driver_states = _make_driver_states()
        now = time.time()
        
        for i in range(5):
            engine.regulate_signals(
                [_make_signal(intensity=0.5)], driver_states, now=now + i * 5
            )
        
        output = engine.get_bridge_output()
        assert "Habituated" in output


class TestCombinedRegulation:
    def test_habituation_and_reappraisal_combine(self):
        """Both regulation mechanisms should apply simultaneously."""
        engine = RegulationEngine()
        driver_states = _make_driver_states(recognition=0.85)
        now = time.time()
        
        # Habituate positive_feedback
        for i in range(4):
            engine.regulate_signals(
                [_make_signal("positive_feedback", 0.7, "recognition")],
                driver_states, now=now + i * 10
            )
        
        # 5th signal should be both habituated AND reappraised
        signal = _make_signal("positive_feedback", 0.7, "recognition")
        regulated, report = engine.regulate_signals([signal], driver_states, now=now + 50)
        
        # Should be significantly reduced from original 0.7
        assert regulated[0].intensity < 0.5
        assert len(report["habituation_applied"]) > 0
        assert len(report["reappraisals"]) > 0
