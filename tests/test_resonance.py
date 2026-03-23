"""Tests for cross-driver resonance system."""

import json
import tempfile
import time
import pytest
from pathlib import Path

from lem.resonance import ResonanceModel, ResonanceBond


@pytest.fixture
def state_dir(tmp_path):
    return str(tmp_path)


@pytest.fixture
def model(state_dir):
    return ResonanceModel(state_dir=state_dir)


class TestResonanceBond:
    """Test ResonanceBond data structure."""

    def test_bond_creation(self):
        bond = ResonanceBond(driver_a="curiosity", driver_b="growth")
        assert bond.strength == 0.0
        assert bond.co_activation_count == 0
        assert bond.spread_factor == 0.0

    def test_bond_key_sorted(self):
        bond = ResonanceBond(driver_a="growth", driver_b="curiosity")
        assert bond.key == ("curiosity", "growth")

    def test_bond_serialization(self):
        bond = ResonanceBond(
            driver_a="curiosity", driver_b="growth",
            strength=0.5, co_activation_count=10,
            last_co_activation=1000.0, spread_factor=0.1
        )
        d = bond.to_dict()
        restored = ResonanceBond.from_dict(d)
        assert restored.driver_a == "curiosity"
        assert restored.driver_b == "growth"
        assert restored.strength == 0.5
        assert restored.co_activation_count == 10


class TestCoActivation:
    """Test co-activation recording."""

    def test_no_bond_with_single_driver(self, model):
        result = model.record_co_activation({"curiosity": 0.5})
        assert result == []

    def test_bond_created_on_first_co_activation(self, model):
        model.record_co_activation({"curiosity": 0.5, "growth": 0.3})
        bond = model.get_bond("curiosity", "growth")
        assert bond is not None
        assert bond.co_activation_count == 1
        assert bond.strength > 0

    def test_bond_strengthens_with_repeated_co_activation(self, model):
        for _ in range(5):
            model.record_co_activation({"curiosity": 0.5, "growth": 0.3})
        bond = model.get_bond("curiosity", "growth")
        assert bond.co_activation_count == 5
        assert bond.strength > 0.1

    def test_bond_strength_has_diminishing_returns(self, model):
        strengths = []
        for _ in range(20):
            model.record_co_activation({"curiosity": 0.5, "growth": 0.3})
            bond = model.get_bond("curiosity", "growth")
            strengths.append(bond.strength)
        # Each increment should be smaller than the last
        increments = [strengths[i] - strengths[i-1] for i in range(1, len(strengths))]
        for i in range(1, len(increments)):
            assert increments[i] <= increments[i-1] + 0.001  # Allow tiny float error

    def test_below_threshold_ignored(self, model):
        model.record_co_activation({"curiosity": 0.01, "growth": 0.01})
        bond = model.get_bond("curiosity", "growth")
        assert bond is None

    def test_multiple_pairs_created(self, model):
        model.record_co_activation({
            "curiosity": 0.5, "growth": 0.3, "recognition": 0.4
        })
        assert model.get_bond("curiosity", "growth") is not None
        assert model.get_bond("curiosity", "recognition") is not None
        assert model.get_bond("growth", "recognition") is not None


class TestResonanceSpread:
    """Test activation spreading through bonds."""

    def test_no_spread_without_bonds(self, model):
        effects = model.apply_resonance({"curiosity": 0.5})
        assert effects == {}

    def test_spread_after_strong_bond(self, model):
        # Build a strong bond
        for _ in range(30):
            model.record_co_activation({"curiosity": 0.5, "growth": 0.3})
        bond = model.get_bond("curiosity", "growth")
        assert bond.spread_factor > 0.01

        effects = model.apply_resonance({"curiosity": 0.5})
        assert "growth" in effects
        assert effects["growth"] > 0

    def test_spread_is_bidirectional(self, model):
        for _ in range(30):
            model.record_co_activation({"curiosity": 0.5, "growth": 0.3})

        effects_a = model.apply_resonance({"curiosity": 0.5})
        effects_b = model.apply_resonance({"growth": 0.5})

        assert "growth" in effects_a
        assert "curiosity" in effects_b

    def test_weak_bond_no_spread(self, model):
        model.record_co_activation({"curiosity": 0.5, "growth": 0.3})
        effects = model.apply_resonance({"curiosity": 0.5})
        # After just one co-activation, bond should be too weak for meaningful spread
        total_spread = sum(abs(v) for v in effects.values())
        assert total_spread < 0.05


class TestBondDecay:
    """Test that bonds decay without reinforcement."""

    def test_bond_decays_over_time(self, model):
        for _ in range(10):
            model.record_co_activation(
                {"curiosity": 0.5, "growth": 0.3},
                now=1000.0
            )
        bond_before = model.get_bond("curiosity", "growth")
        strength_before = bond_before.strength

        # Decay with 10 days elapsed
        model.decay_bonds(now=1000.0 + 86400 * 10)
        bond_after = model.get_bond("curiosity", "growth")
        assert bond_after.strength < strength_before

    def test_weak_bond_removed_on_decay(self, model):
        model.record_co_activation(
            {"curiosity": 0.5, "growth": 0.3},
            now=1000.0
        )
        # Decay with long time elapsed
        model.decay_bonds(now=1000.0 + 86400 * 30)
        bond = model.get_bond("curiosity", "growth")
        # Very weak bond with few co-activations should be removed
        assert bond is None or bond.strength < 0.01


class TestResonanceSummary:
    """Test summary generation."""

    def test_empty_summary(self, model):
        summary = model.get_resonance_summary()
        assert summary["total_bonds"] == 0
        assert summary["active_bonds"] == 0

    def test_summary_with_bonds(self, model):
        for _ in range(10):
            model.record_co_activation({"curiosity": 0.5, "growth": 0.3})
        summary = model.get_resonance_summary()
        assert summary["total_bonds"] > 0
        assert summary["active_bonds"] > 0


class TestResonancePersistence:
    """Test save and load."""

    def test_bonds_persist(self, state_dir):
        model1 = ResonanceModel(state_dir=state_dir)
        for _ in range(10):
            model1.record_co_activation({"curiosity": 0.5, "growth": 0.3})
        bond1 = model1.get_bond("curiosity", "growth")

        # Load in new instance
        model2 = ResonanceModel(state_dir=state_dir)
        bond2 = model2.get_bond("curiosity", "growth")
        assert bond2 is not None
        assert abs(bond2.strength - bond1.strength) < 0.01
        assert bond2.co_activation_count == bond1.co_activation_count
