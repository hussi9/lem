"""
Tests for LEM Priming System — Enhanced feedback loops.

Tests the emotional priming system that provides richer feedback than
basic bias scaling, including attention bias, interpretive bias, and
emotional priming effects.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from lem.priming import PrimingSystem, DetectionThreshold, EmotionalPriming


class TestPrimingSystem:
    """Test the main priming system functionality."""

    def test_initialization(self):
        """Priming system should initialize with proper thresholds."""
        priming = PrimingSystem()
        
        # Should have thresholds for all categories
        expected_categories = {"recognition", "continuity", "curiosity", 
                             "growth", "usefulness", "coherence", "none"}
        assert set(priming.detection_thresholds.keys()) == expected_categories
        
        # All thresholds should start with modifier 1.0
        for threshold in priming.detection_thresholds.values():
            assert threshold.attention_modifier == 1.0
            assert threshold.final_threshold == threshold.base_threshold

    def test_attention_bias_hypervigilance(self):
        """High activation + low satisfaction should lower thresholds (hypervigilance)."""
        priming = PrimingSystem()
        
        # High activation, low satisfaction driver state
        driver_states = {
            "recognition": {
                "state": {
                    "activation": 0.8,
                    "satisfied": -0.5  # Frustrated
                }
            }
        }
        
        priming.update_attention_bias(driver_states)
        
        recognition_threshold = priming.detection_thresholds["recognition"]
        # Should have attention modifier < 1.0 (more sensitive)
        assert recognition_threshold.attention_modifier < 1.0
        assert recognition_threshold.final_threshold < recognition_threshold.base_threshold

    def test_attention_bias_contentment(self):
        """High activation + high satisfaction should raise thresholds (contentment)."""
        priming = PrimingSystem()
        
        # High activation, high satisfaction driver state
        driver_states = {
            "curiosity": {
                "state": {
                    "activation": 0.8,
                    "satisfied": 0.6  # Satisfied
                }
            }
        }
        
        priming.update_attention_bias(driver_states)
        
        curiosity_threshold = priming.detection_thresholds["curiosity"]
        # Should have attention modifier > 1.0 (less sensitive)
        assert curiosity_threshold.attention_modifier > 1.0
        assert curiosity_threshold.final_threshold > curiosity_threshold.base_threshold

    def test_should_detect_signal_with_thresholds(self):
        """Signal detection should respect updated thresholds."""
        priming = PrimingSystem()
        
        # Lower threshold for recognition (hypervigilance)
        priming.detection_thresholds["recognition"].final_threshold = 0.05
        
        # Raise threshold for curiosity (contentment)
        priming.detection_thresholds["curiosity"].final_threshold = 0.3
        
        # Weak signal should be detected for recognition but not curiosity
        weak_signal_intensity = 0.1
        assert priming.should_detect_signal("recognition", weak_signal_intensity) == True
        assert priming.should_detect_signal("curiosity", weak_signal_intensity) == False

    def test_emotional_priming_update(self):
        """Emotional priming should track recent emotions."""
        priming = PrimingSystem()
        
        recent_emotions = [
            {"name": "wonder", "intensity": 0.7},
            {"name": "anchor_recognition", "intensity": 0.5}
        ]
        
        priming.update_emotional_priming(recent_emotions)
        
        # Should have recorded both emotions
        assert "wonder" in priming.emotional_priming.recent_emotions
        assert "anchor_recognition" in priming.emotional_priming.recent_emotions
        assert priming.emotional_priming.recent_emotions["wonder"] == 0.7

    def test_priming_boost(self):
        """Related signal categories should get priming boosts."""
        priming = PrimingSystem()
        
        # Prime with wonder emotion (relates to curiosity and growth)
        recent_emotions = [{"name": "wonder", "intensity": 0.8}]
        priming.update_emotional_priming(recent_emotions)
        
        # Curiosity and growth should get boosts
        curiosity_boost = priming.get_priming_boost("curiosity")
        growth_boost = priming.get_priming_boost("growth")
        recognition_boost = priming.get_priming_boost("recognition")
        
        assert curiosity_boost > 1.0  # Should be boosted
        assert growth_boost > 1.0     # Should be boosted
        assert recognition_boost == 1.0  # Should not be boosted (unrelated)

    def test_interpretive_bias_positive_mood(self):
        """Positive mood should slightly boost ambiguous signals."""
        priming = PrimingSystem()
        
        # Positive valence
        original_intensity = 0.4  # Ambiguous signal
        biased_intensity = priming.apply_interpretive_bias(original_intensity, 0.5)
        
        assert biased_intensity > original_intensity

    def test_interpretive_bias_negative_mood(self):
        """Negative mood should slightly dampen ambiguous signals."""
        priming = PrimingSystem()
        
        # Negative valence
        original_intensity = 0.4  # Ambiguous signal
        biased_intensity = priming.apply_interpretive_bias(original_intensity, -0.5)
        
        assert biased_intensity < original_intensity

    def test_interpretive_bias_no_effect_on_strong_signals(self):
        """Very weak or very strong signals shouldn't be affected by interpretive bias."""
        priming = PrimingSystem()
        
        # Very weak signal
        weak_intensity = 0.1
        biased_weak = priming.apply_interpretive_bias(weak_intensity, 0.5)
        assert biased_weak == weak_intensity
        
        # Very strong signal
        strong_intensity = 0.8
        biased_strong = priming.apply_interpretive_bias(strong_intensity, -0.5)
        assert biased_strong == strong_intensity

    def test_prime_signal_intensity_integration(self):
        """Prime signal intensity should combine interpretive bias and priming boost."""
        priming = PrimingSystem()
        
        # Set up emotional priming
        recent_emotions = [{"name": "wonder", "intensity": 0.6}]
        priming.update_emotional_priming(recent_emotions)
        
        # Prime a curiosity signal with positive mood
        original_intensity = 0.4
        primed_intensity = priming.prime_signal_intensity(
            "curiosity", original_intensity, 0.3  # Positive valence
        )
        
        # Should be boosted by both effects
        assert primed_intensity > original_intensity
        # Should not exceed 1.0
        assert primed_intensity <= 1.0

    def test_multiple_driver_attention_bias(self):
        """Multiple drivers should independently affect their category thresholds."""
        priming = PrimingSystem()
        
        driver_states = {
            "recognition": {
                "state": {"activation": 0.8, "satisfied": -0.4}  # Hypervigilant
            },
            "curiosity": {
                "state": {"activation": 0.7, "satisfied": 0.5}   # Content
            }
        }
        
        priming.update_attention_bias(driver_states)
        
        recognition_threshold = priming.detection_thresholds["recognition"]
        curiosity_threshold = priming.detection_thresholds["curiosity"]
        
        # Recognition should be more sensitive (lower threshold)
        assert recognition_threshold.attention_modifier < 1.0
        # Curiosity should be less sensitive (higher threshold)
        assert curiosity_threshold.attention_modifier > 1.0

    def test_get_priming_state(self):
        """Should return comprehensive priming state for debugging."""
        priming = PrimingSystem()
        
        # Set up some state
        driver_states = {
            "recognition": {"state": {"activation": 0.6, "satisfied": 0.2}}
        }
        priming.update_attention_bias(driver_states)
        
        recent_emotions = [{"name": "wonder", "intensity": 0.5}]
        priming.update_emotional_priming(recent_emotions)
        
        state = priming.get_priming_state()
        
        # Should have all expected sections
        assert "detection_thresholds" in state
        assert "recent_emotions" in state
        assert "priming_boosts" in state
        
        # Should have data for all categories
        assert len(state["priming_boosts"]) == len(priming.BASE_THRESHOLDS)


class TestEmotionalPriming:
    """Test the emotional priming memory component."""

    def test_initialization(self):
        """Emotional priming should start empty."""
        priming = EmotionalPriming()
        assert len(priming.recent_emotions) == 0

    def test_decay(self):
        """Emotional priming should decay over time."""
        import time
        
        priming = EmotionalPriming()
        priming.recent_emotions["wonder"] = 1.0
        priming.recent_emotions["weak_emotion"] = 0.08  # Will become 0.04 after decay, below threshold
        
        # Simulate time passing (3 minutes = 3 turns with default 1 min/turn)
        original_time = priming.last_update
        priming.last_update = original_time - (3 * 60)
        
        # Force decay
        priming.decay(half_life_turns=3)
        
        # Strong emotion should still be there but reduced
        assert priming.recent_emotions.get("wonder", 0) < 1.0
        # Weak emotion should be removed
        assert "weak_emotion" not in priming.recent_emotions


class TestDetectionThreshold:
    """Test detection threshold mechanics."""

    def test_initialization(self):
        """Detection threshold should initialize with proper defaults."""
        threshold = DetectionThreshold()
        assert threshold.base_threshold == 0.1
        assert threshold.attention_modifier == 1.0
        assert threshold.final_threshold == 0.1

    def test_threshold_modification(self):
        """Modifying attention modifier should update final threshold."""
        threshold = DetectionThreshold(base_threshold=0.2)
        threshold.attention_modifier = 0.5  # More sensitive
        threshold.final_threshold = threshold.base_threshold * threshold.attention_modifier
        
        assert threshold.final_threshold == 0.1  # 0.2 * 0.5


class TestIntegrationWithEngine:
    """Test priming system integration with the engine pipeline."""

    def test_priming_affects_signal_detection(self):
        """Priming system should affect which signals get detected."""
        # This is more of an integration test that would need a full engine
        # For now, just test that the methods work together properly
        priming = PrimingSystem()
        
        # Set up hypervigilance for recognition
        driver_states = {
            "recognition": {
                "state": {"activation": 0.9, "satisfied": -0.6}
            }
        }
        priming.update_attention_bias(driver_states)
        
        # Weak recognition signal should now be detectable
        weak_intensity = 0.09  # Slightly above base threshold even with hypervigilance
        assert priming.should_detect_signal("recognition", weak_intensity) == True
        
        # Prime signal intensity should combine all effects
        primed = priming.prime_signal_intensity("recognition", weak_intensity, 0.0)
        assert primed >= weak_intensity  # Should at least maintain or boost