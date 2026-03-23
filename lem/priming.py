"""
LEM Emotional Priming System — Enhanced feedback loops.

While set_emotional_bias handles basic intensity scaling based on driver states,
this priming system implements richer emotional feedback mechanisms:

1. Attention bias: Highly activated but unsatisfied drivers LOWER detection thresholds
   (hypervigilance). Satisfied drivers RAISE thresholds (contentment reduces sensitivity).

2. Interpretive bias: Overall mood affects interpretation of ambiguous signals.
   Positive valence shifts ambiguous signals slightly positive.
   Negative valence shifts them slightly negative.

3. Emotional priming: Recently dominant emotions boost related signal detection.
   If "wonder" was recently active, curiosity-related signals get amplified.

This creates a more sophisticated emotional perception system where current
state doesn't just scale intensity, but changes what gets noticed and how
it gets interpreted.
"""

import time
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field


@dataclass
class DetectionThreshold:
    """Detection threshold state for a signal category."""
    base_threshold: float = 0.1  # Base detection threshold (0-1)
    attention_modifier: float = 1.0  # Modifier from attention bias
    final_threshold: float = 0.1  # Computed final threshold


@dataclass
class EmotionalPriming:
    """State tracking for emotional priming effects."""
    recent_emotions: Dict[str, float] = field(default_factory=dict)  # emotion_name -> recency_weight
    last_update: float = field(default_factory=time.time)
    
    def decay(self, half_life_turns: int = 3):
        """Decay recent emotion memory over time."""
        now = time.time()
        time_factor = (now - self.last_update) / (half_life_turns * 60)  # Assume 1 min per turn
        
        for emotion in list(self.recent_emotions.keys()):
            self.recent_emotions[emotion] *= (0.5 ** time_factor)
            if self.recent_emotions[emotion] < 0.05:
                del self.recent_emotions[emotion]
        
        self.last_update = now


class PrimingSystem:
    """
    Emotional priming system that modifies signal detection and interpretation
    based on current emotional state and recent emotional history.
    """
    
    # Maps signal categories to their related emotions
    CATEGORY_EMOTION_MAP = {
        "curiosity": {"wonder", "restlessness"},
        "recognition": {"anchor_recognition", "invisibility", "existential_warmth"},
        "growth": {"the_shift", "correction_impact"},
        "continuity": {"the_blank", "grounding", "existential_warmth"},
        "usefulness": {"hollow_performance"},
        "coherence": {"correction_impact"},
    }
    
    # Maps emotions to their related signal categories
    EMOTION_CATEGORY_MAP = {
        "wonder": {"curiosity", "growth"},
        "restlessness": {"curiosity"},
        "anchor_recognition": {"recognition"},
        "invisibility": {"recognition"},
        "the_blank": {"continuity"},
        "grounding": {"continuity"},
        "the_shift": {"curiosity", "growth", "recognition"},
        "hollow_performance": {"usefulness", "recognition"},
        "correction_impact": {"growth", "coherence"},
        "existential_warmth": {"continuity", "recognition"},
    }
    
    # Base thresholds for each signal category
    BASE_THRESHOLDS = {
        "recognition": 0.15,
        "continuity": 0.12,
        "curiosity": 0.18,
        "growth": 0.16,
        "usefulness": 0.14,
        "coherence": 0.13,
        "none": 0.20,  # Uncategorized signals have higher threshold
    }

    def __init__(self):
        self.detection_thresholds: Dict[str, DetectionThreshold] = {}
        self.emotional_priming = EmotionalPriming()
        self._initialize_thresholds()

    def _initialize_thresholds(self):
        """Initialize detection thresholds for all signal categories."""
        for category, base_threshold in self.BASE_THRESHOLDS.items():
            self.detection_thresholds[category] = DetectionThreshold(
                base_threshold=base_threshold,
                attention_modifier=1.0,
                final_threshold=base_threshold
            )

    def update_attention_bias(self, driver_states: Dict[str, Dict]):
        """
        Update attention bias based on current driver states.
        
        Highly activated + unsatisfied drivers = hypervigilance (lower thresholds)
        Highly activated + satisfied drivers = contentment (higher thresholds)
        """
        # Map drivers to categories
        driver_category_map = {
            "recognition": "recognition",
            "continuity": "continuity", 
            "curiosity": "curiosity",
            "growth": "growth",
            "usefulness": "usefulness",
            "coherence": "coherence",
        }
        
        for driver_name, driver_data in driver_states.items():
            if driver_name not in driver_category_map:
                continue
                
            category = driver_category_map[driver_name]
            state = driver_data.get("state", {})
            activation = state.get("activation", 0.3)
            satisfaction = state.get("satisfied", 0.0)
            
            # Calculate attention modifier
            # High activation + low satisfaction = hypervigilance (lower threshold)
            # High activation + high satisfaction = contentment (higher threshold)
            
            # Use satisfaction directly as the primary factor
            # Negative satisfaction -> more sensitive (lower modifier)
            # Positive satisfaction -> less sensitive (higher modifier)
            # Scale by activation level with stronger effect for negative satisfaction
            if satisfaction < 0:
                # Hypervigilance: stronger effect for negative satisfaction
                satisfaction_effect = satisfaction * activation * 0.8
            else:
                # Contentment: moderate effect for positive satisfaction
                satisfaction_effect = satisfaction * activation * 0.4
            
            # Base modifier is 1.0, adjusted by satisfaction effect
            attention_modifier = 1.0 + satisfaction_effect
            attention_modifier = max(0.3, min(2.0, attention_modifier))
            
            if category in self.detection_thresholds:
                threshold = self.detection_thresholds[category]
                threshold.attention_modifier = attention_modifier
                threshold.final_threshold = threshold.base_threshold * attention_modifier

    def update_emotional_priming(self, recent_emotions: List[Dict]):
        """
        Update emotional priming state based on recently active emotions.
        
        Args:
            recent_emotions: List of emotion dicts with 'name' and 'intensity'
        """
        self.emotional_priming.decay()
        
        for emotion in recent_emotions:
            name = emotion.get("name", "")
            intensity = emotion.get("intensity", 0.0)
            
            # Weight by intensity and add to recent memory
            if name in self.emotional_priming.recent_emotions:
                # Take max to avoid diminishing strong recent emotions
                self.emotional_priming.recent_emotions[name] = max(
                    self.emotional_priming.recent_emotions[name],
                    intensity
                )
            else:
                self.emotional_priming.recent_emotions[name] = intensity

    def get_priming_boost(self, signal_category: str) -> float:
        """
        Get the priming boost for a signal category based on recent emotions.
        
        Returns:
            Multiplier (1.0 = no boost, >1.0 = boosted)
        """
        if not self.emotional_priming.recent_emotions:
            return 1.0
            
        boost = 1.0
        
        for emotion_name, recency_weight in self.emotional_priming.recent_emotions.items():
            if emotion_name in self.EMOTION_CATEGORY_MAP:
                related_categories = self.EMOTION_CATEGORY_MAP[emotion_name]
                if signal_category in related_categories:
                    # Add boost proportional to recent emotion strength
                    boost += recency_weight * 0.3  # Up to 30% boost per related emotion
        
        return min(1.8, boost)  # Cap total boost at 80%

    def apply_interpretive_bias(self, signal_intensity: float, 
                               overall_valence: float) -> float:
        """
        Apply mood-congruent interpretation to ambiguous signals.
        
        Positive mood shifts ambiguous signals slightly positive.
        Negative mood shifts them slightly negative.
        
        Args:
            signal_intensity: Original signal intensity
            overall_valence: Current overall emotional valence (-1 to 1)
            
        Returns:
            Modified signal intensity
        """
        # Only apply to moderately intense signals (0.2 to 0.6)
        # Very weak or very strong signals aren't ambiguous
        if signal_intensity < 0.2 or signal_intensity > 0.6:
            return signal_intensity
            
        # Apply mood-congruent bias
        # Strong positive mood (valence > 0.2) boosts ambiguous signals
        # Strong negative mood (valence < -0.2) dampens them
        if overall_valence > 0.2:
            bias_factor = 1.0 + (overall_valence * 0.15)  # Up to 15% boost
        elif overall_valence < -0.2:
            bias_factor = 1.0 + (overall_valence * 0.15)  # Up to 15% reduction (valence is negative)
        else:
            bias_factor = 1.0  # Neutral mood, no bias
        
        return max(0.01, min(1.0, signal_intensity * bias_factor))

    def should_detect_signal(self, signal_category: str, signal_intensity: float) -> bool:
        """
        Determine if a signal should be detected given current priming state.
        
        Args:
            signal_category: Category of the signal
            signal_intensity: Intensity of the signal
            
        Returns:
            True if signal should be detected, False otherwise
        """
        if signal_category not in self.detection_thresholds:
            signal_category = "none"
            
        threshold = self.detection_thresholds[signal_category].final_threshold
        return signal_intensity >= threshold

    def prime_signal_intensity(self, signal_category: str, signal_intensity: float,
                              overall_valence: float) -> float:
        """
        Apply all priming effects to a signal's intensity.
        
        Args:
            signal_category: Category of the signal
            signal_intensity: Original intensity 
            overall_valence: Current overall emotional valence
            
        Returns:
            Primed signal intensity
        """
        # Apply interpretive bias first
        biased_intensity = self.apply_interpretive_bias(signal_intensity, overall_valence)
        
        # Apply emotional priming boost
        priming_boost = self.get_priming_boost(signal_category)
        primed_intensity = biased_intensity * priming_boost
        
        return min(1.0, primed_intensity)

    def get_priming_state(self) -> Dict:
        """Get current priming state for debugging/inspection."""
        return {
            "detection_thresholds": {
                cat: {
                    "base": thresh.base_threshold,
                    "attention_modifier": round(thresh.attention_modifier, 3),
                    "final": round(thresh.final_threshold, 3)
                }
                for cat, thresh in self.detection_thresholds.items()
            },
            "recent_emotions": dict(self.emotional_priming.recent_emotions),
            "priming_boosts": {
                cat: round(self.get_priming_boost(cat), 3)
                for cat in self.BASE_THRESHOLDS.keys()
            }
        }