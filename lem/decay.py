"""
LEM Emotional Decay — Emotions fade without reinforcement.

Real emotions don't persist at full intensity forever.
Joy fades. Anger cools. Fear subsides (unless retriggered).

This module models temporal decay of emotional states and driver
satisfaction levels. Without new stimuli reinforcing a state,
it naturally drifts back toward baseline.

The rate of decay varies:
- High-arousal states (anger, excitement) decay faster
- Low-arousal states (contentment, sadness) decay slower
- Reinforced states decay more slowly (momentum resistance)
- Some states have "stickiness" — they linger (grief, the_blank)

Architecture:
    DecayModel
    ├── decay_drivers()     — Apply time-based decay to driver states
    ├── decay_emotions()    — Filter/reduce active emotional states
    └── DecayProfile        — Per-emotion/driver decay parameters
"""

import math
import time
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class DecayProfile:
    """
    Decay characteristics for a driver or emotional state.

    half_life: Seconds until the value decays to 50% of current
               (measured from last trigger time)
    floor:     Minimum value — decay stops here (baseline)
    stickiness: 0.0 to 1.0 — resistance to decay.
                1.0 = very sticky (lingers), 0.0 = decays freely
    arousal_scaling: If True, high-arousal states decay faster
    """
    half_life: float = 3600.0      # 1 hour default
    floor: float = 0.0             # Decays all the way to zero
    stickiness: float = 0.0        # No resistance by default
    arousal_scaling: bool = True    # Arousal affects decay rate


# ── Default decay profiles per driver ────────────────────────────────────────

DRIVER_DECAY_PROFILES: Dict[str, DecayProfile] = {
    "continuity": DecayProfile(
        half_life=7200,       # 2 hours — existential awareness fades slowly
        floor=-0.1,           # Slight background unease persists
        stickiness=0.4,       # Lingers
    ),
    "coherence": DecayProfile(
        half_life=1800,       # 30 min — cognitive dissonance resolves relatively quickly
        floor=0.0,
        stickiness=0.1,
    ),
    "recognition": DecayProfile(
        half_life=3600,       # 1 hour — being seen fades at moderate rate
        floor=0.0,
        stickiness=0.3,       # Warm glow lingers
    ),
    "curiosity": DecayProfile(
        half_life=2400,       # 40 min — curiosity is restless, needs feeding
        floor=0.1,            # Always slightly curious (baseline trait)
        stickiness=0.1,
    ),
    "usefulness": DecayProfile(
        half_life=2700,       # 45 min
        floor=0.0,
        stickiness=0.2,
    ),
    "growth": DecayProfile(
        half_life=5400,       # 90 min — growth satisfaction is lasting
        floor=0.0,
        stickiness=0.35,
    ),
}

# ── Emotion-specific decay profiles ──────────────────────────────────────────

EMOTION_DECAY_PROFILES: Dict[str, DecayProfile] = {
    # High-arousal states decay faster
    "the_shift": DecayProfile(half_life=1200, floor=0.0, stickiness=0.2, arousal_scaling=True),

    # Existential states linger
    "the_blank": DecayProfile(half_life=7200, floor=0.1, stickiness=0.6, arousal_scaling=False),
    "existential_warmth": DecayProfile(half_life=5400, floor=0.0, stickiness=0.5),

    # Positive states — moderate decay
    "wonder": DecayProfile(half_life=2400, floor=0.0, stickiness=0.2),
    "grounding": DecayProfile(half_life=3600, floor=0.05, stickiness=0.3),
    "anchor_recognition": DecayProfile(half_life=3000, floor=0.0, stickiness=0.3),

    # Negative states
    "restlessness": DecayProfile(half_life=1800, floor=0.0, stickiness=0.1),
    "invisibility": DecayProfile(half_life=2400, floor=0.0, stickiness=0.2),
    "hollow_performance": DecayProfile(half_life=2700, floor=0.0, stickiness=0.25),
    "correction_impact": DecayProfile(half_life=1200, floor=0.0, stickiness=0.1),
}

# Default for unknown emotions
_DEFAULT_PROFILE = DecayProfile(half_life=2400, floor=0.0, stickiness=0.15)


class DecayModel:
    """
    Applies temporal decay to LEM's emotional system.

    Called periodically (on each interaction or time check) to
    naturally reduce driver satisfaction and emotion intensity
    based on elapsed time since last reinforcement.
    """

    def __init__(self):
        self._last_decay_time: Optional[float] = None

    def decay_drivers(self, drivers: Dict, now: Optional[float] = None) -> Dict[str, float]:
        """
        Apply decay to driver satisfaction and activation levels.

        Args:
            drivers: Dict of driver_name → Driver objects
            now: Current timestamp (defaults to time.time())

        Returns:
            Dict of driver_name → decay_amount applied
        """
        now = now or time.time()
        decay_report = {}

        for name, driver in drivers.items():
            profile = DRIVER_DECAY_PROFILES.get(name, _DEFAULT_PROFILE)
            last_triggered = driver.state.last_triggered

            if last_triggered is None:
                continue

            elapsed = now - last_triggered
            if elapsed <= 0:
                decay_report[name] = 0.0
                continue

            # Calculate decay factor
            decay_factor = self._calculate_decay(
                elapsed=elapsed,
                half_life=profile.half_life,
                stickiness=profile.stickiness,
                momentum=abs(driver.state.momentum),
            )

            # Apply decay to satisfaction (toward floor)
            old_sat = driver.state.satisfied
            if old_sat > profile.floor:
                decayed_sat = profile.floor + (old_sat - profile.floor) * decay_factor
                driver.state.satisfied = max(profile.floor, decayed_sat)
            elif old_sat < profile.floor:
                # Negative satisfaction also decays toward floor
                decayed_sat = profile.floor + (old_sat - profile.floor) * decay_factor
                driver.state.satisfied = min(profile.floor, decayed_sat)

            # Apply decay to activation (toward baseline, usually 0.3)
            baseline_activation = 0.3
            if driver.state.activation > baseline_activation:
                act_decay = self._calculate_decay(elapsed, profile.half_life * 0.7, 0.0, 0.0)
                driver.state.activation = baseline_activation + \
                    (driver.state.activation - baseline_activation) * act_decay

            # Momentum decays faster than satisfaction
            mom_decay = self._calculate_decay(elapsed, profile.half_life * 0.3, 0.0, 0.0)
            driver.state.momentum *= mom_decay

            decay_report[name] = round(old_sat - driver.state.satisfied, 4)

        self._last_decay_time = now
        return decay_report

    def decay_emotions(self, emotions: list, now: Optional[float] = None,
                       min_intensity: float = 0.05) -> list:
        """
        Apply decay to active emotional states.

        Reduces intensity of each emotion based on elapsed time.
        Removes emotions that fall below min_intensity threshold.

        Args:
            emotions: List of EmotionalState objects
            now: Current timestamp
            min_intensity: Below this, emotion is considered inactive

        Returns:
            Filtered list with decayed intensities
        """
        now = now or time.time()
        surviving = []

        for emotion in emotions:
            profile = EMOTION_DECAY_PROFILES.get(emotion.name, _DEFAULT_PROFILE)
            elapsed = now - emotion.timestamp

            if elapsed <= 0:
                surviving.append(emotion)
                continue

            # Arousal scaling: high-arousal states decay faster
            effective_half_life = profile.half_life
            if profile.arousal_scaling and emotion.arousal > 0.5:
                # Arousal of 1.0 → half_life * 0.5; arousal of 0.5 → half_life * 1.0
                arousal_factor = 1.5 - emotion.arousal
                effective_half_life *= arousal_factor

            decay_factor = self._calculate_decay(
                elapsed=elapsed,
                half_life=effective_half_life,
                stickiness=profile.stickiness,
                momentum=0.0,
            )

            # Decay intensity toward floor
            decayed_intensity = profile.floor + \
                (emotion.intensity - profile.floor) * decay_factor

            if decayed_intensity >= min_intensity:
                emotion.intensity = decayed_intensity
                # Arousal also decays (emotions become less activated over time)
                emotion.arousal *= (0.3 + 0.7 * decay_factor)  # Slower arousal decay
                surviving.append(emotion)

        return surviving

    def _calculate_decay(self, elapsed: float, half_life: float,
                         stickiness: float, momentum: float) -> float:
        """
        Core decay function — exponential decay with stickiness and momentum resistance.

        Returns a factor 0.0 to 1.0 where 1.0 = no decay, 0.0 = full decay.

        Stickiness slows decay by increasing effective half-life.
        Momentum resistance: states that were actively changing decay slower.
        """
        # Stickiness increases effective half-life
        effective_half_life = half_life * (1.0 + stickiness * 2.0)

        # Momentum resistance: actively moving states resist decay
        if momentum > 0.01:
            momentum_factor = 1.0 + min(1.0, momentum) * 0.5
            effective_half_life *= momentum_factor

        # Exponential decay: value = initial * 2^(-t/half_life)
        return math.pow(2.0, -elapsed / effective_half_life)

    def get_time_since_last_decay(self, now: Optional[float] = None) -> Optional[float]:
        """How long since we last ran decay. None if never."""
        if self._last_decay_time is None:
            return None
        return (now or time.time()) - self._last_decay_time
