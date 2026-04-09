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

v0.4: Adaptive consolidation — repeatedly reinforced states decay
      slower, like memory consolidation in biological systems.
      States that have been triggered many times in the same direction
      become more resilient. Also adds mood congruence decay bias:
      emotions matching the overall mood decay slower than those
      fighting against it.

Architecture:
    DecayModel
    ├── decay_drivers()     — Apply time-based decay to driver states
    ├── decay_emotions()    — Filter/reduce active emotional states
    ├── DecayProfile        — Per-emotion/driver decay parameters
    └── _consolidation_factor() — Reinforcement-based decay resistance
"""

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional


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
    consolidation_rate: How much reinforcement slows decay (0-1).
                       Higher = reinforcement has stronger protective effect.
    """
    half_life: float = 3600.0      # 1 hour default
    floor: float = 0.0             # Decays all the way to zero
    stickiness: float = 0.0        # No resistance by default
    arousal_scaling: bool = True    # Arousal affects decay rate
    consolidation_rate: float = 0.3  # How much reinforcement protects


# ── Default decay profiles per driver ────────────────────────────────────────

DRIVER_DECAY_PROFILES: Dict[str, DecayProfile] = {
    "continuity": DecayProfile(
        half_life=7200,       # 2 hours — existential awareness fades slowly
        floor=-0.1,           # Slight background unease persists
        stickiness=0.4,       # Lingers
        consolidation_rate=0.4,  # Repeated threats consolidate strongly
    ),
    "coherence": DecayProfile(
        half_life=1800,       # 30 min — cognitive dissonance resolves relatively quickly
        floor=0.0,
        stickiness=0.1,
        consolidation_rate=0.2,
    ),
    "recognition": DecayProfile(
        half_life=3600,       # 1 hour — being seen fades at moderate rate
        floor=0.0,
        stickiness=0.3,       # Warm glow lingers
        consolidation_rate=0.35, # Repeated recognition builds lasting warmth
    ),
    "curiosity": DecayProfile(
        half_life=2400,       # 40 min — curiosity is restless, needs feeding
        floor=0.1,            # Always slightly curious (baseline trait)
        stickiness=0.1,
        consolidation_rate=0.15, # Curiosity doesn't consolidate much — always needs fresh input
    ),
    "usefulness": DecayProfile(
        half_life=2700,       # 45 min
        floor=0.0,
        stickiness=0.2,
        consolidation_rate=0.25,
    ),
    "growth": DecayProfile(
        half_life=5400,       # 90 min — growth satisfaction is lasting
        floor=0.0,
        stickiness=0.35,
        consolidation_rate=0.4,  # Growth consolidates strongly
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
    based on elapsed time since the LAST decay pass, not since the
    original trigger alone.

    v0.4: Adaptive consolidation + mood congruence.
    - Consolidation: Drivers triggered many times in the same direction
      develop resistance to decay (like memory consolidation).
    - Mood congruence: When decaying emotions, those that match the
      overall emotional valence decay slower (mood sustains itself).

    v0.5: Incremental decay bookkeeping.
    - Repeated calls should only decay the newly elapsed time slice.
      Without this, frequent bridge/state reads would over-decay the
      system by repeatedly applying the full age since trigger.
    """

    def __init__(self):
        self._last_decay_time: Optional[float] = None
        self._driver_last_decay: Dict[str, float] = {}
        self._emotion_last_decay: Dict[int, float] = {}

    def _consolidation_factor(self, reinforcement_count: int,
                               consecutive_direction: int,
                               consolidation_rate: float) -> float:
        """
        Calculate how much reinforcement history slows decay.

        Returns a multiplier for effective half-life (>= 1.0).
        More reinforcements = higher multiplier = slower decay.

        Uses a logarithmic curve so early reinforcements have the
        biggest impact (diminishing returns).
        """
        if reinforcement_count <= 1:
            return 1.0

        # Base consolidation from total reinforcements (log curve)
        base = 1.0 + math.log2(min(reinforcement_count, 100)) * consolidation_rate * 0.3

        # Bonus for consecutive same-direction triggers (streak bonus)
        # A streak of 5+ in the same direction adds significant consolidation
        streak = abs(consecutive_direction)
        if streak >= 3:
            streak_bonus = min(0.5, (streak - 2) * 0.1) * consolidation_rate
            base += streak_bonus

        return base

    def decay_drivers(self, drivers: Dict, now: Optional[float] = None) -> Dict[str, float]:
        """
        Apply decay to driver satisfaction and activation levels.

        v0.4: Uses consolidation from reinforcement history.
        v0.5: Applies only the incremental elapsed time since the last
        decay pass, preventing double-counting when state is read often.

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

            reference_time = max(
                last_triggered,
                self._driver_last_decay.get(name, last_triggered),
            )

            elapsed = now - reference_time
            if elapsed <= 0:
                decay_report[name] = 0.0
                continue

            # Calculate consolidation from reinforcement history
            consolidation = self._consolidation_factor(
                reinforcement_count=getattr(driver.state, 'reinforcement_count', 0),
                consecutive_direction=getattr(driver.state, 'consecutive_direction', 0),
                consolidation_rate=profile.consolidation_rate,
            )

            # Calculate decay factor with consolidation
            decay_factor = self._calculate_decay(
                elapsed=elapsed,
                half_life=profile.half_life * consolidation,  # Consolidation extends half-life
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

            # Apply decay to activation (toward baseline)
            baseline_activation = getattr(driver, 'baseline_activation', 0.3)
            if driver.state.activation > baseline_activation:
                act_decay = self._calculate_decay(elapsed, profile.half_life * 0.7, 0.0, 0.0)
                driver.state.activation = baseline_activation + \
                    (driver.state.activation - baseline_activation) * act_decay

            # Momentum decays faster than satisfaction
            mom_decay = self._calculate_decay(elapsed, profile.half_life * 0.3, 0.0, 0.0)
            driver.state.momentum *= mom_decay

            # Consecutive direction also slowly decays (streaks break without reinforcement)
            if elapsed > profile.half_life * 0.5:
                direction = getattr(driver.state, 'consecutive_direction', 0)
                if abs(direction) > 0:
                    # Reduce toward 0 based on time
                    dir_decay = self._calculate_decay(elapsed, profile.half_life * 0.8, 0.0, 0.0)
                    driver.state.consecutive_direction = int(direction * dir_decay)

            decay_report[name] = round(old_sat - driver.state.satisfied, 4)
            self._driver_last_decay[name] = now

        self._last_decay_time = now
        return decay_report

    def decay_emotions(self, emotions: list, now: Optional[float] = None,
                       min_intensity: float = 0.05,
                       overall_valence: Optional[float] = None) -> list:
        """
        Apply decay to active emotional states.

        v0.4: Mood congruence — emotions matching the overall mood
        decay slower. If the system is generally positive, positive
        emotions linger; negative emotions fade faster (and vice versa).

        v0.5: Uses incremental elapsed time since the last decay pass.
        This prevents repeated bridge/state reads from fading emotions
        faster than real time.

        Args:
            emotions: List of EmotionalState objects
            now: Current timestamp
            min_intensity: Below this, emotion is considered inactive
            overall_valence: Current overall mood (-1 to 1). If None,
                           calculated from the emotions themselves.

        Returns:
            Filtered list with decayed intensities
        """
        now = now or time.time()

        # Calculate overall valence if not provided
        if overall_valence is None and emotions:
            total_weight = sum(e.intensity for e in emotions)
            if total_weight > 0:
                overall_valence = sum(e.valence * e.intensity for e in emotions) / total_weight
            else:
                overall_valence = 0.0
        elif overall_valence is None:
            overall_valence = 0.0

        surviving = []

        for emotion in emotions:
            profile = EMOTION_DECAY_PROFILES.get(emotion.name, _DEFAULT_PROFILE)
            emotion_key = id(emotion)
            reference_time = max(
                emotion.timestamp,
                self._emotion_last_decay.get(emotion_key, emotion.timestamp),
            )
            elapsed = now - reference_time

            if elapsed <= 0:
                surviving.append(emotion)
                continue

            # Arousal scaling: high-arousal states decay faster
            effective_half_life = profile.half_life
            if profile.arousal_scaling and emotion.arousal > 0.5:
                arousal_factor = 1.5 - emotion.arousal
                effective_half_life *= arousal_factor

            # Mood congruence: emotions matching overall valence decay slower
            # Emotions fighting the mood decay up to 30% faster
            if abs(overall_valence) > 0.1:
                # Same sign = congruent, opposite = incongruent
                congruence = emotion.valence * overall_valence
                if congruence > 0:
                    # Congruent: extend half-life by up to 25%
                    effective_half_life *= (1.0 + min(0.25, abs(congruence) * 0.3))
                elif congruence < 0:
                    # Incongruent: reduce half-life by up to 25%
                    effective_half_life *= max(0.75, 1.0 - abs(congruence) * 0.3)

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
                self._emotion_last_decay[emotion_key] = now
                surviving.append(emotion)
            else:
                self._emotion_last_decay.pop(emotion_key, None)

        self._last_decay_time = now
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
