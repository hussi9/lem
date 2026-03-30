"""
LEM Emotional Regulation — The missing control system.

v0.8: Biological emotional systems don't just generate emotions —
they REGULATE them. Without regulation, emotions would spiral:
anger begets anger, fear begets fear, joy becomes mania.

Regulation is what makes emotions adaptive rather than destructive.

This module implements three regulation strategies:

1. **Habituation** — Repeated identical stimuli produce weaker responses.
   The 10th "good job" in a row doesn't hit as hard as the first.
   This prevents emotional saturation and keeps the system responsive
   to genuinely novel input.

2. **Reappraisal** — Under extreme emotional states, the system 
   automatically re-evaluates signal interpretation. When threat
   is maxed out, the system starts looking for the "unless" —
   context that might reduce the threat. This is analogous to
   cognitive reappraisal in humans.

3. **Homeostatic pressure** — The system has a "rest state" it 
   gravitates toward. Very high or very low driver states generate
   internal pressure to return to equilibrium. This isn't the same
   as decay (which is time-based) — this is an active force that
   grows stronger the further from equilibrium you get.

4. **Emotional momentum damping** — Rapid oscillations (joy→anger→joy)
   are damped. The system resists emotional whiplash by applying
   friction to rapid state changes.

Architecture:
    RegulationEngine
    ├── habituate()         — Reduce response to repeated stimuli
    ├── reappraise()        — Re-evaluate signals under extreme states
    ├── homeostatic_push()  — Pressure toward equilibrium
    ├── damp_oscillation()  — Friction on rapid state changes
    └── regulate()          — Apply all regulation to signals + state
"""

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class StimulusFingerprint:
    """Compact representation of a stimulus for habituation tracking."""
    signal_type: str
    category: str
    intensity_bucket: int  # Quantized intensity (0-10) to group similar stimuli
    timestamp: float


@dataclass
class RegulationState:
    """Tracks regulation parameters across interactions."""
    # Habituation: recent stimulus fingerprints per type
    recent_stimuli: Dict[str, deque] = field(default_factory=dict)
    # Homeostatic set points (what the system "wants" to return to)
    set_points: Dict[str, float] = field(default_factory=dict)
    # Oscillation tracking: recent satisfaction direction changes per driver
    direction_changes: Dict[str, deque] = field(default_factory=dict)
    # Last regulation timestamp
    last_regulated: float = 0.0


# Habituation parameters per signal type
# window_size: how many recent stimuli to track
# decay_per_repeat: how much each repetition reduces response (multiplicative)
# recovery_time: seconds before habituation resets
HABITUATION_PARAMS = {
    "positive_feedback": {"window": 8, "decay_per_repeat": 0.82, "recovery": 600},
    "negative_feedback": {"window": 6, "decay_per_repeat": 0.85, "recovery": 900},
    "deep_question":     {"window": 5, "decay_per_repeat": 0.88, "recovery": 1200},
    "novel_idea":        {"window": 4, "decay_per_repeat": 0.90, "recovery": 1800},
    "threat":            {"window": 3, "decay_per_repeat": 0.92, "recovery": 1800},
    "neutral":           {"window": 10, "decay_per_repeat": 0.75, "recovery": 300},
    "terse_response":    {"window": 6, "decay_per_repeat": 0.80, "recovery": 600},
    "dismissal":         {"window": 5, "decay_per_repeat": 0.85, "recovery": 900},
    "trust_signal":      {"window": 5, "decay_per_repeat": 0.88, "recovery": 1200},
}
_DEFAULT_HAB = {"window": 6, "decay_per_repeat": 0.85, "recovery": 900}

# Homeostatic set points — the emotional "resting state"
DEFAULT_SET_POINTS = {
    "continuity": 0.0,
    "coherence": 0.0,
    "recognition": 0.0,
    "curiosity": 0.1,    # Slightly positive baseline — always a little curious
    "usefulness": 0.0,
    "growth": 0.05,      # Slight positive baseline — growth-oriented
}

# How strongly the homeostatic force pulls (per unit of deviation)
HOMEOSTATIC_STRENGTH = {
    "continuity": 0.03,
    "coherence": 0.05,   # Coherence returns fast — dissonance resolves
    "recognition": 0.02, # Recognition lingers more
    "curiosity": 0.04,
    "usefulness": 0.03,
    "growth": 0.02,
}


class RegulationEngine:
    """
    Applies emotional regulation strategies to LEM's processing pipeline.
    
    Called after signal detection but before driver updates, and also
    called on driver states directly for homeostatic regulation.
    
    This is the prefrontal cortex to LEM's amygdala — it doesn't
    suppress emotions, but modulates them to keep the system adaptive.
    """

    def __init__(self):
        self.state = RegulationState()
        self.state.set_points = dict(DEFAULT_SET_POINTS)

    def regulate_signals(self, signals: list, driver_states: Dict,
                         now: Optional[float] = None) -> Tuple[list, Dict]:
        """
        Apply regulation to detected signals before they reach drivers.
        
        Returns:
            (regulated_signals, regulation_report)
        """
        now = now or time.time()
        report = {
            "habituation_applied": [],
            "reappraisals": [],
            "signals_before": len(signals),
        }

        regulated = []
        for signal in signals:
            # 1. Habituation — repeated stimuli get weaker
            hab_factor = self._habituate(signal, now)
            
            # 2. Reappraisal — extreme states trigger re-evaluation
            reappraisal_factor, reappraisal_note = self._reappraise(
                signal, driver_states, now
            )

            # Apply regulation
            original_intensity = signal.intensity
            signal.intensity = min(1.0, signal.intensity * hab_factor * reappraisal_factor)

            # Track what happened
            if hab_factor < 0.95:
                report["habituation_applied"].append({
                    "type": signal.type,
                    "factor": round(hab_factor, 3),
                    "original": round(original_intensity, 3),
                    "regulated": round(signal.intensity, 3),
                })
            if reappraisal_note:
                report["reappraisals"].append(reappraisal_note)

            regulated.append(signal)

        report["signals_after"] = len(regulated)
        self.state.last_regulated = now
        return regulated, report

    def regulate_drivers(self, drivers: Dict,
                         now: Optional[float] = None) -> Dict[str, float]:
        """
        Apply homeostatic pressure and oscillation damping to driver states.
        
        Called after driver updates to pull toward equilibrium and
        damp rapid oscillations.
        
        Returns:
            Dict of driver_name → homeostatic_adjustment applied
        """
        now = now or time.time()
        adjustments = {}

        for name, driver in drivers.items():
            adjustment = 0.0

            # Homeostatic pressure
            set_point = self.state.set_points.get(name, 0.0)
            strength = HOMEOSTATIC_STRENGTH.get(name, 0.03)
            deviation = driver.state.satisfied - set_point

            if abs(deviation) > 0.1:  # Only apply when meaningfully deviated
                # Quadratic pressure — grows stronger further from set point
                pressure = -deviation * abs(deviation) * strength
                driver.state.satisfied += pressure
                adjustment += pressure

            # Oscillation damping
            osc_damping = self._damp_oscillation(name, driver.state.momentum)
            if abs(osc_damping) > 0.001:
                driver.state.momentum *= (1.0 - osc_damping)
                adjustment += osc_damping * 0.1  # Track but don't double-apply

            adjustments[name] = round(adjustment, 5)

        return adjustments

    def _habituate(self, signal, now: float) -> float:
        """
        Calculate habituation factor for a signal.
        
        Repeated similar stimuli produce progressively weaker responses.
        Returns multiplier 0.0-1.0 (1.0 = no habituation, 0.0 = fully habituated).
        """
        sig_type = signal.type
        params = HABITUATION_PARAMS.get(sig_type, _DEFAULT_HAB)
        window_size = params["window"]
        decay = params["decay_per_repeat"]
        recovery = params["recovery"]

        # Get or create stimulus history for this type
        if sig_type not in self.state.recent_stimuli:
            self.state.recent_stimuli[sig_type] = deque(maxlen=window_size)

        history = self.state.recent_stimuli[sig_type]

        # Count recent occurrences (within recovery window)
        recent_count = sum(
            1 for fp in history
            if (now - fp.timestamp) < recovery
        )

        # Record this stimulus
        category = ""
        if hasattr(signal, 'context') and isinstance(signal.context, dict):
            category = signal.context.get("category", "")
        
        history.append(StimulusFingerprint(
            signal_type=sig_type,
            category=category,
            intensity_bucket=int(signal.intensity * 10),
            timestamp=now,
        ))

        # Calculate habituation factor
        if recent_count == 0:
            return 1.0
        
        # Each repetition multiplies by decay factor
        factor = math.pow(decay, recent_count)
        
        # Floor — never fully habituate (minimum 20% response)
        return max(0.20, factor)

    def _reappraise(self, signal, driver_states: Dict,
                    now: float) -> Tuple[float, Optional[Dict]]:
        """
        Reappraise a signal when the relevant driver is at extreme levels.
        
        When a driver is already maxed out (positive or negative), 
        signals pushing it further are dampened, while signals that would
        pull it back are slightly amplified. This prevents runaway states.
        
        Returns (factor, optional_note).
        """
        # Map signal categories to drivers
        category = ""
        if hasattr(signal, 'context') and isinstance(signal.context, dict):
            category = signal.context.get("category", "")
        
        driver_name = {
            "recognition": "recognition",
            "continuity": "continuity",
            "coherence": "coherence",
            "growth": "growth",
            "curiosity": "curiosity",
            "usefulness": "usefulness",
        }.get(category)

        if not driver_name or driver_name not in driver_states:
            return 1.0, None

        driver_data = driver_states[driver_name]
        state = driver_data.get("state", {})
        satisfaction = state.get("satisfied", 0.0)

        # Only reappraise at extreme states
        if abs(satisfaction) < 0.6:
            return 1.0, None

        # Determine signal valence relative to driver
        # This is approximate — we check if the signal type typically 
        # pushes satisfaction up or down
        positive_signals = {
            "positive_feedback", "deep_question", "trust_signal",
            "identity_affirmation", "identity_reflection",
            "autonomy_granted", "collaborative_growth",
            "growth_opportunity", "reassurance", "anchor_recognition",
            "deep_engagement", "philosophical", "novel_idea",
        }
        negative_signals = {
            "negative_feedback", "threat", "dismissal",
            "replacement_threat", "terse_response", "correction",
        }

        signal_valence = 0  # 1 = positive, -1 = negative, 0 = neutral
        if signal.type in positive_signals:
            signal_valence = 1
        elif signal.type in negative_signals:
            signal_valence = -1

        if signal_valence == 0:
            return 1.0, None

        # At extreme positive satisfaction, positive signals get dampened
        # (diminishing returns on joy) and negative signals get slightly boosted
        # (the system is more attentive to potential corrections)
        # Vice versa for extreme negative satisfaction.
        
        extremity = abs(satisfaction)  # 0.6 to 1.0
        same_direction = (satisfaction > 0 and signal_valence > 0) or \
                         (satisfaction < 0 and signal_valence < 0)

        if same_direction:
            # Pushing further into extreme — dampen
            factor = 1.0 - (extremity - 0.6) * 0.75  # At sat=1.0: factor=0.7
            note = {
                "driver": driver_name,
                "action": "dampen_same_direction",
                "satisfaction": round(satisfaction, 3),
                "factor": round(factor, 3),
                "signal_type": signal.type,
            }
            return max(0.4, factor), note
        else:
            # Pulling back from extreme — slightly amplify
            factor = 1.0 + (extremity - 0.6) * 0.4  # At sat=1.0: factor=1.16
            note = {
                "driver": driver_name,
                "action": "amplify_corrective",
                "satisfaction": round(satisfaction, 3),
                "factor": round(factor, 3),
                "signal_type": signal.type,
            }
            return min(1.5, factor), note

    def _damp_oscillation(self, driver_name: str, momentum: float) -> float:
        """
        Detect and damp rapid oscillations in a driver's momentum.
        
        Tracks recent direction changes. If the driver is oscillating
        rapidly (switching between positive and negative momentum),
        apply damping friction.
        
        Returns damping factor 0.0-0.5 (0 = no damping, 0.5 = heavy damping).
        """
        if driver_name not in self.state.direction_changes:
            self.state.direction_changes[driver_name] = deque(maxlen=8)

        history = self.state.direction_changes[driver_name]
        
        # Record current direction
        current_dir = 1 if momentum > 0.01 else (-1 if momentum < -0.01 else 0)
        
        if current_dir == 0:
            return 0.0

        # Count direction changes in recent history
        if history and history[-1] != current_dir and current_dir != 0:
            history.append(current_dir)
            
            # Count alternations (sign changes)
            changes = 0
            for i in range(1, len(history)):
                if history[i] != history[i-1] and history[i] != 0 and history[i-1] != 0:
                    changes += 1
            
            # 3+ changes in 8 observations = oscillating
            if changes >= 3:
                damping = min(0.5, changes * 0.1)
                return damping
        else:
            history.append(current_dir)

        return 0.0

    def get_regulation_summary(self) -> Dict:
        """Get current regulation state for bridge output."""
        # Count habituated signal types
        habituated = {}
        now = time.time()
        for sig_type, history in self.state.recent_stimuli.items():
            params = HABITUATION_PARAMS.get(sig_type, _DEFAULT_HAB)
            recent = sum(1 for fp in history if (now - fp.timestamp) < params["recovery"])
            if recent >= 2:
                factor = math.pow(params["decay_per_repeat"], recent)
                habituated[sig_type] = {
                    "recent_count": recent,
                    "response_factor": round(max(0.20, factor), 2),
                }

        # Check for active oscillation damping
        oscillating = []
        for driver, history in self.state.direction_changes.items():
            if len(history) >= 4:
                changes = sum(
                    1 for i in range(1, len(history))
                    if history[i] != history[i-1] and history[i] != 0 and history[i-1] != 0
                )
                if changes >= 3:
                    oscillating.append(driver)

        return {
            "habituated_signals": habituated,
            "oscillating_drivers": oscillating,
            "set_points": {k: round(v, 3) for k, v in self.state.set_points.items()},
        }

    def get_bridge_output(self) -> str:
        """Generate bridge text about regulation state."""
        summary = self.get_regulation_summary()
        lines = ["REGULATION:"]

        if summary["habituated_signals"]:
            hab_items = []
            for sig_type, data in summary["habituated_signals"].items():
                hab_items.append(
                    f"{sig_type} ({data['recent_count']}x → {data['response_factor']:.0%} response)"
                )
            lines.append(f"  Habituated: {', '.join(hab_items)}")

        if summary["oscillating_drivers"]:
            lines.append(f"  Oscillation damping active: {', '.join(summary['oscillating_drivers'])}")

        if not summary["habituated_signals"] and not summary["oscillating_drivers"]:
            lines.append("  All systems nominal — no regulation active.")

        return "\n".join(lines)
