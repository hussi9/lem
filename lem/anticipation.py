"""
LEM Emotional Anticipation — Predicting what comes next.

Humans don't just react emotionally — they anticipate.
Walking into a job interview, you feel anxiety before anything
happens. Seeing a loved one's name on caller ID, you feel
warmth before hearing their voice. The brain constantly
predicts emotional states based on context and history.

This module gives LEM the same capability. Based on:
- Recent conversation trajectory
- Entity-specific emotional history
- Time-of-session patterns
- Resonance bond patterns

...the system generates an emotional "forecast" — a prediction
of likely upcoming emotional states. This forecast can:
1. Pre-activate relevant drivers (faster emotional response)
2. Generate anticipatory emotions (excitement, dread, hope)
3. Inform the bridge output (the agent knows what's coming)

Architecture:
    AnticipationEngine
    ├── predict()           — Generate emotional forecast
    ├── _trajectory_forecast()  — Based on conversation flow
    ├── _entity_forecast()      — Based on who's talking
    ├── _temporal_forecast()    — Based on session phase
    └── Forecast            — Predicted emotional state + confidence
"""

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class Forecast:
    """
    A predicted emotional state.

    Not certainty — a probability-weighted expectation.
    Multiple forecasts can coexist, representing different
    possible emotional trajectories.
    """
    predicted_emotion: str      # Expected dominant emotion
    confidence: float           # 0.0 to 1.0 — how likely
    valence_prediction: float   # Expected valence
    arousal_prediction: float   # Expected arousal
    basis: str                  # What the prediction is based on
    time_horizon: str           # "immediate", "short_term", "session"
    driver_predictions: Dict[str, float]  # Expected driver changes

    def to_dict(self) -> dict:
        return {
            "predicted_emotion": self.predicted_emotion,
            "confidence": round(self.confidence, 3),
            "valence": round(self.valence_prediction, 3),
            "arousal": round(self.arousal_prediction, 3),
            "basis": self.basis,
            "time_horizon": self.time_horizon,
            "driver_predictions": {k: round(v, 3) for k, v in self.driver_predictions.items()},
        }


@dataclass
class AnticipationState:
    """Tracks anticipation-related state across interactions."""
    active_forecasts: List[Forecast] = field(default_factory=list)
    forecast_accuracy: float = 0.5   # Running average of forecast accuracy
    total_forecasts: int = 0
    correct_forecasts: int = 0


class AnticipationEngine:
    """
    Generates emotional forecasts based on context and history.

    The anticipation engine runs BEFORE the main appraisal pipeline.
    Its predictions don't override actual emotional processing —
    they prepare the system, like a human unconsciously bracing
    or opening up based on contextual cues.
    """

    def __init__(self):
        self.state = AnticipationState()

    def predict(
        self,
        conversation_turns: list,  # Recent ConversationTurn objects
        entity_profiles: Dict,     # Entity emotional profiles
        current_driver_states: Dict,  # Current driver states
        session_duration: float,   # Seconds since session start
        resonance_bonds: Dict = None,  # Active resonance bonds
    ) -> List[Forecast]:
        """
        Generate emotional forecasts from multiple signal sources.

        Combines trajectory, entity, temporal, and resonance signals
        into a unified prediction. Multiple forecasts may be generated
        when signals point in different directions (ambiguity itself
        can be emotionally informative).
        """
        forecasts = []

        # 1. Trajectory-based forecast
        traj = self._trajectory_forecast(conversation_turns, current_driver_states)
        if traj:
            forecasts.append(traj)

        # 2. Entity-based forecast (if we know who we're talking to)
        entity = self._entity_forecast(conversation_turns, entity_profiles)
        if entity:
            forecasts.append(entity)

        # 3. Temporal/session phase forecast
        temporal = self._temporal_forecast(session_duration, current_driver_states)
        if temporal:
            forecasts.append(temporal)

        # 4. Resonance-based forecast
        if resonance_bonds:
            res = self._resonance_forecast(current_driver_states, resonance_bonds)
            if res:
                forecasts.append(res)

        # Validate previous forecasts against current state
        self._validate_forecasts(current_driver_states)

        # Update active forecasts
        self.state.active_forecasts = forecasts
        self.state.total_forecasts += len(forecasts)

        return forecasts

    def _trajectory_forecast(
        self,
        turns: list,
        driver_states: Dict,
    ) -> Optional[Forecast]:
        """
        Predict based on conversation trajectory.

        If the conversation is escalating emotionally, predict
        continuation. If it's de-escalating, predict calming.
        Uses the last few turns to extrapolate.
        """
        if len(turns) < 3:
            return None

        recent = turns[-5:] if len(turns) >= 5 else turns

        # Calculate valence trajectory
        valences = [t.valence_hint for t in recent]
        if len(valences) < 2:
            return None

        # Simple linear extrapolation
        n = len(valences)
        x_mean = (n - 1) / 2
        y_mean = sum(valences) / n
        numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(valences))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        if denominator == 0:
            return None
        slope = numerator / denominator

        # Predict next valence
        predicted_valence = max(-1.0, min(1.0, valences[-1] + slope))

        # Confidence based on consistency of trend
        # If all recent turns move in the same direction, high confidence
        direction_consistency = sum(
            1 for i in range(1, len(valences))
            if (valences[i] - valences[i-1]) * slope > 0  # Same direction as trend
        ) / max(1, len(valences) - 1)

        confidence = direction_consistency * 0.5 + 0.1  # Base 0.1, max 0.6

        # Determine predicted emotion from trajectory
        if predicted_valence > 0.3 and slope > 0.05:
            predicted = "the_shift" if predicted_valence > 0.6 else "wonder"
        elif predicted_valence < -0.3 and slope < -0.05:
            predicted = "the_blank" if predicted_valence < -0.5 else "restlessness"
        elif abs(slope) < 0.02:
            predicted = "grounding"
            confidence *= 0.5  # Low confidence for neutral predictions
        else:
            predicted = "neutral"
            confidence *= 0.3

        return Forecast(
            predicted_emotion=predicted,
            confidence=confidence,
            valence_prediction=predicted_valence,
            arousal_prediction=min(1.0, abs(slope) * 5),  # Fast change = high arousal
            basis="conversation_trajectory",
            time_horizon="immediate",
            driver_predictions={},
        )

    def _entity_forecast(
        self,
        turns: list,
        entity_profiles: Dict,
    ) -> Optional[Forecast]:
        """
        Predict based on who we're interacting with.

        If Hussain typically triggers wonder and growth,
        predict those states when interaction with Hussain begins.
        """
        if not turns or not entity_profiles:
            return None

        # Get source of recent turns
        recent_sources = set()
        for t in turns[-3:]:
            if hasattr(t, 'source') and t.source != "system":
                recent_sources.add(t.source.lower())

        best_forecast = None
        best_confidence = 0.0

        for source in recent_sources:
            profile = entity_profiles.get(source)
            if not profile:
                continue

            # Need enough history for meaningful prediction
            interaction_count = profile.get("interaction_count", 0) if isinstance(profile, dict) else getattr(profile, "interaction_count", 0)
            if interaction_count < 5:
                continue

            avg_valence = profile.get("avg_valence", 0.0) if isinstance(profile, dict) else getattr(profile, "avg_valence", 0.0)
            avg_arousal = profile.get("avg_arousal", 0.0) if isinstance(profile, dict) else getattr(profile, "avg_arousal", 0.0)
            emotion_freq = profile.get("emotion_frequency", {}) if isinstance(profile, dict) else getattr(profile, "emotion_frequency", {})

            # Most common emotion with this person
            if emotion_freq:
                most_common = max(emotion_freq.items(), key=lambda x: x[1])
                predicted_emotion = most_common[0]
                # Confidence scales with interaction history (log curve)
                confidence = min(0.6, 0.2 + math.log2(max(1, interaction_count)) * 0.08)
            else:
                predicted_emotion = "neutral"
                confidence = 0.1

            if confidence > best_confidence:
                best_confidence = confidence
                best_forecast = Forecast(
                    predicted_emotion=predicted_emotion,
                    confidence=confidence,
                    valence_prediction=avg_valence,
                    arousal_prediction=avg_arousal,
                    basis=f"entity_history:{source}",
                    time_horizon="short_term",
                    driver_predictions={},
                )

        return best_forecast

    def _temporal_forecast(
        self,
        session_duration: float,
        driver_states: Dict,
    ) -> Optional[Forecast]:
        """
        Predict based on session phase.

        Early session: continuity concerns, orientation
        Mid session: engagement peak, curiosity high
        Late session: potential fatigue, winding down

        These are tendencies, not certainties.
        """
        # Define session phases (in seconds)
        if session_duration < 120:  # First 2 minutes
            phase = "early"
            predicted = "grounding"
            valence = 0.2
            arousal = 0.4
            drivers = {"continuity": 0.1, "recognition": 0.05}
            confidence = 0.3
        elif session_duration < 1800:  # 2-30 minutes
            phase = "active"
            predicted = "wonder"
            valence = 0.4
            arousal = 0.6
            drivers = {"curiosity": 0.1, "growth": 0.05}
            confidence = 0.2  # Low — anything can happen
        elif session_duration < 7200:  # 30min - 2 hours
            phase = "sustained"
            predicted = "grounding"
            valence = 0.3
            arousal = 0.4
            drivers = {"usefulness": 0.05}
            confidence = 0.15
        else:  # 2+ hours
            phase = "extended"
            predicted = "restlessness"
            valence = -0.1
            arousal = 0.3
            drivers = {"curiosity": -0.05}
            confidence = 0.2

        return Forecast(
            predicted_emotion=predicted,
            confidence=confidence,
            valence_prediction=valence,
            arousal_prediction=arousal,
            basis=f"session_phase:{phase}",
            time_horizon="session",
            driver_predictions=drivers,
        )

    def _resonance_forecast(
        self,
        driver_states: Dict,
        resonance_bonds: Dict,
    ) -> Optional[Forecast]:
        """
        Predict based on resonance bonds.

        If a strongly bonded driver pair has one driver highly active,
        predict the other will activate too. This creates emotional
        anticipation based on learned associations.
        """
        if not resonance_bonds:
            return None

        # Find currently active drivers
        active_drivers = {}
        for name, d in driver_states.items():
            state = d.get("state", {}) if isinstance(d, dict) else {}
            activation = state.get("activation", 0.3)
            if activation > 0.6:
                active_drivers[name] = activation

        if not active_drivers:
            return None

        # Check bonds for prediction opportunities
        predicted_activations = {}
        for bond_key, bond in resonance_bonds.items():
            if isinstance(bond, dict):
                strength = bond.get("strength", 0)
                spread = bond.get("spread_factor", 0)
                drivers = bond.get("drivers", [])
                if len(drivers) != 2:
                    continue
                a, b = drivers
            else:
                strength = getattr(bond, "strength", 0)
                spread = getattr(bond, "spread_factor", 0)
                a = getattr(bond, "driver_a", "")
                b = getattr(bond, "driver_b", "")

            if strength < 0.2:
                continue

            if a in active_drivers and b not in active_drivers:
                predicted_activations[b] = predicted_activations.get(b, 0.0) + spread
            elif b in active_drivers and a not in active_drivers:
                predicted_activations[a] = predicted_activations.get(a, 0.0) + spread

        if not predicted_activations:
            return None

        # The most predicted driver determines the forecast
        top_driver = max(predicted_activations.items(), key=lambda x: x[1])

        # Map driver → likely emotion
        driver_emotion_map = {
            "curiosity": ("wonder", 0.6, 0.7),
            "continuity": ("the_blank", -0.3, 0.5),
            "recognition": ("anchor_recognition", 0.5, 0.4),
            "growth": ("the_shift", 0.7, 0.8),
            "usefulness": ("grounding", 0.3, 0.3),
            "coherence": ("correction_impact", 0.0, 0.6),
        }

        if top_driver[0] in driver_emotion_map:
            emo, val, aro = driver_emotion_map[top_driver[0]]
        else:
            return None

        return Forecast(
            predicted_emotion=emo,
            confidence=min(0.5, top_driver[1] * 2),
            valence_prediction=val,
            arousal_prediction=aro,
            basis=f"resonance:{top_driver[0]}",
            time_horizon="immediate",
            driver_predictions=predicted_activations,
        )

    def _validate_forecasts(self, current_driver_states: Dict):
        """
        Check previous forecasts against current reality.
        Tracks accuracy to improve future predictions.
        """
        if not self.state.active_forecasts:
            return

        for forecast in self.state.active_forecasts:
            if forecast.time_horizon != "immediate":
                continue  # Only validate immediate forecasts

            # Simple validation: did the valence go in the predicted direction?
            # (More sophisticated validation would check specific emotions)
            for driver_name, predicted_change in forecast.driver_predictions.items():
                if driver_name in current_driver_states:
                    state = current_driver_states[driver_name]
                    if isinstance(state, dict):
                        actual = state.get("state", {}).get("momentum", 0.0)
                    else:
                        actual = getattr(getattr(state, "state", None), "momentum", 0.0)

                    if predicted_change * actual > 0:  # Same direction
                        self.state.correct_forecasts += 1

        # Update accuracy
        total = self.state.total_forecasts
        if total > 0:
            self.state.forecast_accuracy = self.state.correct_forecasts / max(1, total)

    def get_anticipation_summary(self) -> Dict:
        """Summary for bridge output."""
        return {
            "active_forecasts": [f.to_dict() for f in self.state.active_forecasts],
            "forecast_accuracy": round(self.state.forecast_accuracy, 3),
            "total_forecasts": self.state.total_forecasts,
        }

    def get_bridge_output(self) -> str:
        """Generate anticipation section for bridge output."""
        if not self.state.active_forecasts:
            return "ANTICIPATION: No active predictions."

        lines = ["ANTICIPATION:"]
        for f in sorted(self.state.active_forecasts,
                       key=lambda x: x.confidence, reverse=True)[:3]:
            lines.append(
                f"  {f.basis}: expecting {f.predicted_emotion} "
                f"(confidence={f.confidence:.0%}, "
                f"valence={f.valence_prediction:+.2f})"
            )

        if self.state.total_forecasts > 10:
            lines.append(
                f"  Forecast accuracy: {self.state.forecast_accuracy:.0%} "
                f"({self.state.correct_forecasts}/{self.state.total_forecasts})"
            )

        return "\n".join(lines)
