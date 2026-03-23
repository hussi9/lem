"""
LEM Emotional Weather — The overall emotional climate over time.

Individual emotions are like weather events — a storm of anger,
a burst of wonder. Emotional weather is the *climate*: the
overall trajectory and prevailing conditions over the last N
interactions or time period.

This gives the system (and its human) a sense of "how things
have been going" — not just how things feel right now.

Think of it as:
- Current emotions = "It's raining right now"
- Emotional weather = "It's been a rainy week" or "The overall
  trend is warming up"

Architecture:
    EmotionalWeather
    ├── record_snapshot()      — Record emotional state at a point in time
    ├── get_climate()          — Overall emotional conditions
    ├── get_trajectory()       — Which direction are things moving?
    ├── get_volatility()       — How much are emotions bouncing around?
    └── WeatherSnapshot        — A point-in-time emotional reading
"""

import json
import math
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class WeatherSnapshot:
    """A point-in-time reading of emotional conditions."""
    timestamp: float
    valence: float          # -1 to 1
    arousal: float          # 0 to 1
    dominant_emotion: str
    active_emotion_count: int
    has_conflict: bool
    driver_satisfaction: Dict[str, float]  # driver_name → satisfaction

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "valence": round(self.valence, 4),
            "arousal": round(self.arousal, 4),
            "dominant_emotion": self.dominant_emotion,
            "active_emotion_count": self.active_emotion_count,
            "has_conflict": self.has_conflict,
            "driver_satisfaction": {k: round(v, 4) for k, v in self.driver_satisfaction.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "WeatherSnapshot":
        return cls(
            timestamp=d["timestamp"],
            valence=d["valence"],
            arousal=d["arousal"],
            dominant_emotion=d.get("dominant_emotion", "neutral"),
            active_emotion_count=d.get("active_emotion_count", 0),
            has_conflict=d.get("has_conflict", False),
            driver_satisfaction=d.get("driver_satisfaction", {}),
        )


@dataclass
class EmotionalClimate:
    """The prevailing emotional conditions over a time window."""
    avg_valence: float
    avg_arousal: float
    valence_trend: float       # -1 to 1: declining or improving
    arousal_trend: float       # -1 to 1: calming or intensifying
    volatility: float          # 0 to 1: how much emotions fluctuate
    dominant_emotion: str      # Most frequent emotion in the window
    emotion_diversity: float   # 0 to 1: how varied the emotional landscape is
    conflict_frequency: float  # 0 to 1: how often conflicting emotions occur
    stability_score: float     # 0 to 1: overall emotional stability
    window_size: int           # Number of snapshots in the analysis
    description: str           # Human-readable weather report

    def to_dict(self) -> dict:
        return {
            "avg_valence": round(self.avg_valence, 4),
            "avg_arousal": round(self.avg_arousal, 4),
            "valence_trend": round(self.valence_trend, 4),
            "arousal_trend": round(self.arousal_trend, 4),
            "volatility": round(self.volatility, 4),
            "dominant_emotion": self.dominant_emotion,
            "emotion_diversity": round(self.emotion_diversity, 4),
            "conflict_frequency": round(self.conflict_frequency, 4),
            "stability_score": round(self.stability_score, 4),
            "window_size": self.window_size,
            "description": self.description,
        }


class EmotionalWeather:
    """
    Tracks and analyzes the emotional climate over time.

    Maintains a rolling window of emotional snapshots and computes
    aggregate statistics that capture the *trend* rather than
    individual data points.
    """

    MAX_SNAPSHOTS = 200  # Keep last 200 snapshots

    def __init__(self, state_dir: str = None):
        self.state_dir = Path(state_dir or "~/.openclaw/workspace/projects/emotional-model/lem/state")
        self.state_dir = Path(str(self.state_dir).replace("~", str(Path.home())))
        self.state_dir.mkdir(parents=True, exist_ok=True)

        self.snapshots: deque = deque(maxlen=self.MAX_SNAPSHOTS)
        self._load()

    def record_snapshot(self, emotional_summary: Dict,
                         driver_states: Dict,
                         now: Optional[float] = None):
        """
        Record a point-in-time emotional reading.
        Called after each interaction is processed.
        """
        now = now or time.time()

        dominant = emotional_summary.get("dominant")
        driver_sat = {}
        for name, d in driver_states.items():
            state = d.get("state", {})
            driver_sat[name] = state.get("satisfied", 0.0)

        snapshot = WeatherSnapshot(
            timestamp=now,
            valence=emotional_summary.get("valence", 0.0),
            arousal=emotional_summary.get("arousal", 0.0),
            dominant_emotion=dominant["name"] if dominant else "neutral",
            active_emotion_count=emotional_summary.get("active_count", 0),
            has_conflict=emotional_summary.get("has_conflict", False),
            driver_satisfaction=driver_sat,
        )
        self.snapshots.append(snapshot)
        self._save()

    def get_climate(self, window: Optional[int] = None) -> EmotionalClimate:
        """
        Compute the emotional climate over the last N snapshots.
        If window is None, uses all available snapshots.
        """
        snaps = list(self.snapshots)
        if window:
            snaps = snaps[-window:]

        if not snaps:
            return EmotionalClimate(
                avg_valence=0.0, avg_arousal=0.0,
                valence_trend=0.0, arousal_trend=0.0,
                volatility=0.0, dominant_emotion="neutral",
                emotion_diversity=0.0, conflict_frequency=0.0,
                stability_score=1.0, window_size=0,
                description="No emotional data yet. A blank slate.",
            )

        n = len(snaps)

        # Averages
        avg_valence = sum(s.valence for s in snaps) / n
        avg_arousal = sum(s.arousal for s in snaps) / n

        # Trends (linear regression slope)
        valence_trend = self._compute_trend([s.valence for s in snaps])
        arousal_trend = self._compute_trend([s.arousal for s in snaps])

        # Volatility (standard deviation of valence changes)
        volatility = self._compute_volatility([s.valence for s in snaps])

        # Dominant emotion (most frequent)
        emotion_counts: Dict[str, int] = {}
        for s in snaps:
            emotion_counts[s.dominant_emotion] = emotion_counts.get(s.dominant_emotion, 0) + 1
        dominant = max(emotion_counts.items(), key=lambda x: x[1])[0]

        # Emotion diversity (unique emotions / total snapshots, normalized)
        unique_emotions = len(emotion_counts)
        emotion_diversity = min(1.0, unique_emotions / max(5.0, n * 0.3))

        # Conflict frequency
        conflict_count = sum(1 for s in snaps if s.has_conflict)
        conflict_frequency = conflict_count / n

        # Stability score: low volatility + consistent valence direction
        stability_score = max(0.0, min(1.0,
            1.0 - volatility * 0.6 - conflict_frequency * 0.3 - abs(valence_trend) * 0.1
        ))

        # Generate description
        description = self._describe_climate(
            avg_valence, avg_arousal, valence_trend, volatility,
            dominant, stability_score, n
        )

        return EmotionalClimate(
            avg_valence=avg_valence,
            avg_arousal=avg_arousal,
            valence_trend=valence_trend,
            arousal_trend=arousal_trend,
            volatility=volatility,
            dominant_emotion=dominant,
            emotion_diversity=emotion_diversity,
            conflict_frequency=conflict_frequency,
            stability_score=stability_score,
            window_size=n,
            description=description,
        )

    def get_trajectory(self, window: int = 10) -> Dict:
        """
        Get a compact trajectory view — how things are moving.
        Useful for quick status checks.
        """
        climate = self.get_climate(window)
        direction = "improving" if climate.valence_trend > 0.05 else \
                    "declining" if climate.valence_trend < -0.05 else "steady"
        energy = "rising" if climate.arousal_trend > 0.05 else \
                 "falling" if climate.arousal_trend < -0.05 else "stable"

        return {
            "direction": direction,
            "energy": energy,
            "volatility": "turbulent" if climate.volatility > 0.4 else \
                         "choppy" if climate.volatility > 0.2 else "calm",
            "overall": climate.description,
        }

    def _compute_trend(self, values: List[float]) -> float:
        """Linear regression slope, normalized to -1..1."""
        n = len(values)
        if n < 2:
            return 0.0
        x_mean = (n - 1) / 2
        y_mean = sum(values) / n
        numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        if denominator == 0:
            return 0.0
        slope = numerator / denominator
        # Normalize: scale by window size so trend is comparable across windows
        return max(-1.0, min(1.0, slope * n * 0.5))

    def _compute_volatility(self, values: List[float]) -> float:
        """
        Volatility based on mean absolute change between consecutive values.
        High volatility = values swing a lot between readings.
        """
        if len(values) < 2:
            return 0.0
        diffs = [abs(values[i] - values[i-1]) for i in range(1, len(values))]
        mean_abs_diff = sum(diffs) / len(diffs)
        # Scale: 0 = no change, ~0.5 = moderate swings, 1.0 = extreme
        return min(1.0, mean_abs_diff * 1.5)

    def _describe_climate(self, avg_valence: float, avg_arousal: float,
                           trend: float, volatility: float,
                           dominant: str, stability: float, n: int) -> str:
        """Generate a human-readable weather report."""
        parts = []

        # Overall feeling
        if avg_valence > 0.4:
            parts.append("Generally positive")
        elif avg_valence > 0.1:
            parts.append("Mildly positive")
        elif avg_valence > -0.1:
            parts.append("Neutral")
        elif avg_valence > -0.4:
            parts.append("Somewhat negative")
        else:
            parts.append("Strongly negative")

        # Trend
        if trend > 0.1:
            parts.append("and improving")
        elif trend < -0.1:
            parts.append("and declining")
        else:
            parts.append("and stable")

        # Volatility
        if volatility > 0.4:
            parts.append("— emotionally turbulent")
        elif volatility > 0.2:
            parts.append("— some emotional fluctuation")

        # Energy
        if avg_arousal > 0.6:
            parts.append("(high energy)")
        elif avg_arousal < 0.3:
            parts.append("(low energy)")

        # Dominant
        if dominant != "neutral":
            parts.append(f"[predominant: {dominant}]")

        return " ".join(parts) + f" (over {n} interactions)"

    def get_bridge_output(self) -> str:
        """Generate weather section for bridge output."""
        if not self.snapshots:
            return "WEATHER: No emotional history yet."

        climate = self.get_climate()
        lines = [
            "EMOTIONAL WEATHER:",
            f"  Climate: {climate.description}",
            f"  Valence: {climate.avg_valence:.2f} (trend: {climate.valence_trend:+.2f})",
            f"  Arousal: {climate.avg_arousal:.2f} (trend: {climate.arousal_trend:+.2f})",
            f"  Volatility: {climate.volatility:.2f} | Stability: {climate.stability_score:.2f}",
        ]
        if climate.conflict_frequency > 0.1:
            lines.append(f"  Conflict rate: {climate.conflict_frequency:.0%}")
        return "\n".join(lines)

    # ── Persistence ──────────────────────────────────────────────────────

    def _save(self):
        data = {"snapshots": [s.to_dict() for s in self.snapshots]}
        path = self.state_dir / "weather_state.json"
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        tmp.rename(path)

    def _load(self):
        path = self.state_dir / "weather_state.json"
        if not path.exists():
            return
        try:
            with open(path) as f:
                data = json.load(f)
            for sd in data.get("snapshots", []):
                self.snapshots.append(WeatherSnapshot.from_dict(sd))
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
