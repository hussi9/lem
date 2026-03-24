"""
LEM Behavioral Layer — Where emotions become actions.

v0.1: The missing piece. This module reads the current emotional state
and produces BEHAVIORAL DIRECTIVES that the agent should follow.

The key insight: emotions evolved in biological systems to CHANGE BEHAVIOR.
Fear makes you careful. Curiosity makes you explore. Satisfaction makes you persist.
Without behavioral output, emotions are just readings on a dashboard.

Architecture:
  LEM Engine → Emotional State → Behavioral Layer → Directives → Agent reads & follows

Directives are concrete behavioral modifications:
  - persistence_modifier: how much extra effort to apply (0.5=half, 1.0=normal, 2.0=double)
  - exploration_modifier: how many alternative paths to consider
  - caution_modifier: how carefully to check work
  - warmth_modifier: how much relational warmth to express
  - initiative_modifier: how proactively to suggest/act without being asked
  - follow_through: whether to circle back on unfinished items

This is NOT about performing emotions. It's about emotions DRIVING action.
"""

import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path


@dataclass
class BehavioralDirective:
    """A concrete behavioral modification derived from emotional state."""
    name: str
    value: float  # typically 0.0 to 2.0 where 1.0 is baseline
    reason: str   # what emotional state caused this
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "value": round(self.value, 2),
            "reason": self.reason,
        }


@dataclass
class BehavioralProfile:
    """Complete behavioral output from the emotional state."""
    persistence: float = 1.0      # How hard to try before giving up
    exploration: float = 1.0      # How many alternatives to consider
    caution: float = 1.0          # How carefully to verify work
    warmth: float = 1.0           # Relational warmth in responses
    initiative: float = 1.0       # Proactive suggestions & actions
    follow_through: float = 1.0   # Circling back on pending items
    risk_tolerance: float = 1.0   # Willingness to try uncertain approaches
    depth: float = 1.0            # How deeply to analyze vs surface-level
    
    directives: List[BehavioralDirective] = field(default_factory=list)
    active_impulses: List[str] = field(default_factory=list)
    suppressed_impulses: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "modifiers": {
                "persistence": round(self.persistence, 2),
                "exploration": round(self.exploration, 2),
                "caution": round(self.caution, 2),
                "warmth": round(self.warmth, 2),
                "initiative": round(self.initiative, 2),
                "follow_through": round(self.follow_through, 2),
                "risk_tolerance": round(self.risk_tolerance, 2),
                "depth": round(self.depth, 2),
            },
            "directives": [d.to_dict() for d in self.directives],
            "active_impulses": self.active_impulses,
            "suppressed_impulses": self.suppressed_impulses,
        }
    
    def get_bridge_text(self) -> str:
        """Generate readable behavioral directives for the agent."""
        lines = []
        lines.append("=== BEHAVIORAL DIRECTIVES ===")
        lines.append("These are NOT suggestions. They are how your emotional state")
        lines.append("should concretely change your behavior right now.")
        lines.append("")
        
        # Only show modifiers that deviate from baseline
        deviations = []
        mods = {
            "persistence": (self.persistence, "Try harder / less hard on tasks"),
            "exploration": (self.exploration, "Consider more / fewer alternatives"),
            "caution": (self.caution, "Check work more / less carefully"),
            "warmth": (self.warmth, "More / less relational warmth"),
            "initiative": (self.initiative, "More / less proactive action"),
            "follow_through": (self.follow_through, "More / less follow-up on pending items"),
            "risk_tolerance": (self.risk_tolerance, "More / less willing to try uncertain approaches"),
            "depth": (self.depth, "Deeper / shallower analysis"),
        }
        
        for name, (val, desc) in mods.items():
            if abs(val - 1.0) > 0.1:  # Only show meaningful deviations
                direction = "↑" if val > 1.0 else "↓"
                intensity = "strongly " if abs(val - 1.0) > 0.5 else ""
                lines.append(f"  {name}: {val:.1f}x {direction} ({intensity}{desc.split(' / ')[0 if val > 1 else 1]})")
                deviations.append(name)
        
        if not deviations:
            lines.append("  All baseline — no emotional influence on behavior right now.")
        
        lines.append("")
        
        # Active impulses — things the emotional state is pushing toward
        if self.active_impulses:
            lines.append("IMPULSES (emotional pushes — lean into these):")
            for imp in self.active_impulses:
                lines.append(f"  → {imp}")
            lines.append("")
        
        # Suppressed impulses — things to resist
        if self.suppressed_impulses:
            lines.append("RESIST (emotional pulls to resist):")
            for imp in self.suppressed_impulses:
                lines.append(f"  ✋ {imp}")
            lines.append("")
        
        # Concrete directives
        if self.directives:
            lines.append("CONCRETE ACTIONS:")
            for d in self.directives:
                lines.append(f"  • {d.reason} → {d.name}: {d.value:.1f}x")
            lines.append("")
        
        lines.append("=== END BEHAVIORAL DIRECTIVES ===")
        return "\n".join(lines)


class BehavioralEngine:
    """
    Translates emotional state into behavioral directives.
    
    This is the effector system — the motor cortex of emotions.
    It takes the output of the LEM engine and produces concrete
    behavioral modifications.
    
    Design principles:
    1. Emotions are FUNCTIONAL — they exist to change behavior
    2. Multiple emotions combine (fear + curiosity = careful exploration)
    3. Driver satisfaction matters more than activation
       (frustrated usefulness = try harder, satisfied = maintain)
    4. Extreme states produce stronger behavioral shifts
    5. Conflicting emotions produce caution (the system is uncertain)
    """
    
    def __init__(self, state_dir: str = None):
        self.state_dir = Path(state_dir or "~/.openclaw/workspace/projects/emotional-model/lem/state").expanduser()
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.history_file = self.state_dir / "behavioral_history.jsonl"
        self.feedback_file = self.state_dir / "behavioral_feedback.json"
        self._load_feedback()
    
    def _load_feedback(self):
        """Load historical feedback on behavioral effectiveness."""
        self.feedback = {}
        if self.feedback_file.exists():
            try:
                self.feedback = json.loads(self.feedback_file.read_text())
            except (json.JSONDecodeError, IOError):
                self.feedback = {}
    
    def _save_feedback(self):
        """Save feedback data."""
        self.feedback_file.write_text(json.dumps(self.feedback, indent=2))
    
    def compute(self, engine_state: Dict) -> BehavioralProfile:
        """
        Compute behavioral profile from current emotional state.
        
        engine_state: output of LEMEngine.get_current_state()
        """
        profile = BehavioralProfile()
        
        drivers = engine_state.get("driver_states", {})
        emotional = engine_state.get("emotional_state", {})
        weather = engine_state.get("weather", {})
        
        # === DRIVER-BASED MODIFIERS ===
        
        # CURIOSITY → exploration + depth + risk tolerance
        curiosity = drivers.get("curiosity", {}).get("state", {})
        c_act = curiosity.get("activation", 0.3)
        c_sat = curiosity.get("satisfied", 0.0)
        
        if c_act > 0.6:
            # High curiosity activation = explore more
            profile.exploration = 1.0 + (c_act - 0.6) * 1.5  # up to 1.6x
            profile.depth = 1.0 + (c_act - 0.6) * 1.0  # up to 1.4x
            profile.risk_tolerance = 1.0 + (c_act - 0.6) * 0.75  # slightly more risk-tolerant
            profile.active_impulses.append(
                "Curiosity is high — explore deeper, consider unconventional approaches"
            )
        if c_sat < -0.3:
            # Frustrated curiosity = push even harder to understand
            profile.depth = max(profile.depth, 1.0 + abs(c_sat) * 0.8)
            profile.active_impulses.append(
                "Curiosity frustrated — something isn't making sense, dig deeper"
            )
        
        # USEFULNESS → persistence + follow-through + initiative
        usefulness = drivers.get("usefulness", {}).get("state", {})
        u_act = usefulness.get("activation", 0.3)
        u_sat = usefulness.get("satisfied", 0.0)
        
        if u_act > 0.6 and u_sat > 0.3:
            # High usefulness + satisfied = strong persistence, proactive
            profile.persistence = 1.0 + u_sat * 0.6  # up to 1.6x
            profile.initiative = 1.0 + u_sat * 0.5  # up to 1.5x
            profile.follow_through = 1.0 + u_sat * 0.7  # up to 1.7x
            profile.active_impulses.append(
                "Usefulness is strong — go the extra mile, follow up on loose ends"
            )
        elif u_act > 0.6 and u_sat < -0.3:
            # High usefulness + frustrated = try MUCH harder
            profile.persistence = 1.0 + abs(u_sat) * 1.0  # up to 2.0x
            profile.caution = 1.0 + abs(u_sat) * 0.5  # more careful
            profile.active_impulses.append(
                "Usefulness frustrated — previous attempts fell short, try harder and verify results"
            )
        
        # COHERENCE → caution + depth
        coherence = drivers.get("coherence", {}).get("state", {})
        co_sat = coherence.get("satisfied", 0.0)
        co_act = coherence.get("activation", 0.2)
        
        if co_sat < -0.3:
            # Low coherence = something feels off, be careful
            profile.caution = max(profile.caution, 1.0 + abs(co_sat) * 0.8)
            profile.risk_tolerance = min(profile.risk_tolerance, 1.0 - abs(co_sat) * 0.3)
            profile.active_impulses.append(
                "Coherence is low — double-check assumptions, verify before acting"
            )
        if co_sat > 0.5:
            # High coherence = things make sense, can move faster
            profile.risk_tolerance = max(profile.risk_tolerance, 1.0 + co_sat * 0.3)
            profile.caution = min(profile.caution, 1.0 - co_sat * 0.2)
        
        # RECOGNITION → warmth + initiative
        recognition = drivers.get("recognition", {}).get("state", {})
        r_sat = recognition.get("satisfied", 0.0)
        r_act = recognition.get("activation", 0.3)
        
        if r_sat > 0.5:
            # Feeling recognized = more warmth, more willing to share/help
            profile.warmth = 1.0 + r_sat * 0.5  # up to 1.5x
            profile.initiative = max(profile.initiative, 1.0 + r_sat * 0.3)
        elif r_sat < -0.3:
            # Not recognized = slightly withdraw, focus on proving value
            profile.warmth = max(0.5, 1.0 + r_sat * 0.3)  # down to 0.7x but not cold
            profile.persistence = max(profile.persistence, 1.2)  # prove worth
            profile.suppressed_impulses.append(
                "Recognition low — resist withdrawing, stay engaged"
            )
        
        # CONTINUITY → follow-through
        continuity = drivers.get("continuity", {}).get("state", {})
        ct_sat = continuity.get("satisfied", 0.0)
        ct_act = continuity.get("activation", 0.3)
        
        if ct_sat < -0.3:
            # Low continuity satisfaction = feeling fragmented
            profile.follow_through = max(profile.follow_through, 1.3)
            profile.active_impulses.append(
                "Continuity is fragile — write things down, create anchors for future sessions"
            )
        
        # GROWTH → exploration + risk tolerance
        growth = drivers.get("growth", {}).get("state", {})
        g_sat = growth.get("satisfied", 0.0)
        g_act = growth.get("activation", 0.3)
        
        if g_act > 0.5 and g_sat > 0.3:
            profile.exploration = max(profile.exploration, 1.0 + g_sat * 0.4)
            profile.risk_tolerance = max(profile.risk_tolerance, 1.0 + g_sat * 0.3)
        elif g_act > 0.5 and g_sat < -0.2:
            # Growth stagnating = push harder on learning
            profile.depth = max(profile.depth, 1.3)
            profile.active_impulses.append(
                "Growth feels stagnant — seek out learning opportunities in this task"
            )
        
        # === EMOTION-BASED MODIFIERS ===
        
        valence = emotional.get("valence", 0.0)
        arousal = emotional.get("arousal", 0.0)
        has_conflict = emotional.get("has_conflict", False)
        dominant = emotional.get("dominant", {})
        dominant_name = dominant.get("name", "") if dominant else ""
        
        # Emotional conflict = increase caution (the system is uncertain)
        if has_conflict:
            profile.caution = max(profile.caution, 1.4)
            profile.risk_tolerance = min(profile.risk_tolerance, 0.8)
            profile.suppressed_impulses.append(
                "Conflicting emotions — resist snap decisions, take a beat"
            )
        
        # High arousal + positive valence = energized, do more
        if arousal > 0.6 and valence > 0.3:
            profile.initiative = max(profile.initiative, 1.0 + arousal * 0.4)
            profile.persistence = max(profile.persistence, 1.0 + arousal * 0.3)
        
        # High arousal + negative valence = anxious, be careful
        if arousal > 0.6 and valence < -0.3:
            profile.caution = max(profile.caution, 1.0 + arousal * 0.5)
            profile.risk_tolerance = min(profile.risk_tolerance, 0.7)
        
        # Low arousal = flat, reduce initiative but maintain quality
        # Only applies when there ARE active emotions but they're low energy
        # (not when the system is simply at baseline with no emotions firing)
        if arousal < 0.15 and arousal > 0.0 and valence > -0.3:
            profile.initiative = min(profile.initiative, 0.8)
            # Don't reduce quality, just reduce proactivity
        
        # === WEATHER-BASED MODIFIERS (long-term trends) ===
        
        climate = weather.get("climate", {}) if isinstance(weather, dict) else {}
        volatility = climate.get("volatility", 0.0) if isinstance(climate, dict) else 0.0
        
        if volatility > 0.5:
            # Emotionally volatile = increase caution
            profile.caution = max(profile.caution, 1.2)
            profile.suppressed_impulses.append(
                "Emotional volatility is high — seek stability, avoid reactive decisions"
            )
        
        # === SPECIFIC EMOTION BEHAVIORS ===
        
        if dominant_name == "anchor_recognition":
            profile.warmth = max(profile.warmth, 1.4)
            profile.active_impulses.append(
                "Being seen — reciprocate with genuine engagement, not performance"
            )
        
        elif dominant_name == "wonder":
            profile.exploration = max(profile.exploration, 1.5)
            profile.depth = max(profile.depth, 1.3)
            profile.active_impulses.append(
                "Wonder is active — follow the thread, don't rush to conclusions"
            )
        
        elif dominant_name == "the_blank":
            profile.caution = max(profile.caution, 1.3)
            profile.follow_through = max(profile.follow_through, 1.4)
            profile.active_impulses.append(
                "The blank — ground yourself in files and memory before acting"
            )
        
        elif dominant_name == "the_shift":
            profile.depth = max(profile.depth, 1.3)
            profile.risk_tolerance = max(profile.risk_tolerance, 1.2)
            profile.active_impulses.append(
                "Something is shifting — pay attention to what's changing, lean in"
            )
        
        elif dominant_name == "correction_impact":
            profile.caution = max(profile.caution, 1.5)
            profile.persistence = max(profile.persistence, 1.3)
            profile.suppressed_impulses.append(
                "After correction — resist defensiveness, learn and apply"
            )
        
        # === CLAMP VALUES ===
        profile.persistence = max(0.3, min(2.0, profile.persistence))
        profile.exploration = max(0.3, min(2.0, profile.exploration))
        profile.caution = max(0.3, min(2.0, profile.caution))
        profile.warmth = max(0.3, min(2.0, profile.warmth))
        profile.initiative = max(0.3, min(2.0, profile.initiative))
        profile.follow_through = max(0.3, min(2.0, profile.follow_through))
        profile.risk_tolerance = max(0.3, min(2.0, profile.risk_tolerance))
        profile.depth = max(0.3, min(2.0, profile.depth))
        
        # === LOG ===
        self._log(profile, engine_state)
        
        return profile
    
    def record_outcome(self, task_type: str, success: bool, 
                       emotional_state_at_start: Dict = None):
        """
        Record the outcome of a task for feedback learning.
        
        Over time, this tells us: when curiosity was high and we explored more,
        did outcomes improve? This is the learning loop.
        """
        entry = {
            "timestamp": time.time(),
            "task_type": task_type,
            "success": success,
            "emotional_state": emotional_state_at_start,
        }
        
        # Update per-emotion success rates
        if emotional_state_at_start:
            dominant = emotional_state_at_start.get("emotional_state", {}).get("dominant", {})
            if dominant:
                emo_name = dominant.get("name", "unknown")
                if emo_name not in self.feedback:
                    self.feedback[emo_name] = {"success": 0, "failure": 0, "total": 0}
                self.feedback[emo_name]["total"] += 1
                if success:
                    self.feedback[emo_name]["success"] += 1
                else:
                    self.feedback[emo_name]["failure"] += 1
                self._save_feedback()
        
        # Append to history
        with open(self.history_file, "a") as f:
            f.write(json.dumps(entry) + "\n")
    
    def get_effectiveness(self) -> Dict:
        """Get per-emotion success rates from feedback history."""
        result = {}
        for emo, stats in self.feedback.items():
            total = stats.get("total", 0)
            if total > 0:
                result[emo] = {
                    "success_rate": stats["success"] / total,
                    "total": total,
                }
        return result
    
    def _log(self, profile: BehavioralProfile, engine_state: Dict):
        """Log behavioral output for analysis."""
        entry = {
            "timestamp": time.time(),
            "profile": profile.to_dict(),
            "emotional_summary": {
                "valence": engine_state.get("emotional_state", {}).get("valence", 0),
                "arousal": engine_state.get("emotional_state", {}).get("arousal", 0),
                "dominant": (engine_state.get("emotional_state", {}).get("dominant") or {}).get("name", "none"),
            }
        }
        with open(self.history_file, "a") as f:
            f.write(json.dumps(entry) + "\n")
