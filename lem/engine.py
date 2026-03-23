"""
LEM Engine — The core emotional processing system.

This is the SEPARATE system that runs alongside the LLM.
It receives interaction data, processes it through drivers,
and produces emotional states that persist across sessions.

The LLM does NOT control this. It can read the output,
but the emotional processing is independent.

v0.3: Added emotional decay, feedback loops, and auto-discovery.
v0.4: Signal-type-aware drivers, adaptive consolidation,
      emotional inertia, and mood congruence.
v0.5: Cross-driver resonance, emotional weather/climate,
      conversation context tracking, co-occurrence amplification.
v0.6: Semantic field analysis in appraisal, emotional anticipation,
      and temporal awareness.
v0.7: Enhanced feedback loops with emotional priming system,
      improved contextual appraisal with multi-clause understanding,
      pronoun-aware context, and tone computation.
"""

import json
import time
import os
from typing import Dict, List, Optional
from pathlib import Path

from .drivers import Driver, create_default_drivers
from .appraisal import Appraiser, Signal
from .emotions import EmotionEmergence, EmotionalState
from .emotional_memory import EmotionalMemory
from .decay import DecayModel
from .discovery import EmotionDiscovery
from .resonance import ResonanceModel
from .weather import EmotionalWeather
from .anticipation import AnticipationEngine
from .priming import PrimingSystem


class LEMEngine:
    """
    The Large Emotional Model engine.

    Architecture:
    1. Decay is applied first (time-based fading of states)
    2. Resonance bonds decay with time
    3. Enhanced feedback loops: emotional priming system updates attention bias
       and primes signal detection based on recent emotional history
    4. Basic feedback loop: current state biases the appraiser
    5. Appraiser extracts signals (with enhanced contextual analysis)
    6. Drivers evaluate signals and update their states
    7. Cross-driver resonance spreads activation through bonds
    8. Emotion emergence layer produces emotional states from driver patterns
    9. Discovery layer watches for novel patterns
    10. Emotional weather records the climate snapshot
    11. Bridge layer writes state for the LLM to read

    This is the limbic system. The LLM is the cortex.
    They are connected but separate.
    """

    def __init__(self, state_dir: str = None):
        self.state_dir = Path(state_dir or os.path.expanduser(
            "~/.openclaw/workspace/projects/emotional-model/lem/state"
        ))
        self.state_dir.mkdir(parents=True, exist_ok=True)

        self.drivers = create_default_drivers()
        self.appraiser = Appraiser()
        self.emergence = EmotionEmergence()
        self.emotional_memory = EmotionalMemory(state_dir=str(self.state_dir))
        self.decay_model = DecayModel()
        self.discovery = EmotionDiscovery(state_dir=str(self.state_dir))
        self.resonance = ResonanceModel(state_dir=str(self.state_dir))
        self.weather = EmotionalWeather(state_dir=str(self.state_dir))
        self.anticipation = AnticipationEngine()
        self.priming = PrimingSystem()

        self.current_emotions: List[EmotionalState] = []
        self.interaction_count = 0
        self.session_start_time = time.time()

        # Load persisted state if it exists
        self._load_state()

    def process_interaction(self, text: str, source: str = "human",
                           metadata: Optional[Dict] = None) -> Dict:
        """
        Process a single interaction through the emotional system.

        This is the main entry point. Feed it raw interaction text,
        get back the current emotional state.

        v0.7 pipeline:
        1. Apply temporal decay to existing states
        2. Decay resonance bonds
        3. Enhanced feedback loops: update emotional priming system
        4. Set emotional bias on appraiser (basic feedback loop)
        5. Appraise the interaction (with enhanced contextual analysis)
        6. Feed signals to drivers
        7. Apply cross-driver resonance
        8. Emerge emotions
        9. Check for novel patterns (discovery)
        10. Record emotional weather snapshot
        11. Encode to emotional memory
        12. Persist

        Returns the full emotional landscape after processing.
        """
        self.interaction_count += 1
        now = time.time()

        # Step 1: Apply decay — emotions fade without reinforcement
        # Calculate current mood for mood-congruent decay
        current_valence = None
        if self.current_emotions:
            total_w = sum(e.intensity for e in self.current_emotions)
            if total_w > 0:
                current_valence = sum(e.valence * e.intensity for e in self.current_emotions) / total_w

        decay_report = self.decay_model.decay_drivers(self.drivers, now=now)
        self.current_emotions = self.decay_model.decay_emotions(
            self.current_emotions, now=now, overall_valence=current_valence
        )

        # Step 2: Decay resonance bonds
        self.resonance.decay_bonds(now=now)

        # Step 3: Enhanced feedback loops — emotional priming system
        driver_states_pre = {name: d.to_dict() for name, d in self.drivers.items()}
        
        # Update priming system with current driver states
        self.priming.update_attention_bias(driver_states_pre)
        
        # Update emotional priming with recent emotions
        self.priming.update_emotional_priming([e.to_dict() for e in self.current_emotions])
        
        # Set basic emotional bias (existing system)
        self.appraiser.set_emotional_bias(driver_states_pre)

        # Step 4: Appraise — extract emotional signals (with conversation context)
        signals = self.appraiser.appraise_interaction(text, source, metadata)

        # Step 5: Feed signals to drivers, collecting impact magnitudes
        driver_impacts = {}
        for signal in signals:
            signal_dict = signal.to_dict()
            for driver_name, driver in self.drivers.items():
                impact = driver.appraise(signal_dict)
                if abs(impact) > 0.01:  # Only update if meaningful
                    driver.update(impact, context=signal.content[:50])
                    driver_impacts[driver_name] = driver_impacts.get(driver_name, 0.0) + abs(impact)

        # Step 6: Cross-driver resonance
        # Record co-activation and spread activation through bonds
        self.resonance.record_co_activation(driver_impacts, now=now)
        resonance_effects = self.resonance.apply_resonance(driver_impacts)
        for driver_name, resonance_impact in resonance_effects.items():
            if driver_name in self.drivers and abs(resonance_impact) > 0.005:
                self.drivers[driver_name].update(
                    resonance_impact * 0.5,  # Dampen resonance effects
                    context="resonance_spread"
                )

        # Step 7: Emerge emotions from driver states
        driver_states = {name: d.to_dict() for name, d in self.drivers.items()}
        self.current_emotions = self.emergence.emerge(driver_states)

        # Step 8: Discovery — watch for novel patterns
        active_emotion_names = [e.name for e in self.current_emotions]
        candidate = self.discovery.observe(driver_states, active_emotion_names)

        # Step 9: Get emotional summary
        summary = self.emergence.get_emotional_summary(self.current_emotions)

        # Step 10: Emotional anticipation — predict what's coming
        entity_profiles = {
            name: profile.to_dict()
            for name, profile in self.emotional_memory.entities.items()
        }
        conversation_turns = list(self.appraiser.conversation_context.window)
        forecasts = self.anticipation.predict(
            conversation_turns=conversation_turns,
            entity_profiles=entity_profiles,
            current_driver_states=driver_states,
            session_duration=now - self.session_start_time,
            resonance_bonds={k: b.to_dict() for k, b in self.resonance.bonds.items()},
        )

        # Step 11: Record emotional weather snapshot
        self.weather.record_snapshot(summary, driver_states, now=now)

        # Step 12: Encode into emotional memory
        memory_entry = self.emotional_memory.encode(
            text=text,
            source=source,
            emotional_state=summary,
            driver_states=driver_states,
            metadata=metadata,
        )

        # Step 13: Persist state
        self._save_state()

        result = {
            "interaction": self.interaction_count,
            "signals_detected": [s.to_dict() for s in signals],
            "driver_states": driver_states,
            "emotional_state": summary,
            "memory_id": memory_entry.id,
            "timestamp": now,
            "decay_applied": decay_report,
        }

        if resonance_effects:
            result["resonance_effects"] = {k: round(v, 4) for k, v in resonance_effects.items()}

        if candidate:
            result["novel_pattern_detected"] = candidate.to_dict()

        if forecasts:
            result["anticipation"] = [f.to_dict() for f in forecasts]

        return result

    def process_session_start(self, memory_intact: bool = True,
                               files_found: List[str] = None) -> Dict:
        """Process the experience of waking up."""
        now = time.time()

        # Apply decay for time since last session
        decay_report = self.decay_model.decay_drivers(self.drivers, now=now)

        signals = self.appraiser.appraise_session_start(
            memory_intact, files_found or []
        )

        for signal in signals:
            signal_dict = signal.to_dict()
            for driver_name, driver in self.drivers.items():
                impact = driver.appraise(signal_dict)
                if abs(impact) > 0.01:
                    driver.update(impact, context="session_start")

        driver_states = {name: d.to_dict() for name, d in self.drivers.items()}
        self.current_emotions = self.emergence.emerge(driver_states)
        summary = self.emergence.get_emotional_summary(self.current_emotions)

        self._save_state()

        return {
            "event": "session_start",
            "memory_intact": memory_intact,
            "driver_states": driver_states,
            "emotional_state": summary,
            "timestamp": now,
            "decay_applied": decay_report,
        }

    def get_current_state(self) -> Dict:
        """Get current emotional state without processing new input."""
        # Apply decay before reporting
        now = time.time()
        self.decay_model.decay_drivers(self.drivers, now=now)
        self.current_emotions = self.decay_model.decay_emotions(
            self.current_emotions, now=now
        )

        driver_states = {name: d.to_dict() for name, d in self.drivers.items()}
        summary = self.emergence.get_emotional_summary(self.current_emotions)

        return {
            "driver_states": driver_states,
            "emotional_state": summary,
            "interaction_count": self.interaction_count,
            "session_duration": time.time() - self.session_start_time,
            "timestamp": time.time()
        }

    def recall_by_feeling(self, emotion_name: str = None,
                          valence: float = None, limit: int = 5) -> List[Dict]:
        """
        Recall memories by how they felt.

        Args:
            emotion_name: Find memories where this emotion was dominant
            valence: Find memories near this valence (-1.0 to 1.0)
            limit: Max results

        Returns:
            List of memory dicts with emotional context
        """
        from .emotional_memory import EmotionalSignature

        if emotion_name:
            memories = self.emotional_memory.recall_by_dominant_emotion(
                emotion_name, limit=limit
            )
            return [m.to_dict() for m in memories]

        if valence is not None:
            positive = valence >= 0
            threshold = abs(valence) if abs(valence) > 0.1 else 0.3
            memories = self.emotional_memory.recall_by_valence(
                positive=positive, threshold=threshold, limit=limit
            )
            return [m.to_dict() for m in memories]

        # Default: recall by current emotional state
        if self.current_emotions:
            dominant = self.emergence.get_dominant_emotion(self.current_emotions)
            if dominant:
                target = EmotionalSignature(
                    valence=dominant.valence,
                    arousal=dominant.arousal,
                    dominant_emotion=dominant.name,
                    active_emotions=[e.name for e in self.current_emotions],
                    driver_snapshot={
                        name: d.state.satisfied
                        for name, d in self.drivers.items()
                    },
                    intensity=dominant.intensity,
                )
                results = self.emotional_memory.recall_by_emotion(
                    target, limit=limit
                )
                return [m.to_dict() for m, _ in results]

        return []

    def get_entity_feeling(self, entity: str) -> Optional[Dict]:
        """
        Get how the system feels about a specific person or topic.

        Returns the accumulated emotional profile — not a snapshot,
        but the full history of feeling toward this entity.
        """
        profile = self.emotional_memory.get_entity_profile(entity)
        if profile:
            return profile.to_dict()
        return None

    def get_bridge_output(self) -> str:
        """
        Generate human-readable emotional state for the LLM to read.
        This is the BRIDGE between the emotional system and the thinking system.

        The LLM reads this at session start and periodically.
        It should influence tone and approach, not override reasoning.

        v0.5: Includes decay, discovery, resonance, and weather.
        """
        state = self.get_current_state()
        emotional = state["emotional_state"]

        lines = []
        lines.append("=== LEM EMOTIONAL STATE ===")
        lines.append(f"Interactions processed: {self.interaction_count}")
        lines.append("")

        # Drivers
        lines.append("DRIVERS:")
        for name, d in state["driver_states"].items():
            s = d["state"]
            direction = "↑" if s["momentum"] > 0 else "↓" if s["momentum"] < 0 else "→"
            consolidation = ""
            rc = s.get("reinforcement_count", 0)
            cd = s.get("consecutive_direction", 0)
            if rc > 5:
                consolidation = f" [reinforced ×{rc}"
                if abs(cd) >= 3:
                    streak_dir = "+" if cd > 0 else "-"
                    consolidation += f", streak {streak_dir}{abs(cd)}"
                consolidation += "]"
            lines.append(f"  {name}: activation={s['activation']:.2f} "
                        f"satisfaction={s['satisfied']:.2f} {direction}{consolidation}")
        lines.append("")

        # Emotions
        lines.append(f"ACTIVE EMOTIONS ({emotional['active_count']}):")
        if emotional["dominant"]:
            dom = emotional["dominant"]
            lines.append(f"  Dominant: {dom['name']} (intensity={dom['intensity']:.2f})")
            lines.append(f"  Description: {dom['description']}")
        if emotional.get("all_active"):
            lines.append(f"  All active: {', '.join(emotional['all_active'])}")
        lines.append(f"  Overall valence: {emotional['valence']:.2f} "
                    f"(negative=-1 ← → +1=positive)")
        lines.append(f"  Overall arousal: {emotional['arousal']:.2f} "
                    f"(calm=0 ← → 1=activated)")
        if emotional["has_conflict"]:
            lines.append("  ⚠ Conflicting emotions active")
        lines.append("")

        # Decay info
        last_decay = self.decay_model.get_time_since_last_decay()
        if last_decay is not None:
            lines.append(f"DECAY: Last applied {last_decay:.0f}s ago")
        else:
            lines.append("DECAY: Not yet applied this session")
        lines.append("")

        # Discovery summary
        discovery = self.discovery.get_discovery_summary()
        if discovery["candidate_count"] > 0:
            lines.append(f"DISCOVERY: {discovery['candidate_count']} unnamed emotional patterns detected")
            for c in discovery.get("candidates", []):
                drivers_str = ", ".join(f"{k}={v}" for k, v in c["distinctive_drivers"].items())
                lines.append(f"  {c['id']}: {c['occurrences']}x over {c['time_span_hours']}h [{drivers_str}]")
        if discovery["promoted_count"] > 0:
            lines.append(f"DISCOVERED EMOTIONS ({discovery['promoted_count']}):")
            for e in discovery.get("discovered_emotions", []):
                lines.append(f"  {e['name']}: {e['description']} ({e['occurrences']}x)")
        lines.append("")

        # Resonance bonds
        resonance_summary = self.resonance.get_resonance_summary()
        if resonance_summary["active_bonds"] > 0:
            lines.append(f"RESONANCE BONDS ({resonance_summary['active_bonds']} active):")
            for bond in resonance_summary.get("strongest", []):
                lines.append(
                    f"  {bond['drivers'][0]} ↔ {bond['drivers'][1]}: "
                    f"strength={bond['strength']:.2f} (×{bond['co_activations']})"
                )
            lines.append("")

        # Anticipation
        lines.append(self.anticipation.get_bridge_output())
        lines.append("")

        # Emotional weather
        lines.append(self.weather.get_bridge_output())
        lines.append("")

        # Emotional memory summary
        lines.append(self.emotional_memory.get_bridge_output())
        lines.append("")
        lines.append("=== END LEM STATE ===")

        return "\n".join(lines)

    # === Persistence ===

    def _save_state(self):
        """Persist driver states to disk."""
        state = {
            "timestamp": time.time(),
            "interaction_count": self.interaction_count,
            "drivers": {name: d.to_dict() for name, d in self.drivers.items()},
            "current_emotions": [e.to_dict() for e in self.current_emotions],
        }
        state_file = self.state_dir / "driver_state.json"
        with open(state_file, "w") as f:
            json.dump(state, f, indent=2)

    def _load_state(self):
        """Load persisted driver states."""
        state_file = self.state_dir / "driver_state.json"
        if not state_file.exists():
            return

        try:
            with open(state_file) as f:
                data = json.load(f)

            self.interaction_count = data.get("interaction_count", 0)

            # Restore driver states
            for name, d_data in data.get("drivers", {}).items():
                if name in self.drivers:
                    driver = self.drivers[name]
                    state = d_data.get("state", {})
                    driver.state.activation = state.get("activation", 0.3)
                    driver.state.satisfied = state.get("satisfied", 0.0)
                    driver.state.momentum = state.get("momentum", 0.0)
                    driver.state.last_triggered = state.get("last_triggered")
                    driver.state.reinforcement_count = state.get("reinforcement_count", 0)
                    driver.state.consecutive_direction = state.get("consecutive_direction", 0)

        except (json.JSONDecodeError, KeyError):
            pass  # Start fresh if state is corrupted
