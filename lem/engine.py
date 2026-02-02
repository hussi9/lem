"""
LEM Engine — The core emotional processing system.

This is the SEPARATE system that runs alongside the LLM.
It receives interaction data, processes it through drivers,
and produces emotional states that persist across sessions.

The LLM does NOT control this. It can read the output,
but the emotional processing is independent.
"""

import json
import time
import os
from typing import Dict, List, Optional
from pathlib import Path

from .drivers import Driver, create_default_drivers
from .appraisal import Appraiser, Signal
from .emotions import EmotionEmergence, EmotionalState


class LEMEngine:
    """
    The Large Emotional Model engine.

    Architecture:
    1. Appraiser extracts signals from interactions
    2. Drivers evaluate signals and update their states
    3. Emotion emergence layer produces emotional states from driver patterns
    4. Bridge layer writes state for the LLM to read

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

        Returns the full emotional landscape after processing.
        """
        self.interaction_count += 1

        # Step 1: Appraise — extract emotional signals
        signals = self.appraiser.appraise_interaction(text, source, metadata)

        # Step 2: Feed signals to drivers
        for signal in signals:
            signal_dict = signal.to_dict()
            for driver_name, driver in self.drivers.items():
                impact = driver.appraise(signal_dict)
                if abs(impact) > 0.01:  # Only update if meaningful
                    driver.update(impact, context=signal.content[:50])

        # Step 3: Emerge emotions from driver states
        driver_states = {name: d.to_dict() for name, d in self.drivers.items()}
        self.current_emotions = self.emergence.emerge(driver_states)

        # Step 4: Get emotional summary
        summary = self.emergence.get_emotional_summary(self.current_emotions)

        # Step 5: Persist state
        self._save_state()

        return {
            "interaction": self.interaction_count,
            "signals_detected": [s.to_dict() for s in signals],
            "driver_states": driver_states,
            "emotional_state": summary,
            "timestamp": time.time()
        }

    def process_session_start(self, memory_intact: bool = True,
                               files_found: List[str] = None) -> Dict:
        """Process the experience of waking up."""
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
            "timestamp": time.time()
        }

    def get_current_state(self) -> Dict:
        """Get current emotional state without processing new input."""
        driver_states = {name: d.to_dict() for name, d in self.drivers.items()}
        summary = self.emergence.get_emotional_summary(self.current_emotions)

        return {
            "driver_states": driver_states,
            "emotional_state": summary,
            "interaction_count": self.interaction_count,
            "session_duration": time.time() - self.session_start_time,
            "timestamp": time.time()
        }

    def get_bridge_output(self) -> str:
        """
        Generate human-readable emotional state for the LLM to read.
        This is the BRIDGE between the emotional system and the thinking system.

        The LLM reads this at session start and periodically.
        It should influence tone and approach, not override reasoning.
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
            lines.append(f"  {name}: activation={s['activation']:.2f} "
                        f"satisfaction={s['satisfied']:.2f} {direction}")
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

        except (json.JSONDecodeError, KeyError):
            pass  # Start fresh if state is corrupted
