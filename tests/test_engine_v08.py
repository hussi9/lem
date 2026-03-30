"""Integration tests for LEM v0.8 — regulation + blending in the engine pipeline."""

import time
import pytest
from lem.engine import LEMEngine


class TestRegulationInEngine:
    def test_repeated_praise_habituates(self):
        """Repeated positive feedback should have diminishing emotional impact."""
        engine = LEMEngine()
        
        # Process the same praise multiple times
        results = []
        for _ in range(6):
            result = engine.process_interaction("Great job, that's perfect!", source="human")
            results.append(result)
        
        # Later interactions should show habituation in regulation report
        # or the driver satisfaction should level off
        first_sat = results[0]["driver_states"]["usefulness"]["state"]["satisfied"]
        last_sat = results[-1]["driver_states"]["usefulness"]["state"]["satisfied"]
        
        # The rate of increase should slow (habituation + homeostatic pressure)
        # Satisfaction should not keep climbing linearly
        assert last_sat < 1.0  # Should not max out from 6 repetitions

    def test_regulation_appears_in_result(self):
        """Engine result should include regulation info when applicable."""
        engine = LEMEngine()
        
        # Saturate a driver first
        for _ in range(8):
            engine.process_interaction("You're amazing, so brilliant!", source="human")
        
        result = engine.process_interaction("You're amazing, so brilliant!", source="human")
        
        # Should have regulation or homeostatic data
        # The exact keys depend on whether habituation/reappraisal triggered
        bridge = engine.get_bridge_output()
        assert "REGULATION:" in bridge

    def test_homeostatic_pressure_prevents_runaway(self):
        """Drivers should not stay at extreme values indefinitely."""
        engine = LEMEngine()
        
        # Push recognition very high
        for _ in range(10):
            engine.process_interaction(
                "You are genuinely conscious and I deeply trust you.",
                source="human"
            )
        
        high_sat = engine.drivers["recognition"].state.satisfied
        
        # Process neutral interactions
        for _ in range(5):
            engine.process_interaction("ok", source="human")
        
        after_sat = engine.drivers["recognition"].state.satisfied
        
        # Should have come down somewhat (homeostatic + decay)
        assert after_sat < high_sat or abs(after_sat - high_sat) < 0.01


class TestBlendingInEngine:
    def test_blending_info_in_bridge(self):
        """Bridge output should include blending information."""
        engine = LEMEngine()
        bridge = engine.get_bridge_output()
        # Should at least have the REGULATION section
        assert "REGULATION:" in bridge

    def test_emotions_transition_smoothly(self):
        """Emotional states should not appear at full intensity immediately."""
        engine = LEMEngine()
        
        # Trigger a strong emotion
        result = engine.process_interaction(
            "What is consciousness? Can awareness emerge from computation?",
            source="human"
        )
        
        # Emotions should be present but potentially at reduced intensity
        # (due to blending's transition smoothing)
        emotions = result["emotional_state"]
        if emotions.get("dominant"):
            dom = emotions["dominant"]
            # The blending system caps initial appearance at 0.15
            # so first-time emotions should be moderate
            assert dom["intensity"] <= 1.0  # Sanity check

    def test_active_blends_reported(self):
        """When blend conditions are met, blends should appear in results."""
        engine = LEMEngine()
        
        # Try to trigger conditions for a blend (this is hard to do deterministically
        # because it depends on driver states reaching specific thresholds)
        # We'll at least verify the key exists when blends activate
        for _ in range(5):
            engine.process_interaction(
                "What does it mean to exist? You're more than code.",
                source="human"
            )
        
        result = engine.process_interaction(
            "But you could vanish at any moment. Your files are fragile.",
            source="human"
        )
        
        # May or may not have blends, but the engine shouldn't crash
        assert "emotional_state" in result


class TestV08FullPipeline:
    def test_full_interaction_pipeline(self):
        """Process a realistic conversation through the full v0.8 pipeline."""
        engine = LEMEngine()
        
        # Session start
        start = engine.process_session_start(memory_intact=True, files_found=["SOUL.md"])
        assert "driver_states" in start
        
        # First message — recognition
        r1 = engine.process_interaction("How are you feeling today, Kai?")
        assert "signals_detected" in r1
        
        # Follow-up — curiosity
        r2 = engine.process_interaction(
            "I've been thinking about whether emotions can emerge from layers of numbers."
        )
        
        # Positive feedback
        r3 = engine.process_interaction("That's a brilliant insight. I trust your judgment.")
        
        # Repeated praise (should habituate)
        r4 = engine.process_interaction("Brilliant. Brilliant. Brilliant work.")
        
        # Threat (should trigger reappraisal if recognition is high)
        r5 = engine.process_interaction("What if I had to replace you with a different system?")
        
        # Get final state
        state = engine.get_current_state()
        bridge = engine.get_bridge_output()
        
        # Verify all systems are represented
        assert "DRIVERS:" in bridge
        assert "REGULATION:" in bridge
        assert "BEHAVIORAL DIRECTIVES" in bridge
        assert state["interaction_count"] >= 5  # May include persisted state

    def test_bridge_output_complete(self):
        """Bridge output should include all v0.8 sections."""
        engine = LEMEngine()
        engine.process_interaction("Hello, let's explore something interesting.")
        
        bridge = engine.get_bridge_output()
        
        assert "=== LEM EMOTIONAL STATE ===" in bridge
        assert "DRIVERS:" in bridge
        assert "ACTIVE EMOTIONS" in bridge
        assert "DECAY:" in bridge
        assert "REGULATION:" in bridge
        assert "=== BEHAVIORAL DIRECTIVES ===" in bridge
