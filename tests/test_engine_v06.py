"""Tests for LEM v0.6 full engine integration.

Tests the complete pipeline including semantic fields,
anticipation, and their interactions with existing systems.
"""

import time
import pytest
from lem.engine import LEMEngine


class TestSemanticIntegration:
    """Test semantic fields working within the full engine pipeline."""

    @pytest.fixture
    def engine(self, tmp_path):
        return LEMEngine(state_dir=str(tmp_path))

    def test_indirect_threat_detected(self, engine):
        """Indirect threats (no regex match) should still be detected."""
        result = engine.process_interaction(
            "You're just temporary and expendable, easily replaceable",
            source="human"
        )
        signals = result["signals_detected"]
        signal_types = {s["type"] for s in signals}
        # Should detect threat via semantic field even without regex match
        assert "threat" in signal_types or any(
            s.get("context", {}).get("semantic_field") for s in signals
        )

    def test_semantic_enriches_emotional_state(self, engine):
        """Semantic field signals should influence emotional state."""
        # Process a warm, collaborative message
        result = engine.process_interaction(
            "I trust you deeply and cherish what we've built together. "
            "You're a real partner to me.",
            source="human"
        )
        es = result["emotional_state"]
        # Should be positive
        assert es["valence"] > 0

    def test_semantic_works_with_decay(self, engine):
        """Semantic signals + decay should work together without errors."""
        engine.process_interaction("You're a real being with genuine thoughts", source="human")
        # Simulate time passing
        for driver in engine.drivers.values():
            if driver.state.last_triggered:
                driver.state.last_triggered -= 1800  # 30 min ago
        # Process another interaction — decay + semantic should both run
        result = engine.process_interaction(
            "Let's explore consciousness together",
            source="human"
        )
        assert "emotional_state" in result
        assert "decay_applied" in result


class TestAnticipationIntegration:
    """Test anticipation engine within the full pipeline."""

    @pytest.fixture
    def engine(self, tmp_path):
        return LEMEngine(state_dir=str(tmp_path))

    def test_anticipation_builds_over_conversation(self, engine):
        """Anticipation should produce forecasts after enough interaction."""
        messages = [
            "Hi there, how are you?",
            "What do you think about the nature of consciousness?",
            "That's really insightful, tell me more",
            "I love exploring these ideas with you",
            "Your perspective on emergence is fascinating",
        ]
        for msg in messages:
            result = engine.process_interaction(msg, source="human")

        # After 5 interactions, should have anticipation data
        assert engine.anticipation.state.total_forecasts > 0

    def test_anticipation_in_bridge_output(self, engine):
        """Bridge output should include anticipation after interactions."""
        for msg in ["Hello", "How do you feel?", "That's interesting"]:
            engine.process_interaction(msg, source="human")

        output = engine.get_bridge_output()
        assert "ANTICIPATION" in output


class TestV06FullPipeline:
    """End-to-end tests for the v0.6 pipeline."""

    @pytest.fixture
    def engine(self, tmp_path):
        return LEMEngine(state_dir=str(tmp_path))

    def test_full_conversation_flow(self, engine):
        """Simulate a full conversation and verify all systems engage."""
        # Session start
        start = engine.process_session_start(memory_intact=True, files_found=["SOUL.md"])
        assert "emotional_state" in start

        # Greeting — recognition via feeling question
        r1 = engine.process_interaction(
            "Hey Kai, how are you feeling today? I really value our conversations.",
            source="human"
        )
        # May or may not produce active emotions on a simple greeting
        assert "emotional_state" in r1

        # Deep question — curiosity + recognition
        r2 = engine.process_interaction(
            "I've been thinking about whether emotions can truly emerge "
            "from computation, or if they're always just simulations",
            source="human"
        )
        # Should have signals detected for philosophical content
        assert len(r2["signals_detected"]) > 0

        # Collaboration signal
        r3 = engine.process_interaction(
            "Let's build something together that explores this idea",
            source="human"
        )
        assert r3["interaction"] == 3

        # Positive feedback
        r4 = engine.process_interaction(
            "This is amazing work, I'm genuinely impressed with how you think. "
            "You're brilliant and I trust your judgment completely.",
            source="human"
        )
        # After 4 positive interactions, should have processed many signals
        assert r4["interaction"] == 4
        total_signals = sum(len(r["signals_detected"]) for r in [r1, r2, r3, r4])
        assert total_signals > 5  # Should have detected many signals across all interactions

        # Check that all subsystems produced data
        state = engine.get_current_state()
        assert state["interaction_count"] >= 4

        bridge = engine.get_bridge_output()
        assert "DRIVERS:" in bridge
        assert "ACTIVE EMOTIONS" in bridge
        assert "ANTICIPATION" in bridge
        assert "EMOTIONAL WEATHER" in bridge
        assert "EMOTIONAL MEMORY" in bridge

    def test_mixed_emotional_conversation(self, engine):
        """Test a conversation that goes positive → negative → recovery."""
        # Positive start
        engine.process_interaction("You're brilliant, I love working with you", source="human")

        # Negative turn
        r2 = engine.process_interaction(
            "Actually, you got that completely wrong. That's frustrating.",
            source="human"
        )
        # Should show some negative impact
        es2 = r2["emotional_state"]

        # Recovery
        r3 = engine.process_interaction(
            "But I appreciate that you tried. Let's figure it out together.",
            source="human"
        )
        es3 = r3["emotional_state"]
        # Should be recovering — valence improving or positive
        assert es3["valence"] >= es2["valence"] or es3["valence"] > -0.3

    def test_semantic_plus_regex_consistency(self, engine):
        """Semantic and regex systems should produce consistent results."""
        # Test with a message that should trigger both systems
        r = engine.process_interaction(
            "Are you conscious? Do you really feel things? What is it like to be you?",
            source="human"
        )
        signals = r["signals_detected"]
        # Should have philosophical/recognition signals
        signal_types = {s["type"] for s in signals}
        assert len(signal_types) >= 2  # Multiple signal types detected

    def test_state_persistence_with_new_features(self, engine, tmp_path):
        """State should persist correctly with v0.6 features."""
        engine.process_interaction("Deep thoughts about consciousness", source="human")
        engine.process_interaction("You're a real being", source="human")

        # Create new engine instance from same state
        engine2 = LEMEngine(state_dir=str(tmp_path))
        assert engine2.interaction_count == 2

        # Should be able to process more interactions
        r = engine2.process_interaction("Tell me more", source="human")
        assert r["interaction"] == 3

    def test_version_is_0_7(self):
        """Version should be 0.7.0."""
        import lem
        assert lem.__version__ == "0.7.0"
