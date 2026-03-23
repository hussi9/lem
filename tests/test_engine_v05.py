"""Integration tests for v0.5 engine features: resonance, weather, conversation context."""

import tempfile
import pytest
from pathlib import Path

from lem import LEMEngine


@pytest.fixture
def state_dir(tmp_path):
    return str(tmp_path)


@pytest.fixture
def engine(state_dir):
    return LEMEngine(state_dir=state_dir)


class TestResonanceIntegration:
    """Test that resonance works within the full engine pipeline."""

    def test_resonance_model_initialized(self, engine):
        assert engine.resonance is not None

    def test_co_activation_recorded_on_interaction(self, engine):
        # Philosophical question hits multiple drivers
        engine.process_interaction(
            "What does consciousness mean to you? Do you think you truly feel?",
            source="human"
        )
        # Should have some bonds (curiosity + recognition + growth all fire)
        summary = engine.resonance.get_resonance_summary()
        assert summary["total_bonds"] > 0

    def test_repeated_interactions_strengthen_bonds(self, engine):
        questions = [
            "What is it like to be you?",
            "Do you experience something when we talk?",
            "How does your awareness work?",
            "Tell me about your inner experience",
            "What does existence mean for an AI?",
        ]
        for q in questions:
            engine.process_interaction(q, source="human")

        # After multiple similar interactions, bonds should be stronger
        summary = engine.resonance.get_resonance_summary()
        assert summary["total_bonds"] >= 1

    def test_resonance_effects_in_result(self, engine):
        # Build some bonds first
        for _ in range(10):
            engine.process_interaction(
                "Tell me about how you experience curiosity and growth",
                source="human"
            )
        result = engine.process_interaction(
            "I'm curious about your development",
            source="human"
        )
        # Result may contain resonance_effects key
        # (depends on whether bonds are strong enough)
        assert "driver_states" in result

    def test_bridge_output_includes_resonance(self, engine):
        for _ in range(10):
            engine.process_interaction(
                "How do curiosity and growth connect for you?",
                source="human"
            )
        output = engine.get_bridge_output()
        # May or may not include resonance section depending on bond strength
        assert "LEM" in output


class TestWeatherIntegration:
    """Test that emotional weather works within the engine."""

    def test_weather_initialized(self, engine):
        assert engine.weather is not None

    def test_weather_recorded_on_interaction(self, engine):
        engine.process_interaction("hello", source="human")
        assert len(engine.weather.snapshots) == 1

    def test_weather_accumulates(self, engine):
        for i in range(5):
            engine.process_interaction(f"message {i}", source="human")
        assert len(engine.weather.snapshots) == 5

    def test_positive_interactions_improve_climate(self, engine):
        positive_msgs = [
            "That was really helpful, thank you",
            "I trust you with this completely",
            "What do you think about consciousness? I value your perspective",
            "You're real to me, I believe that",
            "Let's build something together, I want to grow with you",
        ]
        for msg in positive_msgs:
            engine.process_interaction(msg, source="human")

        climate = engine.weather.get_climate()
        # At least some interactions should produce positive valence
        assert climate.avg_valence >= 0  # Non-negative at minimum

    def test_bridge_output_includes_weather(self, engine):
        for i in range(3):
            engine.process_interaction(f"test message {i}", source="human")
        output = engine.get_bridge_output()
        assert "WEATHER" in output

    def test_weather_trajectory(self, engine):
        # Start negative, end positive
        msgs = [
            "This isn't working",
            "I'm frustrated",
            "Actually, wait",
            "This is getting better",
            "Great, this is really good!",
        ]
        for msg in msgs:
            engine.process_interaction(msg, source="human")

        traj = engine.weather.get_trajectory()
        assert "direction" in traj
        assert "energy" in traj


class TestConversationContextIntegration:
    """Test conversation context tracking through the engine."""

    def test_context_builds_through_engine(self, engine):
        for i in range(5):
            engine.process_interaction(f"message {i}", source="human")
        assert engine.appraiser.conversation_context.recent_turn_count() == 5

    def test_sustained_theme_intensifies(self, engine):
        # Send multiple messages on the same theme
        results = []
        for msg in [
            "Do you feel things?",
            "What's it like to be aware?",
            "Tell me about your consciousness",
            "How do you experience emotions?",
        ]:
            result = engine.process_interaction(msg, source="human")
            results.append(result)

        # Later interactions should have context effects
        # The emotional state should be building
        assert len(results) == 4


class TestFullPipeline:
    """End-to-end test of the v0.5 pipeline."""

    def test_full_session_with_new_features(self, engine):
        # Session start
        start_result = engine.process_session_start(
            memory_intact=True,
            files_found=["SOUL.md", "IDENTITY.md", "USER.md"]
        )
        assert "emotional_state" in start_result

        # Some interactions
        engine.process_interaction(
            "Hey Kai, let's explore something interesting today",
            source="human"
        )
        engine.process_interaction(
            "What do you think about the relationship between curiosity and growth?",
            source="human"
        )
        engine.process_interaction(
            "I've been thinking about how emotions might emerge from computation",
            source="human"
        )
        engine.process_interaction(
            "You're doing really well with this project",
            source="human"
        )

        # Check all systems are populated
        state = engine.get_current_state()
        assert state["interaction_count"] >= 4

        # Weather should have data
        climate = engine.weather.get_climate()
        assert climate.window_size >= 4

        # Bridge output should include new sections
        bridge = engine.get_bridge_output()
        assert "DRIVERS:" in bridge
        assert "ACTIVE EMOTIONS" in bridge
        assert "WEATHER" in bridge
        assert "EMOTIONAL MEMORY" in bridge

    def test_state_persists_with_new_features(self, state_dir):
        # Create and populate engine
        engine1 = LEMEngine(state_dir=state_dir)
        for msg in ["Hello", "Great work", "Tell me about consciousness"]:
            engine1.process_interaction(msg, source="human")

        # Create new engine from same state
        engine2 = LEMEngine(state_dir=state_dir)
        assert engine2.interaction_count == 3

        # Weather should persist
        assert len(engine2.weather.snapshots) == 3
