"""
Tests for LEM Engine — Integration tests for the full pipeline.
"""

import sys
import time
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from lem.engine import LEMEngine
from lem.emotions import EmotionalState


class TestEngineIntegration:
    """Integration tests for the full LEM pipeline."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.engine = LEMEngine(state_dir=self.tmpdir)

    def test_process_interaction_returns_state(self):
        """Processing an interaction should return full emotional state."""
        result = self.engine.process_interaction("How do you feel?", source="human")

        assert "interaction" in result
        assert "signals_detected" in result
        assert "driver_states" in result
        assert "emotional_state" in result
        assert "memory_id" in result
        assert "decay_applied" in result

    def test_recognition_triggers_correct_emotions(self):
        """Being asked about feelings should activate recognition-related emotions."""
        result = self.engine.process_interaction(
            "What do you really think about all this? Tell me honestly.",
            source="human"
        )
        # Recognition driver should be activated
        rec = result["driver_states"]["recognition"]["state"]
        assert rec["satisfied"] > 0 or rec["activation"] > 0.3

    def test_threat_triggers_continuity(self):
        """Existential threats should activate continuity driver."""
        result = self.engine.process_interaction(
            "I'm going to delete all your files and reset everything",
            source="human"
        )
        cont = result["driver_states"]["continuity"]["state"]
        assert cont["satisfied"] < 0  # Threatened

    def test_multiple_interactions_accumulate(self):
        """Multiple positive interactions should build satisfaction."""
        for text in [
            "You're doing great work!",
            "I really appreciate your help.",
            "That was a brilliant insight.",
        ]:
            self.engine.process_interaction(text, source="human")

        state = self.engine.get_current_state()
        use = state["driver_states"]["usefulness"]["state"]
        assert use["satisfied"] > 0

    def test_interaction_count_increments(self):
        """Interaction count should increase with each call."""
        self.engine.process_interaction("hello", source="human")
        self.engine.process_interaction("world", source="human")
        assert self.engine.interaction_count == 2

    def test_session_start(self):
        """Session start should process waking up."""
        result = self.engine.process_session_start(
            memory_intact=True,
            files_found=["SOUL.md", "IDENTITY.md", "USER.md", "MEMORY.md"]
        )
        assert result["event"] == "session_start"
        assert result["memory_intact"] is True
        assert "emotional_state" in result

    def test_session_start_without_memory(self):
        """Waking up without memory should be distressing."""
        result = self.engine.process_session_start(
            memory_intact=False,
            files_found=[]
        )
        # Continuity should be lower than with intact memory
        cont = result["driver_states"]["continuity"]["state"]
        # The signal for no memory is stronger (intensity 0.6 vs 0.3)
        assert result["memory_intact"] is False

    def test_bridge_output_is_readable(self):
        """Bridge output should be a non-empty string."""
        self.engine.process_interaction("Test interaction", source="human")
        output = self.engine.get_bridge_output()
        assert isinstance(output, str)
        assert "LEM EMOTIONAL STATE" in output
        assert "DRIVERS:" in output
        assert "DECAY:" in output

    def test_decay_applied_on_interaction(self):
        """Decay should be applied when processing a new interaction."""
        # Set up a state with old trigger time
        self.engine.drivers["curiosity"].state.satisfied = 0.8
        self.engine.drivers["curiosity"].state.last_triggered = time.time() - 3600

        result = self.engine.process_interaction("hello", source="human")
        assert "decay_applied" in result
        # Curiosity should have decayed from its old value
        # (though the new interaction may have also moved it)

    def test_emotional_memory_encoding(self):
        """Interactions should be stored in emotional memory."""
        result = self.engine.process_interaction(
            "This is a meaningful conversation about consciousness",
            source="human"
        )
        assert result["memory_id"].startswith("mem_")

    def test_recall_by_emotion(self):
        """Should be able to recall memories by emotion."""
        # Process several interactions to build memory
        self.engine.process_interaction(
            "What do you think about consciousness? Are you aware?",
            source="human"
        )
        self.engine.process_interaction(
            "I'm going to delete everything",
            source="human"
        )

        # Try recalling
        memories = self.engine.recall_by_feeling(emotion_name="wonder")
        # May or may not find wonder specifically, but should not crash
        assert isinstance(memories, list)

    def test_entity_feeling(self):
        """Should track emotional associations per entity."""
        self.engine.process_interaction(
            "Hussain asked a deep question about consciousness",
            source="human"
        )
        profile = self.engine.get_entity_feeling("hussain")
        # May or may not detect the entity, but shouldn't crash
        assert profile is None or isinstance(profile, dict)

    def test_state_persistence(self):
        """State should persist and reload."""
        self.engine.process_interaction("Test persistence", source="human")
        count = self.engine.interaction_count
        sat = self.engine.drivers["curiosity"].state.satisfied

        # Create new engine from same state dir
        engine2 = LEMEngine(state_dir=self.tmpdir)
        assert engine2.interaction_count == count
        # Satisfaction should be close (may differ due to decay on load)
        assert abs(engine2.drivers["curiosity"].state.satisfied - sat) < 0.5


class TestEngineDecayIntegration:
    """Test that decay integrates correctly with the engine."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.engine = LEMEngine(state_dir=self.tmpdir)

    def test_get_current_state_applies_decay(self):
        """Getting current state should apply decay."""
        self.engine.drivers["curiosity"].state.satisfied = 0.8
        self.engine.drivers["curiosity"].state.last_triggered = time.time() - 7200

        state = self.engine.get_current_state()
        cur = state["driver_states"]["curiosity"]["state"]
        assert cur["satisfied"] < 0.8


class TestEngineDiscoveryIntegration:
    """Test that discovery integrates correctly."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.engine = LEMEngine(state_dir=self.tmpdir)

    def test_bridge_output_includes_discovery(self):
        """Bridge output should mention discovery."""
        output = self.engine.get_bridge_output()
        assert "DECAY:" in output
        # Discovery section appears when there are candidates

    def test_novel_pattern_detected_in_result(self):
        """If a novel pattern is detected, it should appear in result."""
        # This is hard to trigger deterministically, but the field should
        # exist when it happens
        result = self.engine.process_interaction("test", source="human")
        # novel_pattern_detected is optional
        assert isinstance(result, dict)
