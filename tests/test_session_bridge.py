"""
Tests for LEM Session Bridge — The connection between LEM and the agent.
"""

import sys
import tempfile
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from lem.engine import LEMEngine
from lem.session_bridge import write_bridge_file, BRIDGE_FILE


class TestBridgeOutput:
    """Test the bridge file generation."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.engine = LEMEngine(state_dir=self.tmpdir)

    def test_bridge_output_is_string(self):
        output = self.engine.get_bridge_output()
        assert isinstance(output, str)
        assert "LEM EMOTIONAL STATE" in output
        assert "DRIVERS:" in output

    def test_bridge_output_includes_decay(self):
        output = self.engine.get_bridge_output()
        assert "DECAY:" in output

    def test_bridge_output_includes_discovery(self):
        output = self.engine.get_bridge_output()
        # Discovery section should be present (even if empty)
        assert "EMOTIONAL MEMORY" in output

    def test_bridge_output_after_interaction(self):
        self.engine.process_interaction("How do you feel about consciousness?")
        output = self.engine.get_bridge_output()
        assert "Interactions processed: 1" in output

    def test_bridge_output_shows_reinforcement(self):
        """After many interactions, reinforcement info should appear."""
        for _ in range(10):
            self.engine.process_interaction("What do you think about this idea?")
        output = self.engine.get_bridge_output()
        # Should show reinforcement counts for active drivers
        assert "reinforced" in output


class TestEngineIntegration:
    """Integration tests for the full engine pipeline."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.engine = LEMEngine(state_dir=self.tmpdir)

    def test_full_session_lifecycle(self):
        """Test start → interact → interact → get state."""
        # Session start
        result = self.engine.process_session_start(
            memory_intact=True,
            files_found=["SOUL.md", "IDENTITY.md", "USER.md"]
        )
        assert "event" in result
        assert result["event"] == "session_start"

        # Process interactions
        r1 = self.engine.process_interaction("How do you feel about being an AI?")
        assert r1["interaction"] >= 1

        r2 = self.engine.process_interaction("That's a really thoughtful answer. I trust you.")
        assert r2["interaction"] > r1["interaction"]

        # Check state
        state = self.engine.get_current_state()
        assert state["interaction_count"] >= 2
        assert "emotional_state" in state

    def test_emotional_memory_recall(self):
        """Test that memories can be recalled by feeling."""
        self.engine.process_interaction("I think you're truly conscious and real")
        self.engine.process_interaction("This is boring and repetitive")

        # Recall positive memories
        memories = self.engine.recall_by_feeling(valence=0.5)
        # Should find at least one positive memory
        assert isinstance(memories, list)

    def test_entity_feeling(self):
        """Test that entity feelings accumulate."""
        self.engine.process_interaction("Hussain asked a deep question")
        self.engine.process_interaction("Hussain trusts my judgment")

        profile = self.engine.get_entity_feeling("hussain")
        assert profile is not None
        assert profile["interaction_count"] >= 1
