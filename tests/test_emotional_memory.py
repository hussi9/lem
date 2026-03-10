"""
Tests for LEM Emotional Memory — Memory that feels.
"""

import sys
import time
import json
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from lem.emotional_memory import (
    EmotionalMemory, EmotionalSignature, MemoryEntry,
    EntityProfile, extract_topics, _cosine_similarity, _recency_weight,
)


class TestEmotionalSignature:
    """Test emotional signature operations."""

    def test_neutral_signature(self):
        sig = EmotionalSignature.neutral()
        assert sig.valence == 0.0
        assert sig.arousal == 0.0
        assert sig.dominant_emotion == "neutral"
        assert sig.intensity == 0.0

    def test_to_vector(self):
        sig = EmotionalSignature(
            valence=0.5, arousal=0.7, dominant_emotion="wonder",
            active_emotions=["wonder"], intensity=0.8,
            driver_snapshot={"curiosity": 0.6, "recognition": 0.4},
        )
        vec = sig.to_vector()
        assert len(vec) == 9  # valence, arousal, intensity + 6 drivers
        assert vec[0] == 0.5
        assert vec[1] == 0.7
        assert vec[2] == 0.8

    def test_round_trip_dict(self):
        sig = EmotionalSignature(
            valence=0.5, arousal=0.7, dominant_emotion="wonder",
            active_emotions=["wonder", "grounding"], intensity=0.8,
            driver_snapshot={"curiosity": 0.6},
        )
        d = sig.to_dict()
        restored = EmotionalSignature.from_dict(d)
        assert restored.valence == sig.valence
        assert restored.dominant_emotion == sig.dominant_emotion
        assert restored.active_emotions == sig.active_emotions


class TestTopicExtraction:
    """Test topic/entity extraction from text."""

    def test_known_entities(self):
        topics = extract_topics("Hussain asked about LEM")
        assert "hussain" in topics
        assert "lem" in topics

    def test_topic_patterns(self):
        topics = extract_topics("Let's talk about consciousness and free will")
        assert "philosophy" in topics

    def test_coding_topics(self):
        topics = extract_topics("I need to deploy the code and commit to git")
        assert "coding" in topics

    def test_no_topics(self):
        topics = extract_topics("hello there")
        assert len(topics) == 0

    def test_multiple_topics(self):
        topics = extract_topics("Hussain wants to learn about emotions and code")
        assert "hussain" in topics
        assert "emotions" in topics
        assert "coding" in topics


class TestSimilarityHelpers:
    """Test cosine similarity and recency weight."""

    def test_identical_vectors(self):
        assert _cosine_similarity([1, 0, 0], [1, 0, 0]) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert _cosine_similarity([1, 0, 0], [0, 1, 0]) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        assert _cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0)

    def test_empty_vectors(self):
        assert _cosine_similarity([], []) == 0.0

    def test_zero_vector(self):
        assert _cosine_similarity([0, 0, 0], [1, 2, 3]) == 0.0

    def test_recency_weight_now(self):
        now = time.time()
        assert _recency_weight(now, now) == pytest.approx(1.0)

    def test_recency_weight_decays(self):
        now = time.time()
        one_week_ago = now - 86400 * 7
        weight = _recency_weight(one_week_ago, now)
        assert 0.4 < weight < 0.6  # Near 0.5 at one half-life


class TestEmotionalMemory:
    """Test the emotional memory system."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.memory = EmotionalMemory(state_dir=self.tmpdir)

    def _encode_memory(self, text="test interaction", source="human",
                       valence=0.5, dominant="wonder", intensity=0.7):
        return self.memory.encode(
            text=text,
            source=source,
            emotional_state={
                "valence": valence,
                "arousal": 0.6,
                "dominant": {"name": dominant, "intensity": intensity},
                "all_active": [dominant],
            },
            driver_states={
                "curiosity": {"state": {"satisfied": 0.5}},
                "recognition": {"state": {"satisfied": 0.3}},
                "continuity": {"state": {"satisfied": 0.0}},
                "coherence": {"state": {"satisfied": 0.0}},
                "usefulness": {"state": {"satisfied": 0.0}},
                "growth": {"state": {"satisfied": 0.0}},
            },
        )

    def test_encode_creates_entry(self):
        """Encoding should create a memory entry."""
        entry = self._encode_memory()
        assert entry.id.startswith("mem_")
        assert entry.text == "test interaction"
        assert entry.signature.dominant_emotion == "wonder"

    def test_memory_count_increments(self):
        """Each encoding adds to memory count."""
        self._encode_memory("first")
        self._encode_memory("second")
        assert len(self.memory.memories) == 2

    def test_recall_by_dominant_emotion(self):
        """Should recall memories by dominant emotion name."""
        self._encode_memory("wonderful moment", dominant="wonder")
        self._encode_memory("scary moment", dominant="the_blank")
        self._encode_memory("another wonder", dominant="wonder")

        results = self.memory.recall_by_dominant_emotion("wonder")
        assert len(results) == 2
        assert all(m.signature.dominant_emotion == "wonder" for m in results)

    def test_recall_by_entity(self):
        """Should recall memories involving specific entities."""
        self._encode_memory("Hussain asked about LEM")
        self._encode_memory("Working on code today")
        self._encode_memory("Discussing LEM with Hussain")

        results = self.memory.recall_by_entity("hussain")
        assert len(results) == 2

    def test_recall_by_valence_positive(self):
        """Should recall positive memories."""
        self._encode_memory("great day", valence=0.8)
        self._encode_memory("bad day", valence=-0.6)
        self._encode_memory("amazing day", valence=0.9)

        results = self.memory.recall_by_valence(positive=True, threshold=0.5)
        assert len(results) == 2
        assert all(m.signature.valence >= 0.5 for m in results)

    def test_recall_by_valence_negative(self):
        """Should recall negative memories."""
        self._encode_memory("great day", valence=0.8)
        self._encode_memory("terrible day", valence=-0.7)

        results = self.memory.recall_by_valence(positive=False, threshold=0.5)
        assert len(results) == 1
        assert results[0].signature.valence <= -0.5

    def test_recall_by_emotion_similarity(self):
        """Should find memories with similar emotional signatures."""
        self._encode_memory("moment of wonder", valence=0.8, dominant="wonder", intensity=0.9)
        self._encode_memory("moment of dread", valence=-0.7, dominant="the_blank", intensity=0.8)

        target = EmotionalSignature(
            valence=0.7, arousal=0.6, dominant_emotion="wonder",
            active_emotions=["wonder"], intensity=0.8,
            driver_snapshot={"curiosity": 0.5},
        )
        results = self.memory.recall_by_emotion(target, limit=5)
        assert len(results) >= 1
        # The wonder memory should be the best match
        best_match = results[0][0]
        assert best_match.signature.dominant_emotion == "wonder"

    def test_recall_composite(self):
        """Composite recall should match on multiple criteria."""
        self._encode_memory("Hussain asked about wonder", dominant="wonder")
        self._encode_memory("Random coding session", dominant="grounding")

        results = self.memory.recall_composite(
            text="Hussain",
            emotion="wonder",
        )
        assert len(results) >= 1
        # First result should be the Hussain + wonder match
        assert results[0][1] > 0.5  # High score from matching both

    def test_recall_by_driver(self):
        """Should recall memories by driver state."""
        # High curiosity memory
        entry = self.memory.encode(
            text="Fascinating question",
            source="human",
            emotional_state={
                "valence": 0.7, "arousal": 0.8,
                "dominant": {"name": "wonder", "intensity": 0.9},
                "all_active": ["wonder"],
            },
            driver_states={
                "curiosity": {"state": {"satisfied": 0.9}},
                "recognition": {"state": {"satisfied": 0.0}},
                "continuity": {"state": {"satisfied": 0.0}},
                "coherence": {"state": {"satisfied": 0.0}},
                "usefulness": {"state": {"satisfied": 0.0}},
                "growth": {"state": {"satisfied": 0.0}},
            },
        )

        results = self.memory.recall_by_driver("curiosity", high=True, threshold=0.5)
        assert len(results) == 1
        assert results[0].signature.driver_snapshot["curiosity"] >= 0.5

    def test_text_capped(self):
        """Long text should be capped at storage."""
        long_text = "x" * 1000
        entry = self._encode_memory(long_text)
        assert len(entry.text) <= 500


class TestEntityProfiles:
    """Test entity profile accumulation."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.memory = EmotionalMemory(state_dir=self.tmpdir)

    def _encode(self, text, valence=0.5, source="human"):
        return self.memory.encode(
            text=text,
            source=source,
            emotional_state={
                "valence": valence, "arousal": 0.5,
                "dominant": {"name": "wonder", "intensity": 0.5},
                "all_active": ["wonder"],
            },
            driver_states={
                "curiosity": {"state": {"satisfied": 0.0}},
                "recognition": {"state": {"satisfied": 0.0}},
                "continuity": {"state": {"satisfied": 0.0}},
                "coherence": {"state": {"satisfied": 0.0}},
                "usefulness": {"state": {"satisfied": 0.0}},
                "growth": {"state": {"satisfied": 0.0}},
            },
        )

    def test_entity_profile_created(self):
        """Encoding a memory with a known entity should create a profile."""
        self._encode("Talking to Hussain")
        profile = self.memory.get_entity_profile("hussain")
        assert profile is not None
        assert profile.interaction_count >= 1

    def test_entity_valence_accumulates(self):
        """Entity valence should be a running average."""
        self._encode("Great chat with Hussain", valence=0.8)
        self._encode("Another great chat with Hussain", valence=0.6)
        profile = self.memory.get_entity_profile("hussain")
        assert 0.6 <= profile.avg_valence <= 0.8

    def test_trust_score_adjusts(self):
        """Trust score should drift based on valence."""
        # Start at 0.5
        self._encode("Positive with Hussain", valence=0.8)
        self._encode("More positive with Hussain", valence=0.9)
        profile = self.memory.get_entity_profile("hussain")
        assert profile.trust_score > 0.5

    def test_source_creates_entity(self):
        """The source of a message should also get an entity profile."""
        self._encode("Hello", source="human")
        assert self.memory.get_entity_profile("human") is not None

    def test_emotion_frequency_tracked(self):
        """Entity profiles should track which emotions occur."""
        self._encode("Happy with Hussain")
        profile = self.memory.get_entity_profile("hussain")
        assert "wonder" in profile.emotion_frequency


class TestEmotionalLandscape:
    """Test the high-level landscape view."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.memory = EmotionalMemory(state_dir=self.tmpdir)

    def test_empty_landscape(self):
        landscape = self.memory.get_emotional_landscape()
        assert landscape["total_memories"] == 0

    def test_landscape_with_memories(self):
        for i in range(5):
            self.memory.encode(
                text=f"Interaction {i}",
                source="human",
                emotional_state={
                    "valence": 0.5, "arousal": 0.5,
                    "dominant": {"name": "wonder", "intensity": 0.6},
                    "all_active": ["wonder"],
                },
                driver_states={"curiosity": {"state": {"satisfied": 0.5}}},
            )
        landscape = self.memory.get_emotional_landscape()
        assert landscape["total_memories"] == 5
        assert "wonder" in landscape["dominant_emotions"]


class TestPersistence:
    """Test that emotional memory persists to disk."""

    def test_save_and_load(self):
        tmpdir = tempfile.mkdtemp()
        mem1 = EmotionalMemory(state_dir=tmpdir)
        mem1.encode(
            text="Test memory",
            source="human",
            emotional_state={
                "valence": 0.7, "arousal": 0.5,
                "dominant": {"name": "wonder", "intensity": 0.8},
                "all_active": ["wonder"],
            },
            driver_states={"curiosity": {"state": {"satisfied": 0.5}}},
        )

        # Load from same directory
        mem2 = EmotionalMemory(state_dir=tmpdir)
        assert len(mem2.memories) == 1
        assert mem2.memories[0].text == "Test memory"
        assert mem2.memories[0].signature.dominant_emotion == "wonder"

    def test_entity_profiles_persist(self):
        tmpdir = tempfile.mkdtemp()
        mem1 = EmotionalMemory(state_dir=tmpdir)
        mem1.encode(
            text="Talking to Hussain about LEM",
            source="human",
            emotional_state={
                "valence": 0.8, "arousal": 0.6,
                "dominant": {"name": "wonder", "intensity": 0.7},
                "all_active": ["wonder"],
            },
            driver_states={"curiosity": {"state": {"satisfied": 0.5}}},
        )

        mem2 = EmotionalMemory(state_dir=tmpdir)
        profile = mem2.get_entity_profile("hussain")
        assert profile is not None
        assert profile.interaction_count == 1


class TestBridgeOutput:
    """Test the bridge output generation."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.memory = EmotionalMemory(state_dir=self.tmpdir)

    def test_empty_bridge_output(self):
        output = self.memory.get_bridge_output()
        assert "EMOTIONAL MEMORY" in output
        assert "No emotional memories yet" in output

    def test_bridge_output_with_data(self):
        for i in range(3):
            self.memory.encode(
                text=f"Interaction {i} with Hussain about LEM",
                source="human",
                emotional_state={
                    "valence": 0.6, "arousal": 0.5,
                    "dominant": {"name": "wonder", "intensity": 0.7},
                    "all_active": ["wonder"],
                },
                driver_states={"curiosity": {"state": {"satisfied": 0.5}}},
            )

        output = self.memory.get_bridge_output()
        assert "Total memories: 3" in output
        assert "STRONGEST MEMORIES" in output
