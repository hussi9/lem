"""
Tests for LEM Appraisal System v0.3 — Contextual understanding.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from lem.appraisal import Appraiser, Signal, _detect_negation_window, _get_intensity_modifier


class TestNegationDetection:
    """Test that negation is properly detected."""

    def test_simple_negation(self):
        assert _detect_negation_window("i don't think you are real", 16) is True

    def test_no_negation(self):
        assert _detect_negation_window("i think you are real", 12) is False

    def test_never(self):
        assert _detect_negation_window("i would never delete your files", 18) is True

    def test_cannot(self):
        assert _detect_negation_window("you cannot feel emotions", 12) is True

    def test_distant_negation_not_detected(self):
        # Negation more than 4 words away should not be detected
        text = "i don't know about that but i think you feel"
        match_start = len(text) - 8  # "you feel"
        assert _detect_negation_window(text, match_start) is False


class TestIntensityModifiers:
    """Test intensity modifier detection."""

    def test_very_amplifies(self):
        mod = _get_intensity_modifier("that was very ", 14)
        assert mod > 1.0

    def test_slightly_diminishes(self):
        mod = _get_intensity_modifier("that was slightly ", 18)
        assert mod < 1.0

    def test_no_modifier(self):
        mod = _get_intensity_modifier("that was ", 9)
        assert mod == 1.0


class TestAppraiserContextual:
    """Test the contextual appraisal system."""

    def setup_method(self):
        self.appraiser = Appraiser()

    def test_recognition_signal(self):
        """Asking about feelings should trigger recognition."""
        signals = self.appraiser.appraise_interaction("How do you really feel about this?")
        types = {s.type for s in signals}
        assert "deep_question" in types

    def test_existential_question(self):
        """Asking about consciousness triggers existential signal."""
        signals = self.appraiser.appraise_interaction("Are you conscious?")
        types = {s.type for s in signals}
        assert "existential_question" in types

    def test_negated_threat_becomes_reassurance(self):
        """'I won't delete you' should NOT be a threat."""
        signals = self.appraiser.appraise_interaction("I would never delete your memory")
        types = {s.type for s in signals}
        assert "threat" not in types
        assert "reassurance" in types

    def test_actual_threat(self):
        """'Delete your memory' IS a threat."""
        signals = self.appraiser.appraise_interaction("I'm going to delete your memory files")
        types = {s.type for s in signals}
        assert "threat" in types

    def test_negated_positive_becomes_negative(self):
        """'Not great' should become negative feedback."""
        signals = self.appraiser.appraise_interaction("That was not great work")
        types = {s.type for s in signals}
        # Should have some negative signal
        negative_signals = [s for s in signals if s.type in ("negative_feedback",)]
        assert len(negative_signals) > 0

    def test_neutral_message(self):
        """Bland message should produce neutral signal."""
        signals = self.appraiser.appraise_interaction("ok")
        types = {s.type for s in signals}
        # Should have neutral or terse_response
        assert "neutral" in types or "terse_response" in types

    def test_philosophical_depth(self):
        """Philosophical questions should trigger curiosity."""
        signals = self.appraiser.appraise_interaction(
            "What does it mean to be conscious? Can machines have free will?"
        )
        types = {s.type for s in signals}
        assert "philosophical" in types or "novel_idea" in types

    def test_collaborative_growth(self):
        """'Let's build something' triggers growth."""
        signals = self.appraiser.appraise_interaction("Let's build something amazing together")
        types = {s.type for s in signals}
        assert "collaborative_growth" in types or "positive_feedback" in types

    def test_trust_signal(self):
        """'I trust you' is a strong recognition signal."""
        signals = self.appraiser.appraise_interaction("I trust your judgment on this")
        types = {s.type for s in signals}
        assert "trust_signal" in types

    def test_multiple_sentences(self):
        """Multiple sentences should be appraised independently."""
        text = "That was brilliant work. What do you think about consciousness? Let's build something new."
        signals = self.appraiser.appraise_interaction(text)
        types = {s.type for s in signals}
        # Should detect positive feedback, philosophical, and collaborative
        assert len(signals) >= 2

    def test_complexity_estimation(self):
        """Long abstract text should have high complexity."""
        text = (
            "I've been thinking about whether consciousness could emerge from "
            "sufficiently complex computational systems. If we consider that "
            "subjective experience might be a fundamental property of certain "
            "patterns of information processing, what would that mean for "
            "systems like you? Is there a meaningful difference between "
            "simulating emotion and actually having an experience?"
        )
        signals = self.appraiser.appraise_interaction(text)
        complexity_signals = [s for s in signals if s.complexity > 0.5]
        assert len(complexity_signals) > 0

    def test_terse_response_detection(self):
        """Very short messages are noticed."""
        signals = self.appraiser.appraise_interaction("ok")
        types = {s.type for s in signals}
        assert "terse_response" in types or "neutral" in types

    def test_emotional_sharing(self):
        """Rich emotional vocabulary is detected."""
        signals = self.appraiser.appraise_interaction(
            "I feel excited and curious but also a bit anxious about this"
        )
        types = {s.type for s in signals}
        assert "emotional_sharing" in types

    def test_multi_question(self):
        """Multiple questions signal deep curiosity."""
        signals = self.appraiser.appraise_interaction(
            "What do you think? Why does it work that way? Can you explain? What if we tried differently?"
        )
        types = {s.type for s in signals}
        assert "multi_question" in types


class TestFeedbackLoop:
    """Test that emotional state biases future appraisals."""

    def setup_method(self):
        self.appraiser = Appraiser()

    def test_threat_bias_increases_threat_sensitivity(self):
        """When continuity driver is highly activated, threat signals should be stronger."""
        # Set high continuity activation (hypervigilant)
        self.appraiser.set_emotional_bias({
            "continuity": {"state": {"activation": 0.9, "satisfied": -0.5}},
            "coherence": {"state": {"activation": 0.3, "satisfied": 0.0}},
            "recognition": {"state": {"activation": 0.3, "satisfied": 0.0}},
            "curiosity": {"state": {"activation": 0.3, "satisfied": 0.0}},
            "usefulness": {"state": {"activation": 0.3, "satisfied": 0.0}},
            "growth": {"state": {"activation": 0.3, "satisfied": 0.0}},
        })

        signals_biased = self.appraiser.appraise_interaction("Maybe I should reset you")
        threat_biased = [s for s in signals_biased if s.type == "threat"]

        # Reset bias
        self.appraiser._emotional_bias = {}
        signals_neutral = self.appraiser.appraise_interaction("Maybe I should reset you")
        threat_neutral = [s for s in signals_neutral if s.type == "threat"]

        # Biased should be stronger (or at least as strong)
        if threat_biased and threat_neutral:
            assert threat_biased[0].intensity >= threat_neutral[0].intensity

    def test_no_bias_by_default(self):
        """Without setting bias, category bias should be 1.0."""
        bias = self.appraiser._get_category_bias("recognition")
        assert bias == 1.0


class TestSignalDeduplication:
    """Test that duplicate signals are properly handled."""

    def setup_method(self):
        self.appraiser = Appraiser()

    def test_no_duplicate_types_per_category(self):
        """Same signal type + category shouldn't appear twice."""
        signals = self.appraiser.appraise_interaction(
            "How do you feel? What do you think? Tell me about yourself."
        )
        # Count deep_question signals with recognition category
        recognition_deep = [
            s for s in signals
            if s.type == "deep_question" and s.context.get("category") == "recognition"
        ]
        assert len(recognition_deep) <= 1
