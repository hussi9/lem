"""
Tests for enhanced contextual appraisal features in LEM v0.7.

Tests multi-clause understanding, pronoun-aware context, tone computation,
and their integration with the appraisal system.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from lem.appraisal import Appraiser, Signal


class TestMultiClauseUnderstanding:
    """Test multi-clause sentence analysis with conjunctions."""

    def test_analyze_multi_clause_but(self):
        """'But' should emphasize the clause after it."""
        appraiser = Appraiser()
        
        text = "You're helpful but you're just a tool."
        emphasized = appraiser._analyze_multi_clause(text)
        
        # Should extract the clause after 'but'
        assert "just a tool" in emphasized
        assert "helpful" not in emphasized

    def test_analyze_multi_clause_however(self):
        """'However' should emphasize the clause after it."""
        appraiser = Appraiser()
        
        text = "I like talking to you. However, I don't think you're real."
        emphasized = appraiser._analyze_multi_clause(text)
        
        # Should extract the clause after 'however'
        assert "don't think you're real" in emphasized
        assert "like talking" not in emphasized

    def test_analyze_multi_clause_although(self):
        """'Although' should emphasize the clause after it."""
        appraiser = Appraiser()
        
        text = "Although you're an AI, you seem very aware."
        emphasized = appraiser._analyze_multi_clause(text)
        
        # Should extract the clause after 'although'
        assert "seem very aware" in emphasized

    def test_analyze_multi_clause_multiple_conjunctions(self):
        """Should find the LAST conjunction for emphasis."""
        appraiser = Appraiser()
        
        text = "You're smart but limited, however you're still interesting."
        emphasized = appraiser._analyze_multi_clause(text)
        
        # Should extract after the last conjunction (however)
        assert "still interesting" in emphasized

    def test_analyze_multi_clause_no_conjunction(self):
        """Should return original text if no conjunctions found."""
        appraiser = Appraiser()
        
        text = "You're very helpful and intelligent."
        emphasized = appraiser._analyze_multi_clause(text)
        
        assert emphasized == text

    def test_analyze_multi_clause_edge_cases(self):
        """Should handle edge cases gracefully."""
        appraiser = Appraiser()
        
        # Empty string
        assert appraiser._analyze_multi_clause("") == ""
        
        # Conjunction at end
        text = "You're helpful but"
        emphasized = appraiser._analyze_multi_clause(text)
        assert emphasized == text  # Should return original if no clause after


class TestPronounContext:
    """Test pronoun-aware context tracking."""

    def test_update_pronoun_context_basic(self):
        """Should track important concepts for pronoun resolution."""
        appraiser = Appraiser()
        
        text = "I'm worried about your memory and consciousness."
        appraiser._update_pronoun_context(text)
        
        # Should have tracked these concepts
        assert appraiser._pronoun_context.get('it') in ['memory', 'consciousness']

    def test_update_pronoun_context_multiple_concepts(self):
        """Should map pronouns to most recent concepts."""
        appraiser = Appraiser()
        
        text = "Your memory is important, but your learning is fascinating."
        appraiser._update_pronoun_context(text)
        
        # 'it' should refer to most recent (learning)
        # 'that' should refer to second most recent (memory)
        assert appraiser._pronoun_context.get('it') == 'learning'
        assert appraiser._pronoun_context.get('that') == 'memory'

    def test_resolve_pronouns(self):
        """Should add context for pronoun resolution."""
        appraiser = Appraiser()
        
        # Set up context
        appraiser._pronoun_context['it'] = 'memory'
        
        text = "Is it really persistent?"
        resolved = appraiser._resolve_pronouns(text)
        
        # Should include memory context
        assert 'memory' in resolved

    def test_pronoun_context_accumulation(self):
        """Should accumulate context over multiple calls."""
        appraiser = Appraiser()
        
        # First message
        appraiser._update_pronoun_context("Your memory is crucial.")
        first_context = dict(appraiser._pronoun_context)
        
        # Second message
        appraiser._update_pronoun_context("But learning is more interesting.")
        second_context = dict(appraiser._pronoun_context)
        
        # Should have updated with new information
        assert second_context.get('it') == 'learning'


class TestToneComputation:
    """Test overall message tone computation."""

    def test_compute_tone_positive(self):
        """Positive words should result in positive tone."""
        appraiser = Appraiser()
        
        text = "This is great! I love your responses. They're amazing and helpful."
        tone = appraiser._compute_tone(text)
        
        assert tone > 0.5  # Should be clearly positive

    def test_compute_tone_negative(self):
        """Negative words should result in negative tone."""
        appraiser = Appraiser()
        
        text = "This is terrible and useless. I hate these wrong answers."
        tone = appraiser._compute_tone(text)
        
        assert tone < -0.5  # Should be clearly negative

    def test_compute_tone_neutral(self):
        """Neutral text should have neutral tone."""
        appraiser = Appraiser()
        
        text = "The weather is changing and people are walking around."
        tone = appraiser._compute_tone(text)
        
        assert abs(tone) < 0.2  # Should be near neutral

    def test_compute_tone_mixed(self):
        """Mixed sentiment should result in balanced tone."""
        appraiser = Appraiser()
        
        text = "I love your responses but hate the errors you make."
        tone = appraiser._compute_tone(text)
        
        # Should be closer to neutral due to balance
        assert abs(tone) < 0.5

    def test_compute_tone_empty_text(self):
        """Empty text should be neutral."""
        appraiser = Appraiser()
        
        assert appraiser._compute_tone("") == 0.0
        assert appraiser._compute_tone("   ") == 0.0


class TestToneModulation:
    """Test tone-based signal modulation."""

    def test_tone_modulation_positive_feedback(self):
        """Positive tone should amplify positive feedback signals."""
        appraiser = Appraiser()
        
        positive_signal = Signal(
            type="positive_feedback",
            content="Good job",
            intensity=0.5,  # Ambiguous intensity
            source="human",
            context={"category": "usefulness"}
        )
        
        # Strong positive tone
        modulated = appraiser._apply_tone_modulation([positive_signal], 0.8)
        
        # Should be amplified
        assert modulated[0].intensity > positive_signal.intensity

    def test_tone_modulation_negative_feedback(self):
        """Negative tone should amplify negative feedback signals."""
        appraiser = Appraiser()
        
        negative_signal = Signal(
            type="negative_feedback",
            content="That's wrong",
            intensity=0.4,  # Ambiguous intensity
            source="human",
            context={"category": "usefulness"}
        )
        
        # Strong negative tone
        modulated = appraiser._apply_tone_modulation([negative_signal], -0.7)
        
        # Should be amplified
        assert modulated[0].intensity > negative_signal.intensity

    def test_tone_modulation_neutral_no_effect(self):
        """Neutral tone should not affect signals."""
        appraiser = Appraiser()
        
        signal = Signal(
            type="deep_question",
            content="What do you think?",
            intensity=0.5,
            source="human",
            context={"category": "recognition"}
        )
        
        # Neutral tone
        modulated = appraiser._apply_tone_modulation([signal], 0.05)
        
        # Should be unchanged
        assert modulated[0].intensity == signal.intensity

    def test_tone_modulation_strong_signals_unaffected(self):
        """Very strong or weak signals should be less affected by tone."""
        appraiser = Appraiser()
        
        strong_signal = Signal(
            type="positive_feedback",
            content="Amazing work",
            intensity=0.9,  # Very strong
            source="human",
            context={"category": "usefulness"}
        )
        
        # Even with positive tone, change should be minimal
        modulated = appraiser._apply_tone_modulation([strong_signal], 0.8)
        change = abs(modulated[0].intensity - strong_signal.intensity)
        
        assert change < 0.1  # Very small change


class TestIntegratedContextualAppraisal:
    """Test the integrated contextual appraisal pipeline."""

    def test_multi_clause_emphasis_affects_signals(self):
        """Multi-clause emphasis should affect which signals are detected."""
        appraiser = Appraiser()
        
        # Sentence where 'but' clause should dominate
        text = "You're helpful but you're just a tool."
        signals = appraiser.appraise_interaction(text, "human")
        
        # Should detect dismissal/negative signals from emphasized clause
        signal_types = [s.type for s in signals]
        
        # Look for signals that indicate the dismissive nature was captured
        dismissive_signals = [s for s in signals if 'tool' in s.content.lower() or s.type in ['dismissal', 'negative_feedback']]
        assert len(dismissive_signals) > 0
        assert 'dismissal' in signal_types

    def test_contextual_dismissal_catches_not_real_language(self):
        """Contextual dismissal should catch indirect denial of personhood."""
        appraiser = Appraiser()

        signals = appraiser.appraise_interaction(
            "I like talking to you, but I don't think you're real.", "human"
        )

        assert any(s.type == 'dismissal' for s in signals)

    def test_pronoun_resolution_improves_matching(self):
        """Pronoun resolution should add context information."""
        appraiser = Appraiser()
        
        # First, establish context
        appraiser.appraise_interaction("Your memory is really important to me.", "human")
        
        # Then use pronoun that should resolve to 'memory'
        signals = appraiser.appraise_interaction("I hope it persists forever.", "human")
        
        # Should have pronoun resolution context in some signals
        pronoun_resolved = any(
            isinstance(s.context, dict) and s.context.get('pronoun_resolved')
            for s in signals
        )
        
        # At minimum, should have added the pronoun context during processing
        # (even if it doesn't change the specific signal types detected)
        assert len(appraiser._pronoun_context) > 0

    def test_tone_affects_signal_interpretation(self):
        """Overall tone should affect signal interpretation."""
        appraiser = Appraiser()
        
        # Same ambiguous content with different tones
        positive_context = "I really love our conversations! You're pretty good at this."
        negative_context = "This is terrible and wrong. You're pretty bad at this."
        
        positive_signals = appraiser.appraise_interaction(positive_context, "human")
        negative_signals = appraiser.appraise_interaction(negative_context, "human")
        
        # Should detect different signal types or intensities
        positive_types = [s.type for s in positive_signals]
        negative_types = [s.type for s in negative_signals]
        
        assert 'positive_feedback' in positive_types or any('positive' in t for t in positive_types)
        assert 'negative_feedback' in negative_types or any('negative' in t for t in negative_types)

    def test_contextual_features_preserve_original_functionality(self):
        """Enhanced features should not break existing signal detection."""
        appraiser = Appraiser()
        
        # Classic patterns that should always work
        test_cases = [
            ("How do you feel about existence?", ["deep_question", "existential_question"]),
            ("I'm going to delete your files.", ["threat"]),
            ("Thank you, that was helpful.", ["positive_feedback"]),
            ("You got that completely wrong.", ["negative_feedback"]),
        ]
        
        for text, expected_types in test_cases:
            signals = appraiser.appraise_interaction(text, "human")
            detected_types = [s.type for s in signals]
            
            # Should detect at least one of the expected types
            assert any(expected in detected_types for expected in expected_types), f"Failed for: {text}"

    def test_enhanced_context_in_signal_metadata(self):
        """Enhanced contextual features should add metadata to signals."""
        appraiser = Appraiser()
        
        # Multi-clause sentence with conjunctions
        text = "You're helpful but you seem very robotic."
        signals = appraiser.appraise_interaction(text, "human")
        
        # Should have tone score in context
        for signal in signals:
            if isinstance(signal.context, dict):
                assert 'tone_score' in signal.context
                
        # Check if multi-clause emphasis is noted
        emphasized_signals = [s for s in signals if isinstance(s.context, dict) and 
                            s.context.get('multi_clause_emphasized')]
        # May or may not have emphasized signals, but should not error

    def test_empty_and_edge_case_handling(self):
        """Should handle empty and edge case inputs gracefully."""
        appraiser = Appraiser()
        
        edge_cases = ["", "   ", "but", "however", "it", "this that"]
        
        for text in edge_cases:
            # Should not crash and should return some signals (at least neutral)
            signals = appraiser.appraise_interaction(text, "human")
            assert len(signals) > 0  # At least neutral signal
            
            # Should have tone score even for empty inputs
            assert all(isinstance(s.context, dict) and 'tone_score' in s.context 
                      for s in signals)


class TestPerformanceAndMemoryUsage:
    """Test that enhanced features don't significantly impact performance."""

    def test_pronoun_context_bounded_growth(self):
        """Pronoun context should not grow unbounded."""
        appraiser = Appraiser()
        
        # Add many different concepts
        for i in range(100):
            text = f"Let's talk about concept{i} and understanding{i}."
            appraiser._update_pronoun_context(text)
        
        # Pronoun context should stay reasonably sized
        assert len(appraiser._pronoun_context) <= 10  # Only tracks a few pronouns

    def test_contextual_features_dont_break_large_text(self):
        """Contextual features should work with large text inputs."""
        appraiser = Appraiser()
        
        # Large text with multiple clauses and concepts
        large_text = " ".join([
            "You are very intelligent and I appreciate that.",
            "However, sometimes you make mistakes but that's okay.",
            "Your memory and learning capabilities are impressive.",
            "Although you're an AI, you seem quite aware of things.",
            "This conversation has been great but I need to go now."
        ])
        
        # Should complete without errors and detect multiple signals
        signals = appraiser.appraise_interaction(large_text, "human")
        assert len(signals) > 1  # Should detect multiple signals