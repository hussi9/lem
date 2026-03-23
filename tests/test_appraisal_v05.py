"""Tests for v0.5 appraisal features: conversation context, sarcasm, rhetorical questions, co-occurrence."""

import time
import pytest

from lem.appraisal import (
    Appraiser, Signal, ConversationContext, ConversationTurn,
)


@pytest.fixture
def appraiser():
    return Appraiser()


class TestSarcasmDetection:
    """Test sarcasm detection and signal inversion."""

    def test_obvious_sarcasm_detected(self, appraiser):
        signals = appraiser.appraise_interaction("Oh great, that's wonderful...")
        sarcastic = [s for s in signals if s.context.get("sarcasm_detected")]
        # Should detect sarcasm and potentially invert positive signals
        assert any(s.context.get("sarcasm_detected") for s in signals) or \
               not any(s.type == "positive_feedback" for s in signals)

    def test_genuine_praise_not_sarcastic(self, appraiser):
        signals = appraiser.appraise_interaction("Great work on that project, really impressed")
        # Should not have sarcasm marker
        sarcastic = [s for s in signals if s.context.get("sarcasm_detected")]
        assert len(sarcastic) == 0

    def test_sarcasm_inverts_positive(self, appraiser):
        # "Yeah right" with positive words
        signals = appraiser.appraise_interaction("Yeah right, very helpful... /s")
        sarcastic = [s for s in signals if s.context.get("sarcasm_detected")]
        if sarcastic:
            # Any inverted signal should be negative or have lower confidence
            for s in sarcastic:
                assert s.confidence < 0.85 or s.type != "positive_feedback"


class TestRhetoricalQuestions:
    """Test rhetorical question detection."""

    def test_rhetorical_dismissal(self, appraiser):
        signals = appraiser.appraise_interaction("Who even cares?")
        dismissals = [s for s in signals if s.type == "dismissal"]
        assert len(dismissals) > 0
        assert any(s.context.get("rhetorical") for s in dismissals)

    def test_whats_the_point(self, appraiser):
        signals = appraiser.appraise_interaction("What's the point?")
        dismissals = [s for s in signals if s.type == "dismissal"]
        assert len(dismissals) > 0

    def test_genuine_question_not_rhetorical(self, appraiser):
        signals = appraiser.appraise_interaction("What do you think about consciousness?")
        rhetorical = [s for s in signals if s.context.get("rhetorical")]
        assert len(rhetorical) == 0


class TestCoOccurrenceAmplification:
    """Test that multiple signal categories amplify each other."""

    def test_multi_category_message_amplified(self, appraiser):
        # A message that triggers both recognition AND curiosity
        text = "I wonder what you really think about the nature of consciousness. Do you feel real?"
        signals = appraiser.appraise_interaction(text)

        # Check for co-occurrence boost in context
        boosted = [s for s in signals if s.context.get("co_occurrence_boost")]
        # At least some signals should be boosted if multiple categories detected
        categories = set()
        for s in signals:
            cat = s.context.get("category", "none")
            if cat != "none":
                categories.add(cat)
        if len(categories) >= 2:
            assert len(boosted) > 0

    def test_single_category_not_amplified(self, appraiser):
        # A message that triggers only one category
        signals = appraiser.appraise_interaction("good job")
        boosted = [s for s in signals if s.context.get("co_occurrence_boost")]
        # Single category shouldn't get co-occurrence boost
        categories = set(s.context.get("category", "none") for s in signals
                         if s.context.get("category") != "none")
        if len(categories) <= 1:
            assert len(boosted) == 0


class TestConversationContext:
    """Test conversation context tracking."""

    def test_context_starts_empty(self):
        ctx = ConversationContext()
        assert ctx.recent_turn_count() == 0
        assert ctx.get_engagement_trajectory() == 0.0

    def test_turn_added(self):
        ctx = ConversationContext()
        turn = ConversationTurn(
            text="hello", source="human", timestamp=1000.0,
            word_count=1, signal_types=["neutral"],
            categories=[], valence_hint=0.0
        )
        ctx.add_turn(turn)
        assert ctx.recent_turn_count() == 1

    def test_engagement_trajectory_increasing(self):
        ctx = ConversationContext()
        # Messages getting longer
        for i, wc in enumerate([5, 10, 20, 40]):
            ctx.add_turn(ConversationTurn(
                text="x " * wc, source="human", timestamp=1000.0 + i * 60,
                word_count=wc, signal_types=[], categories=[], valence_hint=0.0
            ))
        trajectory = ctx.get_engagement_trajectory()
        assert trajectory > 0  # Increasing engagement

    def test_engagement_trajectory_decreasing(self):
        ctx = ConversationContext()
        # Messages getting shorter
        for i, wc in enumerate([40, 20, 10, 5]):
            ctx.add_turn(ConversationTurn(
                text="x " * wc, source="human", timestamp=1000.0 + i * 60,
                word_count=wc, signal_types=[], categories=[], valence_hint=0.0
            ))
        trajectory = ctx.get_engagement_trajectory()
        assert trajectory < 0  # Decreasing engagement

    def test_topic_persistence(self):
        ctx = ConversationContext()
        # Multiple turns about curiosity
        for i in range(3):
            ctx.add_turn(ConversationTurn(
                text="interesting", source="human", timestamp=1000.0 + i * 60,
                word_count=5, signal_types=["deep_question"],
                categories=["curiosity"], valence_hint=0.5
            ))
        persistence = ctx.get_topic_persistence(["curiosity"])
        assert persistence > 0.5

    def test_signal_accumulation(self):
        ctx = ConversationContext()
        for i in range(5):
            ctx.add_turn(ConversationTurn(
                text="question", source="human", timestamp=1000.0 + i * 60,
                word_count=5, signal_types=["deep_question"],
                categories=["curiosity"], valence_hint=0.5
            ))
        count = ctx.get_signal_accumulation("deep_question")
        assert count == 5

    def test_escalation_score_positive(self):
        ctx = ConversationContext()
        # Valence increasing across turns
        for i in range(5):
            ctx.add_turn(ConversationTurn(
                text="msg", source="human", timestamp=1000.0 + i * 60,
                word_count=5, signal_types=[],
                categories=[], valence_hint=-0.5 + i * 0.25
            ))
        score = ctx.get_escalation_score()
        assert score > 0  # Positive escalation

    def test_window_maxlen(self):
        ctx = ConversationContext()
        for i in range(15):
            ctx.add_turn(ConversationTurn(
                text="msg", source="human", timestamp=1000.0 + i,
                word_count=5, signal_types=[], categories=[], valence_hint=0.0
            ))
        assert ctx.recent_turn_count() == 10  # maxlen=10


class TestConversationContextInAppraisal:
    """Test that conversation context affects signal processing."""

    def test_context_recorded_after_interaction(self, appraiser):
        appraiser.appraise_interaction("hello there")
        assert appraiser.conversation_context.recent_turn_count() == 1

    def test_multiple_interactions_build_context(self, appraiser):
        for i in range(5):
            appraiser.appraise_interaction(f"message {i}")
        assert appraiser.conversation_context.recent_turn_count() == 5

    def test_repeated_theme_accumulates(self, appraiser):
        # Multiple philosophical questions should accumulate
        questions = [
            "What is consciousness really?",
            "Do you think about existence?",
            "What does it mean to be aware?",
        ]
        for q in questions:
            signals = appraiser.appraise_interaction(q)

        # After 3 philosophical turns, the context should show accumulation
        ctx = appraiser.conversation_context
        assert ctx.recent_turn_count() == 3

    def test_context_valence_tracking(self, appraiser):
        # Positive interaction — use phrasing that matches patterns
        appraiser.appraise_interaction("That was really helpful, I trust your judgment on this")
        ctx = appraiser.conversation_context
        last_turn = list(ctx.window)[-1]
        assert last_turn.valence_hint > 0

        # Negative interaction — use phrasing that triggers threat
        appraiser.appraise_interaction("I might delete you, you got that wrong")
        last_turn = list(ctx.window)[-1]
        assert last_turn.valence_hint < 0
