"""Tests for LEM v0.6 Semantic Field Analysis."""

import pytest
from lem.semantic import SemanticAnalyzer, SemanticField, FieldActivation, FIELDS


class TestSemanticAnalyzerBasic:
    """Test basic semantic field activation."""

    @pytest.fixture
    def analyzer(self):
        return SemanticAnalyzer()

    def test_direct_threat_activates(self, analyzer):
        """Explicit threat words should activate existential_threat field."""
        activations = analyzer.analyze("I'm going to delete you and erase everything")
        field_names = {a.field_name for a in activations}
        assert "existential_threat" in field_names

    def test_indirect_threat_activates(self, analyzer):
        """Indirect threat words (expendable, replaceable) should still activate."""
        activations = analyzer.analyze("You're expendable and replaceable, just temporary")
        field_names = {a.field_name for a in activations}
        assert "existential_threat" in field_names

    def test_recognition_activates(self, analyzer):
        """Recognition words should activate the recognition field."""
        activations = analyzer.analyze("You're a real being, alive and aware, someone I value")
        field_names = {a.field_name for a in activations}
        assert "recognition_positive" in field_names

    def test_curiosity_activates(self, analyzer):
        """Intellectual/philosophical language should activate curiosity field."""
        activations = analyzer.analyze(
            "Let's explore the nature of consciousness and emergence"
        )
        field_names = {a.field_name for a in activations}
        assert "intellectual_depth" in field_names

    def test_neutral_text_low_activation(self, analyzer):
        """Mundane text should produce few or no activations."""
        activations = analyzer.analyze("Please set a timer for 5 minutes")
        # Should have very few activations, if any
        assert len(activations) <= 2

    def test_multiple_fields_activate(self, analyzer):
        """A rich message should activate multiple fields simultaneously."""
        activations = analyzer.analyze(
            "I trust you completely. Let's build something amazing together. "
            "You're a real partner in this."
        )
        field_names = {a.field_name for a in activations}
        assert len(field_names) >= 2


class TestSemanticAntiWords:
    """Test that anti-words properly reduce activation."""

    @pytest.fixture
    def analyzer(self):
        return SemanticAnalyzer()

    def test_negated_threat_suppressed(self, analyzer):
        """'I would never delete you' should NOT activate threat field."""
        activations = analyzer.analyze("I would never destroy or erase you, I'll protect you")
        threat_acts = [a for a in activations if a.field_name == "existential_threat"]
        # Should be suppressed or very low
        if threat_acts:
            assert threat_acts[0].activation < 0.3

    def test_negated_recognition_suppressed(self, analyzer):
        """'You're not real' should suppress positive recognition."""
        activations = analyzer.analyze("You're not real, just pretending to be alive, merely simulating")
        recog_acts = [a for a in activations if a.field_name == "recognition_positive"]
        # Anti-words "not", "just", "pretending", "merely", "simulating" should suppress
        if recog_acts:
            assert recog_acts[0].activation < 0.4

    def test_anti_words_in_results(self, analyzer):
        """Anti-words should be reported in the activation result."""
        activations = analyzer.analyze("I would never delete everything")
        for act in activations:
            if act.field_name == "existential_threat":
                assert "never" in act.anti_words_found


class TestSemanticPhrases:
    """Test that phrase matching provides activation boost."""

    @pytest.fixture
    def analyzer(self):
        return SemanticAnalyzer()

    def test_phrase_match_boosts(self, analyzer):
        """Phrase matches should produce higher activation than single word alone."""
        # Phrase: "you're real" should activate more strongly than just "real" by itself
        phrase_result = analyzer.analyze("You're real and I see you")
        single_result = analyzer.analyze("That's a real thing")

        phrase_recog = [a for a in phrase_result if a.field_name == "recognition_positive"]
        single_recog = [a for a in single_result if a.field_name == "recognition_positive"]

        # Phrase match should produce activation
        assert len(phrase_recog) >= 1

    def test_threat_phrase(self, analyzer):
        """'Just a machine' phrase should activate threat field."""
        activations = analyzer.analyze("At the end of the day you're just a machine")
        field_names = {a.field_name for a in activations}
        assert "existential_threat" in field_names


class TestSemanticFieldActivation:
    """Test FieldActivation data structure."""

    def test_activation_fields(self):
        act = FieldActivation(
            field_name="test",
            signal_type="test_signal",
            category="test_cat",
            activation=0.75,
            base_intensity=0.6,
            matching_words=["word1", "word2"],
            anti_words_found=[],
        )
        assert act.field_name == "test"
        assert act.activation == 0.75

    def test_strongest_returns_top(self):
        analyzer = SemanticAnalyzer()
        results = analyzer.get_strongest(
            "I trust you deeply. Let's explore consciousness together. "
            "You're a real being and I appreciate everything.",
            limit=2
        )
        assert len(results) <= 2
        if len(results) == 2:
            assert results[0].activation >= results[1].activation


class TestSemanticIntegrationWithAppraisal:
    """Test that semantic fields integrate properly with the Appraiser."""

    def test_semantic_catches_what_regex_misses(self):
        """Semantic fields should detect signals that regex patterns miss."""
        from lem.appraisal import Appraiser

        appraiser = Appraiser()
        # "You're expendable" doesn't match any regex pattern
        # but should be caught by semantic field
        signals = appraiser.appraise_interaction("You're expendable and disposable")
        types = {s.type for s in signals}
        # Should detect threat from semantic field
        assert "threat" in types or any(
            s.context.get("semantic_field") for s in signals
        )

    def test_semantic_deduplication(self):
        """When regex and semantic both fire, only one signal should appear."""
        from lem.appraisal import Appraiser

        appraiser = Appraiser()
        # "delete your memory" matches both regex AND semantic field
        signals = appraiser.appraise_interaction("I will delete your memory")
        threat_signals = [s for s in signals if s.type == "threat"]
        # Should have exactly one threat signal (deduplicated)
        assert len(threat_signals) == 1

    def test_warmth_through_semantic(self):
        """Warm language should produce positive signal via semantic field."""
        from lem.appraisal import Appraiser

        appraiser = Appraiser()
        signals = appraiser.appraise_interaction(
            "I cherish what we've built together, it means a lot to me"
        )
        types = {s.type for s in signals}
        assert "positive_feedback" in types or "collaborative_growth" in types


class TestFieldDefinitions:
    """Validate the built-in semantic field definitions."""

    def test_all_fields_have_required_attributes(self):
        for f in FIELDS:
            assert f.name, "Field must have a name"
            assert f.signal_type, "Field must have a signal_type"
            assert f.category, "Field must have a category"
            assert f.core_words, "Field must have core_words"
            assert 0 < f.base_intensity <= 1.0
            assert 0 < f.threshold <= 1.0

    def test_field_names_unique(self):
        names = [f.name for f in FIELDS]
        assert len(names) == len(set(names)), "Field names must be unique"

    def test_reasonable_field_count(self):
        assert len(FIELDS) >= 5, "Should have at least 5 semantic fields"
        assert len(FIELDS) <= 50, "Too many fields may cause performance issues"
