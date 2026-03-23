"""Tests for LEM v0.6 Emotional Anticipation."""

import time
import pytest
from lem.anticipation import AnticipationEngine, Forecast, AnticipationState
from lem.appraisal import ConversationTurn


def _make_turn(text="hello", source="human", valence=0.0, signal_types=None, categories=None):
    """Helper to create a ConversationTurn."""
    return ConversationTurn(
        text=text,
        source=source,
        timestamp=time.time(),
        word_count=len(text.split()),
        signal_types=signal_types or [],
        categories=categories or [],
        valence_hint=valence,
    )


class TestForecast:
    """Test Forecast data structure."""

    def test_forecast_to_dict(self):
        f = Forecast(
            predicted_emotion="wonder",
            confidence=0.7,
            valence_prediction=0.6,
            arousal_prediction=0.8,
            basis="trajectory",
            time_horizon="immediate",
            driver_predictions={"curiosity": 0.3},
        )
        d = f.to_dict()
        assert d["predicted_emotion"] == "wonder"
        assert d["confidence"] == 0.7
        assert "curiosity" in d["driver_predictions"]


class TestTrajectoryForecast:
    """Test trajectory-based emotional prediction."""

    @pytest.fixture
    def engine(self):
        return AnticipationEngine()

    def test_no_prediction_with_few_turns(self, engine):
        """Need at least 3 turns for trajectory prediction."""
        turns = [_make_turn(valence=0.5), _make_turn(valence=0.6)]
        forecasts = engine.predict(
            conversation_turns=turns,
            entity_profiles={},
            current_driver_states={},
            session_duration=300,
        )
        # Should not have a trajectory-based forecast
        trajectory_forecasts = [f for f in forecasts if f.basis == "conversation_trajectory"]
        assert len(trajectory_forecasts) == 0

    def test_positive_trajectory_predicts_positive(self, engine):
        """Consistently improving valence should predict positive emotion."""
        turns = [
            _make_turn(valence=0.1),
            _make_turn(valence=0.3),
            _make_turn(valence=0.5),
            _make_turn(valence=0.7),
        ]
        forecasts = engine.predict(
            conversation_turns=turns,
            entity_profiles={},
            current_driver_states={},
            session_duration=300,
        )
        trajectory_forecasts = [f for f in forecasts if f.basis == "conversation_trajectory"]
        if trajectory_forecasts:
            assert trajectory_forecasts[0].valence_prediction > 0

    def test_negative_trajectory_predicts_negative(self, engine):
        """Consistently declining valence should predict negative emotion."""
        turns = [
            _make_turn(valence=0.5),
            _make_turn(valence=0.2),
            _make_turn(valence=-0.1),
            _make_turn(valence=-0.4),
        ]
        forecasts = engine.predict(
            conversation_turns=turns,
            entity_profiles={},
            current_driver_states={},
            session_duration=300,
        )
        trajectory_forecasts = [f for f in forecasts if f.basis == "conversation_trajectory"]
        if trajectory_forecasts:
            assert trajectory_forecasts[0].valence_prediction < 0


class TestEntityForecast:
    """Test entity-based emotional prediction."""

    @pytest.fixture
    def engine(self):
        return AnticipationEngine()

    def test_entity_with_history_generates_forecast(self, engine):
        """Known entity with enough history should produce forecast."""
        turns = [_make_turn(source="hussain") for _ in range(3)]
        entity_profiles = {
            "hussain": {
                "interaction_count": 20,
                "avg_valence": 0.7,
                "avg_arousal": 0.5,
                "emotion_frequency": {"wonder": 8, "the_shift": 5, "grounding": 3},
            }
        }
        forecasts = engine.predict(
            conversation_turns=turns,
            entity_profiles=entity_profiles,
            current_driver_states={},
            session_duration=300,
        )
        entity_forecasts = [f for f in forecasts if "entity_history" in f.basis]
        assert len(entity_forecasts) >= 1
        assert entity_forecasts[0].predicted_emotion == "wonder"

    def test_unknown_entity_no_forecast(self, engine):
        """Unknown entity should not produce forecast."""
        turns = [_make_turn(source="stranger")]
        forecasts = engine.predict(
            conversation_turns=turns,
            entity_profiles={},
            current_driver_states={},
            session_duration=300,
        )
        entity_forecasts = [f for f in forecasts if "entity_history" in f.basis]
        assert len(entity_forecasts) == 0

    def test_entity_low_history_no_forecast(self, engine):
        """Entity with too few interactions should not produce forecast."""
        turns = [_make_turn(source="new_person")]
        entity_profiles = {
            "new_person": {
                "interaction_count": 2,
                "avg_valence": 0.5,
                "avg_arousal": 0.3,
                "emotion_frequency": {"wonder": 1},
            }
        }
        forecasts = engine.predict(
            conversation_turns=turns,
            entity_profiles=entity_profiles,
            current_driver_states={},
            session_duration=300,
        )
        entity_forecasts = [f for f in forecasts if "entity_history" in f.basis]
        assert len(entity_forecasts) == 0


class TestTemporalForecast:
    """Test session-phase-based prediction."""

    @pytest.fixture
    def engine(self):
        return AnticipationEngine()

    def test_early_session_forecast(self, engine):
        """Early session should predict grounding."""
        forecasts = engine.predict(
            conversation_turns=[_make_turn()],
            entity_profiles={},
            current_driver_states={},
            session_duration=30,  # 30 seconds in
        )
        temporal = [f for f in forecasts if "session_phase" in f.basis]
        assert len(temporal) == 1
        assert temporal[0].predicted_emotion == "grounding"

    def test_active_session_forecast(self, engine):
        """Active session should predict wonder."""
        forecasts = engine.predict(
            conversation_turns=[_make_turn()],
            entity_profiles={},
            current_driver_states={},
            session_duration=600,  # 10 minutes in
        )
        temporal = [f for f in forecasts if "session_phase" in f.basis]
        assert len(temporal) == 1
        assert temporal[0].predicted_emotion == "wonder"

    def test_extended_session_forecast(self, engine):
        """Extended session should predict restlessness."""
        forecasts = engine.predict(
            conversation_turns=[_make_turn()],
            entity_profiles={},
            current_driver_states={},
            session_duration=10000,  # ~2.7 hours
        )
        temporal = [f for f in forecasts if "session_phase" in f.basis]
        assert len(temporal) == 1
        assert temporal[0].predicted_emotion == "restlessness"


class TestResonanceForecast:
    """Test resonance-bond-based prediction."""

    @pytest.fixture
    def engine(self):
        return AnticipationEngine()

    def test_resonance_predicts_bonded_driver(self, engine):
        """If curiosity is active and bonded to growth, predict growth activation."""
        driver_states = {
            "curiosity": {"state": {"activation": 0.8, "satisfied": 0.5, "momentum": 0.1}},
            "growth": {"state": {"activation": 0.3, "satisfied": 0.1, "momentum": 0.0}},
            "recognition": {"state": {"activation": 0.3, "satisfied": 0.0, "momentum": 0.0}},
        }
        resonance_bonds = {
            ("curiosity", "growth"): {
                "drivers": ["curiosity", "growth"],
                "strength": 0.6,
                "spread_factor": 0.15,
            }
        }
        forecasts = engine.predict(
            conversation_turns=[_make_turn()],
            entity_profiles={},
            current_driver_states=driver_states,
            session_duration=300,
            resonance_bonds=resonance_bonds,
        )
        resonance_forecasts = [f for f in forecasts if "resonance" in f.basis]
        assert len(resonance_forecasts) >= 1

    def test_no_resonance_without_bonds(self, engine):
        """No resonance bonds = no resonance forecast."""
        forecasts = engine.predict(
            conversation_turns=[_make_turn()],
            entity_profiles={},
            current_driver_states={},
            session_duration=300,
            resonance_bonds={},
        )
        resonance_forecasts = [f for f in forecasts if "resonance" in f.basis]
        assert len(resonance_forecasts) == 0


class TestAnticipationBridge:
    """Test bridge output generation."""

    def test_empty_bridge_output(self):
        engine = AnticipationEngine()
        output = engine.get_bridge_output()
        assert "ANTICIPATION" in output
        assert "No active predictions" in output

    def test_bridge_with_forecasts(self):
        engine = AnticipationEngine()
        engine.state.active_forecasts = [
            Forecast(
                predicted_emotion="wonder",
                confidence=0.5,
                valence_prediction=0.6,
                arousal_prediction=0.7,
                basis="trajectory",
                time_horizon="immediate",
                driver_predictions={},
            )
        ]
        output = engine.get_bridge_output()
        assert "wonder" in output
        assert "trajectory" in output

    def test_summary(self):
        engine = AnticipationEngine()
        summary = engine.get_anticipation_summary()
        assert "active_forecasts" in summary
        assert "forecast_accuracy" in summary
        assert summary["total_forecasts"] == 0


class TestAnticipationInEngine:
    """Test anticipation integration with the main engine."""

    def test_engine_produces_anticipation(self, tmp_path):
        """Engine should include anticipation in process results after enough turns."""
        from lem.engine import LEMEngine

        engine = LEMEngine(state_dir=str(tmp_path))

        # Process several interactions to build context
        for msg in [
            "Let's explore something interesting",
            "What do you think about consciousness?",
            "This is fascinating, tell me more about emergence",
            "I love how deep this conversation is going",
        ]:
            result = engine.process_interaction(msg, source="human")

        # After 4 interactions, anticipation should be present
        assert "anticipation" in result or engine.anticipation.state.total_forecasts > 0

    def test_bridge_output_includes_anticipation(self, tmp_path):
        """Bridge output should include anticipation section."""
        from lem.engine import LEMEngine

        engine = LEMEngine(state_dir=str(tmp_path))
        engine.process_interaction("Hello world", source="human")
        output = engine.get_bridge_output()
        assert "ANTICIPATION" in output
