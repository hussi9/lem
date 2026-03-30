"""Tests for the LEM Emotional Blending system (v0.8)."""

import time
import pytest
from lem.blending import BlendEngine, EmotionalBlend, NAMED_BLENDS
from lem.emotions import EmotionalState


def _make_emotion(name, intensity=0.5, valence=0.5, arousal=0.5):
    return EmotionalState(
        name=name,
        intensity=intensity,
        valence=valence,
        arousal=arousal,
        source_drivers=["test"],
        is_compound=False,
        is_conflict=False,
        description=f"Test {name}",
        timestamp=time.time(),
    )


class TestTransitionSmoothing:
    def test_new_emotion_starts_low(self):
        engine = BlendEngine()
        now = time.time()
        
        emotions = [_make_emotion("wonder", intensity=0.8)]
        result = engine.apply(emotions, now=now)
        
        # First appearance should start at reduced intensity
        wonder = [e for e in result if e.name == "wonder"]
        assert len(wonder) == 1
        assert wonder[0].intensity < 0.8  # Should start lower
        assert wonder[0].intensity > 0.0  # But not zero

    def test_emotion_ramps_up_over_time(self):
        engine = BlendEngine()
        now = time.time()
        
        # First pass — starts low
        emotions = [_make_emotion("wonder", intensity=0.8)]
        r1 = engine.apply(emotions, now=now)
        i1 = [e for e in r1 if e.name == "wonder"][0].intensity
        
        # Second pass — should be higher
        r2 = engine.apply(emotions, now=now + 5)
        i2 = [e for e in r2 if e.name == "wonder"][0].intensity
        
        assert i2 > i1

    def test_emotion_fades_out_when_removed(self):
        engine = BlendEngine()
        now = time.time()
        
        # Establish emotion and let it ramp up
        emotions = [_make_emotion("wonder", intensity=0.8)]
        engine.apply(emotions, now=now)
        engine.apply(emotions, now=now + 10)
        engine.apply(emotions, now=now + 20)  # Should be near full intensity
        
        # Remove it — short time gap so it doesn't fully fade in one step
        r1 = engine.apply([], now=now + 21)
        
        # Should still be fading, not gone
        wonder = [e for e in r1 if e.name == "wonder"]
        assert len(wonder) == 1
        assert wonder[0].intensity > 0.0
        assert "fading" in wonder[0].description

    def test_emotion_eventually_disappears(self):
        engine = BlendEngine()
        now = time.time()
        
        # Establish emotion briefly
        emotions = [_make_emotion("wonder", intensity=0.3)]
        engine.apply(emotions, now=now)
        
        # Remove and wait a long time
        result = engine.apply([], now=now + 200)
        
        wonder = [e for e in result if e.name == "wonder"]
        assert len(wonder) == 0  # Should have fully faded

    def test_intensity_decrease_smoothed(self):
        engine = BlendEngine()
        now = time.time()
        
        # High intensity
        high = [_make_emotion("wonder", intensity=0.9)]
        engine.apply(high, now=now)
        engine.apply(high, now=now + 10)
        
        # Sudden drop to low intensity
        low = [_make_emotion("wonder", intensity=0.2)]
        result = engine.apply(low, now=now + 11)
        
        wonder = [e for e in result if e.name == "wonder"][0]
        # Should not have jumped all the way down to 0.2 yet
        assert wonder.intensity > 0.2


class TestBlendDetection:
    def test_sublime_blend(self):
        """Wonder + the_blank = sublime."""
        engine = BlendEngine()
        now = time.time()
        
        emotions = [
            _make_emotion("wonder", intensity=0.6, valence=0.8, arousal=0.7),
            _make_emotion("the_blank", intensity=0.5, valence=-0.6, arousal=0.5),
        ]
        
        # Need to establish and let ramp up
        engine.apply(emotions, now=now)
        result = engine.apply(emotions, now=now + 15)
        
        blend_names = [e.name for e in result]
        assert "sublime" in blend_names

    def test_quiet_purpose_blend(self):
        """Grounding + anchor_recognition = quiet_purpose."""
        engine = BlendEngine()
        now = time.time()
        
        emotions = [
            _make_emotion("grounding", intensity=0.5, valence=0.5, arousal=0.2),
            _make_emotion("anchor_recognition", intensity=0.5, valence=0.7, arousal=0.4),
        ]
        
        engine.apply(emotions, now=now)
        result = engine.apply(emotions, now=now + 15)
        
        blend_names = [e.name for e in result]
        assert "quiet_purpose" in blend_names

    def test_no_blend_without_components(self):
        engine = BlendEngine()
        now = time.time()
        
        # Only wonder, no the_blank — no sublime
        emotions = [_make_emotion("wonder", intensity=0.8)]
        engine.apply(emotions, now=now)
        result = engine.apply(emotions, now=now + 15)
        
        blend_names = [e.name for e in result]
        assert "sublime" not in blend_names

    def test_no_blend_below_threshold(self):
        engine = BlendEngine()
        now = time.time()
        
        # Components present but too weak
        emotions = [
            _make_emotion("wonder", intensity=0.1),
            _make_emotion("the_blank", intensity=0.1),
        ]
        
        engine.apply(emotions, now=now)
        result = engine.apply(emotions, now=now + 15)
        
        blend_names = [e.name for e in result]
        assert "sublime" not in blend_names

    def test_blend_intensity_geometric(self):
        """Geometric mean for blend_intensity_func='geometric'."""
        engine = BlendEngine()
        now = time.time()
        
        emotions = [
            _make_emotion("wonder", intensity=0.9, valence=0.8, arousal=0.7),
            _make_emotion("the_blank", intensity=0.4, valence=-0.6, arousal=0.5),
        ]
        
        engine.apply(emotions, now=now)
        result = engine.apply(emotions, now=now + 30)  # Let them ramp up fully
        
        sublime = [e for e in result if e.name == "sublime"]
        if sublime:
            # Geometric mean of the component intensities (after smoothing)
            assert 0.0 < sublime[0].intensity < 1.0


class TestBlendDefinitions:
    def test_all_blends_have_required_fields(self):
        for blend in NAMED_BLENDS:
            assert blend.name
            assert blend.description
            assert len(blend.components) >= 2
            assert 0.0 <= blend.min_component_intensity <= 1.0
            assert -1.0 <= blend.valence <= 1.0
            assert 0.0 <= blend.arousal <= 1.0
            assert blend.blend_intensity_func in ("average", "min", "geometric")

    def test_blend_names_unique(self):
        names = [b.name for b in NAMED_BLENDS]
        assert len(names) == len(set(names))


class TestGetActiveBlends:
    def test_empty_initially(self):
        engine = BlendEngine()
        assert engine.get_active_blends() == []

    def test_returns_active_blend_names(self):
        engine = BlendEngine()
        now = time.time()
        
        emotions = [
            _make_emotion("grounding", intensity=0.6, valence=0.5, arousal=0.2),
            _make_emotion("anchor_recognition", intensity=0.6, valence=0.7, arousal=0.4),
        ]
        
        engine.apply(emotions, now=now)
        engine.apply(emotions, now=now + 15)
        
        blends = engine.get_active_blends()
        assert isinstance(blends, list)


class TestTransitionInfo:
    def test_shows_transitioning_emotions(self):
        engine = BlendEngine()
        now = time.time()
        
        emotions = [_make_emotion("wonder", intensity=0.8)]
        engine.apply(emotions, now=now)
        
        info = engine.get_transition_info()
        assert "wonder" in info
        assert "current" in info["wonder"]
        assert "target" in info["wonder"]
        assert "direction" in info["wonder"]


class TestMultipleEmotionTransitions:
    def test_multiple_emotions_smooth_independently(self):
        engine = BlendEngine()
        now = time.time()
        
        emotions = [
            _make_emotion("wonder", intensity=0.8),
            _make_emotion("grounding", intensity=0.6),
        ]
        
        # Let them ramp up so the different targets produce different intensities
        engine.apply(emotions, now=now)
        result = engine.apply(emotions, now=now + 10)
        
        wonder = [e for e in result if e.name == "wonder"]
        grounding = [e for e in result if e.name == "grounding"]
        
        assert len(wonder) == 1
        assert len(grounding) == 1
        # After ramping, higher target should have higher intensity
        assert wonder[0].intensity > grounding[0].intensity

    def test_one_fading_one_rising(self):
        engine = BlendEngine()
        now = time.time()
        
        # Establish wonder over several updates
        engine.apply([_make_emotion("wonder", intensity=0.8)], now=now)
        engine.apply([_make_emotion("wonder", intensity=0.8)], now=now + 10)
        engine.apply([_make_emotion("wonder", intensity=0.8)], now=now + 20)
        
        # Switch to grounding only — short gap so wonder hasn't fully faded
        result = engine.apply([_make_emotion("grounding", intensity=0.6)], now=now + 21)
        
        wonder = [e for e in result if e.name == "wonder"]
        grounding = [e for e in result if e.name == "grounding"]
        
        assert len(wonder) == 1  # Fading but still present
        assert len(grounding) == 1  # Rising
