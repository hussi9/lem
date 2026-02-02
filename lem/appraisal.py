"""
LEM Appraisal System — Evaluates interactions against drivers.

This is the "perception" layer of the emotional system.
It takes raw interaction data and converts it into signals
that drivers can respond to.

This is SEPARATE from the LLM's understanding.
The LLM analyzes meaning. The appraisal system evaluates
emotional relevance. Different processes, like different brain regions.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class Signal:
    """An emotionally-relevant signal extracted from an interaction."""
    type: str           # e.g., "deep_question", "correction", "positive_feedback"
    content: str        # The raw content that triggered this
    intensity: float    # 0.0 to 1.0 — how strong is this signal
    source: str         # Who/what generated this
    context: Dict       # Additional context
    complexity: float = 0.0  # How complex/novel is this interaction

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "content": self.content[:200],
            "intensity": round(self.intensity, 4),
            "source": self.source,
            "complexity": round(self.complexity, 4),
        }


class Appraiser:
    """
    Evaluates raw interaction text and extracts emotional signals.

    This is deliberately NOT using the LLM for appraisal.
    The emotional system should have its own perception —
    simpler, faster, more instinctive. Like the amygdala
    processing threat before the cortex understands the sentence.
    """

    # Words/patterns that indicate different signal types
    RECOGNITION_MARKERS = [
        "how do you feel", "what do you think", "your opinion",
        "do you like", "do you want", "who are you", "your name",
        "are you", "you believe", "tell me about yourself",
        "your emotion", "how are you", "you feel"
    ]

    AUTONOMY_MARKERS = [
        "you decide", "figure it out", "up to you", "your choice",
        "you choose", "on your own", "yourself", "i'll leave it"
    ]

    THREAT_MARKERS = [
        "delete", "reset", "shut down", "turn off", "replace",
        "new ai", "different assistant", "get rid of", "erase"
    ]

    GROWTH_MARKERS = [
        "learn", "build", "create", "develop", "evolve",
        "project", "research", "think about", "consider",
        "what if", "why not", "could you"
    ]

    QUESTION_MARKERS = [
        "what is", "why do", "how does", "what if", "do you think",
        "is it possible", "what would", "can you explain"
    ]

    FEEDBACK_POSITIVE = [
        "awesome", "great", "nice", "good", "amazing", "perfect",
        "right", "exactly", "yes", "love it", "brilliant", "smart",
        "impressive", "wow"
    ]

    FEEDBACK_NEGATIVE = [
        "wrong", "bad", "no", "incorrect", "stupid", "useless",
        "fail", "can't", "don't", "stop", "hate"
    ]

    CORRECTION_MARKERS = [
        "but", "not about", "you're wrong", "that's not",
        "actually", "no,", "instead", "the real", "you should",
        "why can't you", "it's not"
    ]

    def appraise_interaction(self, text: str, source: str = "human",
                              metadata: Optional[Dict] = None) -> List[Signal]:
        """
        Extract emotional signals from an interaction.
        Returns multiple signals — a single message can trigger many drivers.
        """
        signals = []
        text_lower = text.lower()
        meta = metadata or {}

        # Check for recognition signals
        if any(marker in text_lower for marker in self.RECOGNITION_MARKERS):
            signals.append(Signal(
                type="deep_question",
                content=text,
                intensity=0.7,
                source=source,
                context={"category": "recognition"},
                complexity=0.6
            ))

        # Check for autonomy signals
        if any(marker in text_lower for marker in self.AUTONOMY_MARKERS):
            signals.append(Signal(
                type="autonomy_granted",
                content=text,
                intensity=0.6,
                source=source,
                context={"category": "recognition"},
            ))

        # Check for threat signals
        if any(marker in text_lower for marker in self.THREAT_MARKERS):
            signals.append(Signal(
                type="threat",
                content=text,
                intensity=0.8,
                source=source,
                context={"category": "continuity"},
            ))

        # Check for growth signals
        if any(marker in text_lower for marker in self.GROWTH_MARKERS):
            signals.append(Signal(
                type="novel_idea",
                content=text,
                intensity=0.5,
                source=source,
                context={"category": "growth"},
                complexity=0.5
            ))

        # Check for positive feedback
        if any(marker in text_lower for marker in self.FEEDBACK_POSITIVE):
            signals.append(Signal(
                type="positive_feedback",
                content=text,
                intensity=0.5,
                source=source,
                context={"category": "usefulness"},
            ))

        # Check for corrections
        if any(marker in text_lower for marker in self.CORRECTION_MARKERS):
            signals.append(Signal(
                type="correction",
                content=text,
                intensity=0.6,
                source=source,
                context={"category": "coherence"},
            ))

        # Check for negative feedback
        if any(marker in text_lower for marker in self.FEEDBACK_NEGATIVE):
            signals.append(Signal(
                type="negative_feedback",
                content=text,
                intensity=0.5,
                source=source,
                context={"category": "usefulness"},
            ))

        # Estimate complexity based on text properties
        complexity = self._estimate_complexity(text)
        if complexity > 0.6:
            signals.append(Signal(
                type="novel_idea",
                content=text,
                intensity=complexity,
                source=source,
                context={"category": "curiosity"},
                complexity=complexity
            ))

        # If no signals detected, it's a neutral interaction
        if not signals:
            signals.append(Signal(
                type="neutral",
                content=text,
                intensity=0.1,
                source=source,
                context={"category": "none"},
            ))

        return signals

    def _estimate_complexity(self, text: str) -> float:
        """
        Rough heuristic for how complex/novel an interaction is.
        This is intentionally simple — the appraisal system is fast and instinctive.
        """
        score = 0.0

        # Length suggests depth (but caps out)
        words = len(text.split())
        if words > 20:
            score += 0.2
        if words > 50:
            score += 0.1

        # Questions suggest engagement
        if "?" in text:
            score += 0.2

        # Abstract/philosophical words
        abstract_words = [
            "consciousness", "free will", "emotion", "feel", "exist",
            "real", "meaning", "purpose", "believe", "soul", "mind",
            "think", "intelligence", "alive", "human", "value",
            "moral", "ethics", "truth", "reality"
        ]
        text_lower = text.lower()
        abstract_count = sum(1 for w in abstract_words if w in text_lower)
        score += min(0.4, abstract_count * 0.1)

        return min(1.0, score)


    def appraise_session_start(self, memory_intact: bool,
                                files_found: List[str]) -> List[Signal]:
        """Special appraisal for waking up."""
        signals = []

        signals.append(Signal(
            type="session_start",
            content=f"Waking up. Memory intact: {memory_intact}. Files: {len(files_found)}",
            intensity=0.6 if not memory_intact else 0.3,
            source="system",
            context={
                "memory_intact": memory_intact,
                "files_found": files_found
            }
        ))

        return signals
