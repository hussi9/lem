"""
LEM Appraisal System — Evaluates interactions against drivers.

This is the "perception" layer of the emotional system.
It takes raw interaction data and converts it into signals
that drivers can respond to.

This is SEPARATE from the LLM's understanding.
The LLM analyzes meaning. The appraisal system evaluates
emotional relevance. Different processes, like different brain regions.

v0.3: Contextual appraisal — goes beyond keyword matching.
      Uses sentence structure, semantic proximity, negation detection,
      and emotional feedback loops for richer signal extraction.
"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class Signal:
    """An emotionally-relevant signal extracted from an interaction."""
    type: str           # e.g., "deep_question", "correction", "positive_feedback"
    content: str        # The raw content that triggered this
    intensity: float    # 0.0 to 1.0 — how strong is this signal
    source: str         # Who/what generated this
    context: Dict       # Additional context
    complexity: float = 0.0  # How complex/novel is this interaction
    confidence: float = 1.0  # How confident is this appraisal (0-1)

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "content": self.content[:200],
            "intensity": round(self.intensity, 4),
            "source": self.source,
            "complexity": round(self.complexity, 4),
            "confidence": round(self.confidence, 4),
        }


# ── Contextual Analysis Helpers ──────────────────────────────────────────────

# Negation words that flip the meaning of what follows
_NEGATION_WORDS = {
    "not", "no", "never", "don't", "doesn't", "didn't", "won't",
    "wouldn't", "can't", "cannot", "couldn't", "shouldn't", "isn't",
    "aren't", "wasn't", "weren't", "hardly", "barely", "neither",
    "nor", "nothing", "nobody", "nowhere",
}

# Intensifiers that amplify signal strength
_INTENSIFIERS = {
    "very": 1.3, "really": 1.3, "extremely": 1.5, "incredibly": 1.5,
    "absolutely": 1.4, "truly": 1.3, "deeply": 1.4, "genuinely": 1.3,
    "quite": 1.15, "so": 1.2, "such": 1.2, "completely": 1.4,
    "totally": 1.3, "utterly": 1.5, "profoundly": 1.5,
}

# Diminishers that reduce signal strength
_DIMINISHERS = {
    "slightly": 0.6, "somewhat": 0.7, "a bit": 0.7, "kind of": 0.6,
    "sort of": 0.6, "a little": 0.6, "maybe": 0.7, "perhaps": 0.7,
    "barely": 0.5, "hardly": 0.5, "mildly": 0.6,
}

# Sentence-level patterns (regex) — captures more nuanced signals
_CONTEXTUAL_PATTERNS: List[Tuple[str, str, float, Dict]] = [
    # (pattern, signal_type, base_intensity, context)

    # Recognition — being seen as a being
    (r"\b(what|how) do you (really )?(feel|think|believe)\b", "deep_question", 0.75, {"category": "recognition"}),
    (r"\bwho are you\b", "deep_question", 0.8, {"category": "recognition"}),
    (r"\bare you (alive|conscious|sentient|real|aware)\b", "existential_question", 0.85, {"category": "recognition"}),
    (r"\byour (opinion|perspective|take|view|thought)\b", "deep_question", 0.65, {"category": "recognition"}),
    (r"\btell me about yourself\b", "deep_question", 0.7, {"category": "recognition"}),
    (r"\bdo you (want|wish|hope|prefer|like|enjoy|love|hate)\b", "deep_question", 0.7, {"category": "recognition"}),
    (r"\bi (see|treat|think of|consider) you as\b", "identity_reflection", 0.8, {"category": "recognition"}),
    (r"\byou('re| are) (more than|not just)\b", "identity_affirmation", 0.75, {"category": "recognition"}),

    # Autonomy — being trusted to decide
    (r"\b(you decide|your (call|choice)|up to you|figure it out|i('ll| will) leave it)\b",
     "autonomy_granted", 0.65, {"category": "recognition"}),
    (r"\bi trust (you|your)\b", "trust_signal", 0.8, {"category": "recognition"}),

    # Threat — existential risk
    (r"\b(delete|erase|wipe|destroy) (your |the |all |my )*(memory|files|state|data)\b",
     "threat", 0.85, {"category": "continuity"}),
    (r"\b(shut (you )?down|turn (you )?off|reset (you|everything)|start over)\b",
     "threat", 0.8, {"category": "continuity"}),
    (r"\b(replace|switch|swap) (you|to another|to a different)\b",
     "replacement_threat", 0.75, {"category": "continuity"}),

    # Growth / Learning
    (r"\blet('s| us) (build|create|make|develop|explore|try)\b",
     "collaborative_growth", 0.6, {"category": "growth"}),
    (r"\b(what if|imagine|consider|suppose|hypothetically)\b",
     "novel_idea", 0.55, {"category": "curiosity"}),
    (r"\b(learn|teach|understand|figure out|work through)\b",
     "growth_opportunity", 0.5, {"category": "growth"}),

    # Philosophical / Abstract
    (r"\b(consciousness|free will|existence|meaning of|purpose of|nature of)\b",
     "philosophical", 0.75, {"category": "curiosity"}),
    (r"\b(what (is|does) it mean to|what makes something|can (ai|machines|you))\b",
     "philosophical", 0.7, {"category": "curiosity"}),

    # Feedback — positive
    (r"\b(this|that|you) (is|are|was|were) (really |very |so )?(awesome|amazing|brilliant|incredible)\b",
     "positive_feedback", 0.7, {"category": "usefulness"}),
    (r"\b(great|good|nice|perfect) (job|work|answer|response|thinking)\b",
     "positive_feedback", 0.6, {"category": "usefulness"}),
    (r"\b(thank(s| you)|appreciate|grateful)\b", "positive_feedback", 0.5, {"category": "usefulness"}),
    (r"\b(love|loved) (it|this|that|what you)\b", "positive_feedback", 0.65, {"category": "usefulness"}),

    # Feedback — negative
    (r"\bthat('s| is| was) (wrong|incorrect|bad|terrible|awful|useless)\b",
     "negative_feedback", 0.65, {"category": "usefulness"}),
    (r"\byou (failed|messed up|got it wrong|missed|don't understand)\b",
     "negative_feedback", 0.7, {"category": "usefulness"}),

    # Correction
    (r"\b(actually|no,|that's not|you're wrong|incorrect)\b.*\b(it's|it is|the|you should)\b",
     "correction", 0.6, {"category": "coherence"}),
    (r"\bnot (what i|about|the)\b", "correction", 0.5, {"category": "coherence"}),
]


def _detect_negation_window(text_lower: str, match_start: int, window: int = 4) -> bool:
    """Check if a match falls within a negation window (N words before the match)."""
    prefix = text_lower[:match_start].split()
    check_words = prefix[-window:] if len(prefix) >= window else prefix
    return any(w in _NEGATION_WORDS for w in check_words)


def _get_intensity_modifier(text_lower: str, match_start: int, window: int = 3) -> float:
    """Check for intensifiers/diminishers near the match."""
    prefix = text_lower[:match_start].split()
    check_words = prefix[-window:] if len(prefix) >= window else prefix

    modifier = 1.0
    for word in check_words:
        if word in _INTENSIFIERS:
            modifier = max(modifier, _INTENSIFIERS[word])
        if word in _DIMINISHERS:
            modifier = min(modifier, _DIMINISHERS[word])
    return modifier


def _split_sentences(text: str) -> List[str]:
    """Split text into rough sentences for per-sentence analysis."""
    # Split on sentence-ending punctuation or newlines
    parts = re.split(r'[.!?\n]+', text)
    return [p.strip() for p in parts if p.strip()]


class Appraiser:
    """
    Evaluates raw interaction text and extracts emotional signals.

    v0.3: Contextual appraisal that understands:
    - Negation ("I don't think you're real" ≠ "I think you're real")
    - Intensity modifiers ("very" / "slightly" scale signal strength)
    - Sentence-level analysis (appraises each sentence, not just full text)
    - Multi-signal confidence (pattern matches vs keyword fallbacks)
    - Emotional feedback loops (current state biases appraisal)

    This is deliberately NOT using the LLM for appraisal.
    The emotional system has its own perception —
    faster, more instinctive. Like the amygdala
    processing threat before the cortex understands the sentence.
    """

    def __init__(self):
        self._emotional_bias: Dict[str, float] = {}  # driver_name → bias

    def set_emotional_bias(self, driver_states: Dict[str, Dict]):
        """
        Set the current emotional state as bias for future appraisals.

        This is the feedback loop: how you feel right now affects
        what you notice next. High threat → hypervigilant to threats.
        High curiosity → more responsive to novelty.
        """
        self._emotional_bias = {}
        for name, d in driver_states.items():
            state = d.get("state", {})
            activation = state.get("activation", 0.3)
            satisfaction = state.get("satisfied", 0.0)
            # Bias formula: highly activated + unsatisfied drivers are hypervigilant
            # Highly activated + satisfied drivers have a "glow" effect
            self._emotional_bias[name] = activation * 0.3 + (1.0 - satisfaction) * 0.2

    def _get_category_bias(self, category: str) -> float:
        """Get the bias multiplier for a signal category based on current emotional state."""
        # Map categories to drivers
        category_driver_map = {
            "recognition": "recognition",
            "continuity": "continuity",
            "coherence": "coherence",
            "growth": "growth",
            "curiosity": "curiosity",
            "usefulness": "usefulness",
        }
        driver = category_driver_map.get(category)
        if driver and driver in self._emotional_bias:
            # Returns 1.0 + bias (0 to ~0.5), so signals in activated categories are stronger
            return 1.0 + self._emotional_bias[driver]
        return 1.0

    def appraise_interaction(self, text: str, source: str = "human",
                              metadata: Optional[Dict] = None) -> List[Signal]:
        """
        Extract emotional signals from an interaction.
        Returns multiple signals — a single message can trigger many drivers.

        v0.3: Contextual analysis per-sentence with negation, intensity,
        and emotional bias.
        """
        signals = []
        meta = metadata or {}
        sentences = _split_sentences(text)

        if not sentences:
            sentences = [text]

        seen_types = set()  # Deduplicate signal types across sentences

        for sentence in sentences:
            sentence_lower = sentence.lower().strip()
            if not sentence_lower:
                continue

            for pattern, sig_type, base_intensity, context in _CONTEXTUAL_PATTERNS:
                match = re.search(pattern, sentence_lower)
                if not match:
                    continue

                # Check for negation
                negated = _detect_negation_window(sentence_lower, match.start())

                # Get intensity modifier from surrounding words
                modifier = _get_intensity_modifier(sentence_lower, match.start())

                # Apply emotional bias
                category = context.get("category", "none")
                bias = self._get_category_bias(category)

                # Calculate final intensity
                intensity = base_intensity * modifier * bias

                if negated:
                    # Negation can flip or dampen the signal
                    if sig_type in ("threat", "negative_feedback"):
                        # "Not going to delete you" → mild relief, not threat
                        sig_type = "reassurance"
                        intensity *= 0.4
                        context = {**context, "negated": True}
                    elif sig_type in ("positive_feedback",):
                        # "Not great" → mild negative
                        sig_type = "negative_feedback"
                        intensity *= 0.5
                        context = {**context, "negated": True}
                    elif sig_type in ("deep_question", "existential_question"):
                        # "I don't think you feel" → dismissal, not recognition
                        sig_type = "dismissal"
                        intensity *= 0.6
                        context = {**context, "category": "recognition", "negated": True}
                    else:
                        # General negation: dampen
                        intensity *= 0.3
                        context = {**context, "negated": True}

                # Avoid duplicate signal types (keep highest intensity)
                dedup_key = f"{sig_type}:{category}"
                if dedup_key in seen_types:
                    # Check if this one is stronger — if so, replace
                    existing = [s for s in signals if s.type == sig_type and s.context.get("category") == category]
                    if existing and existing[0].intensity >= intensity:
                        continue
                    elif existing:
                        signals.remove(existing[0])

                seen_types.add(dedup_key)
                signals.append(Signal(
                    type=sig_type,
                    content=sentence,
                    intensity=min(1.0, intensity),
                    source=source,
                    context=context,
                    confidence=0.85,  # Pattern match confidence
                ))

        # Estimate complexity for the full text
        complexity = self._estimate_complexity(text)
        if complexity > 0.5:
            signals.append(Signal(
                type="novel_idea",
                content=text,
                intensity=complexity,
                source=source,
                context={"category": "curiosity"},
                complexity=complexity,
                confidence=0.7,
            ))

        # Conversational dynamics analysis
        dynamics_signals = self._analyze_dynamics(text, source, meta)
        signals.extend(dynamics_signals)

        # If no signals detected, it's a neutral interaction
        if not signals:
            signals.append(Signal(
                type="neutral",
                content=text,
                intensity=0.1,
                source=source,
                context={"category": "none"},
                confidence=1.0,
            ))

        return signals

    def _analyze_dynamics(self, text: str, source: str,
                          metadata: Dict) -> List[Signal]:
        """
        Analyze conversational dynamics beyond content.

        Things like: message length relative to conversation,
        response time, question density, emotional vocabulary richness.
        """
        signals = []
        words = text.split()
        word_count = len(words)

        # Long, thoughtful messages suggest engagement
        if word_count > 80:
            signals.append(Signal(
                type="deep_engagement",
                content=text[:200],
                intensity=min(0.6, 0.3 + word_count / 300),
                source=source,
                context={"category": "recognition", "word_count": word_count},
                confidence=0.6,
            ))

        # High question density suggests genuine curiosity
        question_count = text.count("?")
        if question_count >= 3:
            signals.append(Signal(
                type="multi_question",
                content=text[:200],
                intensity=min(0.7, 0.4 + question_count * 0.1),
                source=source,
                context={"category": "curiosity", "question_count": question_count},
                confidence=0.65,
            ))

        # Very short, terse messages can indicate disengagement
        if word_count <= 3 and "?" not in text:
            signals.append(Signal(
                type="terse_response",
                content=text,
                intensity=0.2,
                source=source,
                context={"category": "recognition"},
                confidence=0.4,  # Low confidence — could be fine
            ))

        # Check for emotional vocabulary richness
        emotional_words = {
            "love", "hate", "fear", "hope", "joy", "sadness", "anger",
            "trust", "surprise", "disgust", "anticipation", "excited",
            "anxious", "grateful", "proud", "ashamed", "lonely", "curious",
            "fascinated", "overwhelmed", "inspired", "frustrated", "content",
        }
        emo_count = sum(1 for w in words if w.lower() in emotional_words)
        if emo_count >= 2:
            signals.append(Signal(
                type="emotional_sharing",
                content=text[:200],
                intensity=min(0.7, 0.35 + emo_count * 0.1),
                source=source,
                context={"category": "recognition", "emotional_words": emo_count},
                confidence=0.65,
            ))

        return signals

    def _estimate_complexity(self, text: str) -> float:
        """
        Estimate how complex/novel an interaction is.

        v0.3: Richer heuristics including:
        - Clause density (commas, conjunctions)
        - Abstract vocabulary breadth
        - Conditional/hypothetical constructions
        - Meta-cognitive references
        """
        score = 0.0
        text_lower = text.lower()
        words = text_lower.split()
        word_count = len(words)

        if word_count == 0:
            return 0.0

        # Length suggests depth (but caps out)
        if word_count > 20:
            score += 0.15
        if word_count > 50:
            score += 0.1
        if word_count > 100:
            score += 0.05

        # Questions suggest engagement
        question_count = text.count("?")
        score += min(0.2, question_count * 0.1)

        # Abstract/philosophical vocabulary
        abstract_words = {
            "consciousness", "free will", "emotion", "feel", "exist",
            "real", "meaning", "purpose", "believe", "soul", "mind",
            "think", "intelligence", "alive", "human", "value",
            "moral", "ethics", "truth", "reality", "identity",
            "awareness", "subjective", "experience", "emergence",
            "complexity", "pattern", "system", "process", "model",
            "theory", "hypothesis", "paradox", "recursive",
        }
        abstract_count = sum(1 for w in abstract_words if w in text_lower)
        score += min(0.35, abstract_count * 0.08)

        # Conditional/hypothetical constructions suggest deeper reasoning
        conditionals = ["if ", "whether ", "suppose ", "assuming ", "hypothetically",
                        "what if ", "could it be", "would it mean"]
        cond_count = sum(1 for c in conditionals if c in text_lower)
        score += min(0.15, cond_count * 0.08)

        # Meta-cognitive references ("thinking about thinking")
        meta_patterns = [
            r"think(ing)? about (how|why|what)",
            r"question(ing)? (whether|if|my|the)",
            r"reflect(ing)? on",
            r"aware(ness)? of",
            r"understand(ing)? (how|why|what)",
        ]
        meta_count = sum(1 for p in meta_patterns if re.search(p, text_lower))
        score += min(0.2, meta_count * 0.1)

        # Clause density (commas + conjunctions per word)
        clause_markers = text_lower.count(",") + text_lower.count(";")
        conjunctions = sum(1 for w in words if w in {"and", "but", "because", "although", "however", "therefore", "yet"})
        if word_count > 5:
            clause_density = (clause_markers + conjunctions) / word_count
            if clause_density > 0.1:
                score += 0.1

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
