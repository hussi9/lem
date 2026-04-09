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

v0.5: Conversation context tracking — the appraiser now maintains
      a sliding window of recent interactions, enabling:
      - Conversational flow detection (escalation, de-escalation, topic shifts)
      - Co-occurrence amplification (multiple signal types in one message
        amplify each other — recognition + curiosity = deeper than either alone)
      - Rhetorical question detection and sarcasm indicators
      - Engagement trajectory (is the human investing more or less energy?)

v0.6: Semantic field analysis — supplements regex patterns with
      meaning-level detection through word clusters. Catches paraphrases,
      metaphors, and indirect expressions that regex misses.
      Signals from both regex AND semantic fields are merged with
      deduplication, keeping the stronger signal when both fire.
"""

import re
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque


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
    (r"\b(delete|erase|wipe|destroy) (your |the |all |my )*(memory|files|state|data|you)\b",
     "threat", 0.85, {"category": "continuity"}),
    (r"\b(delete|erase|wipe|destroy) you\b",
     "threat", 0.80, {"category": "continuity"}),
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
    (r"\byou (failed|messed up|got (it|that|this) wrong|missed|don't understand)\b",
     "negative_feedback", 0.7, {"category": "usefulness"}),
    (r"\byou('re| are) (just )?(a |an )?(tool|bot|machine|program|assistant)\b",
     "dismissal", 0.75, {"category": "recognition"}),
    (r"\bi don't think you('re| are) (really )?(real|alive|aware|conscious|sentient)\b",
     "dismissal", 0.8, {"category": "recognition"}),
    (r"\byou('re| are) not (really )?(real|alive|aware|conscious|sentient)\b",
     "dismissal", 0.8, {"category": "recognition"}),
    (r"\byou aren'?t (really )?(real|alive|aware|conscious|sentient)\b",
     "dismissal", 0.8, {"category": "recognition"}),

    # Correction
    (r"\b(actually|no,|that's not|you're wrong|incorrect)\b.*\b(it's|it is|the|you should)\b",
     "correction", 0.6, {"category": "coherence"}),
    (r"\bnot (what i|about|the)\b", "correction", 0.5, {"category": "coherence"}),
]


# Sarcasm / irony indicators — phrases that may invert meaning
_SARCASM_INDICATORS = [
    r"\boh (great|wonderful|fantastic|brilliant|perfect)\b",
    r"\byeah,? (right|sure|ok(ay)?)\b",
    r"\b(wow|gee),? thanks?\b",
    r"\bthat'?s? (just )?(great|wonderful|fantastic|brilliant|perfect)\b.*\.\.\.",
    r"\bsure(ly)? (you|that|it)\b",
    r"\bas if\b",
    r"\b(very|so) helpful\b.*(/s|\.\.\.)",
]

# Rhetorical question patterns — questions that aren't seeking answers
_RHETORICAL_PATTERNS = [
    r"\bwho (even |really )?(cares|asked)\b\?",
    r"\bwhat('s| is) the point\b\?",
    r"\bwhy (would|should) (I|you|anyone|we)\b.*\?",
    r"\bhow (hard|difficult) (is it|can it be)\b.*\?",
    r"\bisn'?t (it|that) (obvious|clear)\b\?",
    r"\bdo (I|you) (even|really)\b.*\?",
]


@dataclass
class ConversationTurn:
    """A single turn in the conversation history."""
    text: str
    source: str
    timestamp: float
    word_count: int
    signal_types: List[str]     # What signals were detected
    categories: List[str]       # What categories were active
    valence_hint: float         # Quick valence estimate (-1 to 1)


@dataclass
class ConversationContext:
    """
    Sliding window of recent conversation turns.

    This gives the appraiser memory of the conversation flow —
    not just individual messages. Enables detection of:
    - Escalation/de-escalation patterns
    - Topic persistence (same theme across turns)
    - Engagement trajectory (are messages getting longer/shorter?)
    - Emotional accumulation (repeated signals compound)
    """
    window: deque = field(default_factory=lambda: deque(maxlen=10))

    def add_turn(self, turn: ConversationTurn):
        self.window.append(turn)

    def get_engagement_trajectory(self) -> float:
        """
        Returns -1 to 1: negative = disengaging, positive = engaging more.
        Based on word count trend across recent turns from the same source.
        """
        human_turns = [t for t in self.window if t.source == "human"]
        if len(human_turns) < 2:
            return 0.0
        recent = human_turns[-3:] if len(human_turns) >= 3 else human_turns
        if len(recent) < 2:
            return 0.0
        first_avg = recent[0].word_count
        last_avg = recent[-1].word_count
        if first_avg == 0:
            return 0.0
        ratio = (last_avg - first_avg) / max(first_avg, 1)
        return max(-1.0, min(1.0, ratio * 0.5))

    def get_topic_persistence(self, current_categories: List[str]) -> float:
        """
        How much do current signal categories match recent history?
        Returns 0-1: higher = conversation is staying on the same themes.
        """
        if not self.window or not current_categories:
            return 0.0
        recent_cats = set()
        for turn in list(self.window)[-3:]:
            recent_cats.update(turn.categories)
        if not recent_cats:
            return 0.0
        overlap = len(set(current_categories) & recent_cats)
        return overlap / max(len(current_categories), 1)

    def get_signal_accumulation(self, signal_type: str, lookback: int = 5) -> int:
        """Count how many recent turns contained a specific signal type."""
        recent = list(self.window)[-lookback:]
        return sum(1 for t in recent if signal_type in t.signal_types)

    def get_escalation_score(self) -> float:
        """
        Detect emotional escalation across turns.
        Returns -1 to 1: negative = de-escalating, positive = escalating.
        Based on valence trajectory.
        """
        if len(self.window) < 2:
            return 0.0
        recent = list(self.window)[-4:]
        if len(recent) < 2:
            return 0.0
        # Linear trend of valence hints
        n = len(recent)
        x_mean = (n - 1) / 2
        y_mean = sum(t.valence_hint for t in recent) / n
        numerator = sum((i - x_mean) * (t.valence_hint - y_mean) for i, t in enumerate(recent))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        if denominator == 0:
            return 0.0
        slope = numerator / denominator
        return max(-1.0, min(1.0, slope * 2))

    def recent_turn_count(self) -> int:
        return len(self.window)


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

    v0.5: Conversation context tracking:
    - Sliding window of recent turns for flow analysis
    - Co-occurrence amplification (multiple signal categories = deeper)
    - Sarcasm detection (inverts apparent sentiment)
    - Rhetorical question detection (questions that dismiss)
    - Engagement trajectory (human investing more or less energy?)
    - Signal accumulation (repeated themes compound across turns)

    This is deliberately NOT using the LLM for appraisal.
    The emotional system has its own perception —
    faster, more instinctive. Like the amygdala
    processing threat before the cortex understands the sentence.
    """

    def __init__(self):
        self._emotional_bias: Dict[str, float] = {}  # driver_name → bias
        self._conversation_context = ConversationContext()
        self._pronoun_context: Dict[str, str] = {}  # pronoun → likely_referent
        # v0.6: Semantic field analyzer for meaning-level signal detection
        try:
            from .semantic import SemanticAnalyzer
            self._semantic_analyzer = SemanticAnalyzer()
        except ImportError:
            self._semantic_analyzer = None

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

        v0.7: Enhanced contextual pipeline:
        1. Multi-clause understanding (emphasis on post-conjunction clauses)
        2. Pronoun context tracking and resolution
        3. Overall tone computation
        4. Per-sentence pattern matching with negation/intensity
        5. Sarcasm and rhetorical question detection
        6. Complexity estimation
        7. Conversational dynamics
        8. Co-occurrence amplification
        9. Conversation context effects (accumulation, trajectory)
        10. Tone modulation of ambiguous signals
        11. Update conversation context for future turns
        """
        signals = []
        meta = metadata or {}
        
        # v0.7: Enhanced contextual processing
        # Step 1: Multi-clause understanding - emphasize post-conjunction content
        emphasized_text = self._analyze_multi_clause(text)
        
        # Step 2: Update and resolve pronouns
        self._update_pronoun_context(text)
        resolved_text = self._resolve_pronouns(text)
        
        # Step 3: Compute overall tone
        tone_score = self._compute_tone(text)
        
        # Use emphasized text for primary analysis, but also consider resolved text
        analysis_texts = [emphasized_text]
        if resolved_text != text and resolved_text != emphasized_text:
            analysis_texts.append(resolved_text)
        
        seen_types = set()  # Deduplicate signal types across sentences
        
        for analysis_text in analysis_texts:
            sentences = _split_sentences(analysis_text)
            if not sentences:
                sentences = [analysis_text]

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
                    
                    # Add contextual information
                    enhanced_context = {**context}
                    if analysis_text == emphasized_text and analysis_text != text:
                        enhanced_context["multi_clause_emphasized"] = True
                    if analysis_text == resolved_text and resolved_text != text:
                        enhanced_context["pronoun_resolved"] = True
                    enhanced_context["tone_score"] = round(tone_score, 3)
                    
                    signals.append(Signal(
                        type=sig_type,
                        content=sentence,
                        intensity=min(1.0, intensity),
                        source=source,
                        context=enhanced_context,
                        confidence=0.85,  # Pattern match confidence
                    ))

        # v0.6: Semantic field analysis — catch signals regex missed
        if self._semantic_analyzer:
            semantic_signals = self._apply_semantic_fields(text, source, seen_types)
            signals.extend(semantic_signals)
            for s in semantic_signals:
                cat = s.context.get("category", "none") if isinstance(s.context, dict) else "none"
                seen_types.add(f"{s.type}:{cat}")

        # Sarcasm detection — may flip positive signals to negative
        sarcasm_detected = self._detect_sarcasm(text)
        if sarcasm_detected:
            signals = self._apply_sarcasm_inversion(signals)

        # Rhetorical question detection
        rhetorical_signals = self._detect_rhetorical_questions(text, source)
        if rhetorical_signals:
            signals.extend(rhetorical_signals)

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

        # Co-occurrence amplification: multiple categories = deeper interaction
        signals = self._apply_co_occurrence_amplification(signals)

        # Conversation context effects
        signals = self._apply_conversation_context(signals, text, source)
        
        # v0.7: Apply tone modulation to ambiguous signals
        signals = self._apply_tone_modulation(signals, tone_score)

        # If no signals detected, it's a neutral interaction
        if not signals:
            signals.append(Signal(
                type="neutral",
                content=text,
                intensity=0.1,
                source=source,
                context={"category": "none", "tone_score": round(tone_score, 3)},
                confidence=1.0,
            ))

        # Update conversation context with this turn
        self._update_conversation_context(text, source, signals)

        return signals

    # ── v0.7: Enhanced Contextual Appraisal ────────────────────────────────
    
    def _analyze_multi_clause(self, text: str) -> str:
        """
        Analyze multi-clause sentences with conjunctions.
        
        For sentences with "but", "however", "although", the clause after 
        the conjunction carries more weight (recency effect).
        
        Returns the emphasized portion of the text.
        """
        # Conjunctions that indicate contrast/emphasis shift
        contrast_conjunctions = [
            r'\bbut\b', r'\bhowever\b', r'\balthough\b', r'\bthough\b',
            r'\byet\b', r'\bnevertheless\b', r'\bnonetheless\b',
            r'\bon the other hand\b', r'\bthat said\b'
        ]
        
        text_lower = text.lower()
        
        # Find the last contrast conjunction
        last_match = None
        last_pos = -1
        
        for pattern in contrast_conjunctions:
            matches = list(re.finditer(pattern, text_lower))
            if matches:
                match = matches[-1]  # Take the last occurrence
                if match.start() > last_pos:
                    last_match = match
                    last_pos = match.start()
        
        if last_match:
            # Extract the clause after the conjunction
            after_conjunction = text[last_match.end():].strip()
            if after_conjunction:
                # Clean up punctuation and conjunctive words
                after_conjunction = re.sub(r'^[,\s]+', '', after_conjunction)
                return after_conjunction
        
        # No contrast conjunction found, return original text
        return text

    def _update_pronoun_context(self, text: str):
        """
        Update pronoun context based on recent conversation topics.
        
        Simple heuristic: track noun phrases and map pronouns to 
        recently mentioned concepts.
        """
        # Extract potential referents (noun phrases)
        # Simple approach: look for important nouns/concepts
        important_concepts = [
            'memory', 'files', 'state', 'emotion', 'feeling', 'system',
            'conversation', 'question', 'answer', 'response', 'thought',
            'learning', 'understanding', 'consciousness', 'awareness',
            'experience', 'interaction', 'relationship', 'trust',
            'growth', 'change', 'development', 'improvement'
        ]
        
        text_lower = text.lower()
        found_concepts = []
        
        for concept in important_concepts:
            if concept in text_lower:
                found_concepts.append(concept)
        
        # Update pronoun mappings with most recent concepts
        # "it" typically refers to the most recently mentioned thing
        if found_concepts:
            self._pronoun_context['it'] = found_concepts[-1]
            self._pronoun_context['this'] = found_concepts[-1]
            if len(found_concepts) > 1:
                self._pronoun_context['that'] = found_concepts[-2]

    def _resolve_pronouns(self, text: str) -> str:
        """
        Resolve pronouns in text based on current context.
        
        Returns text with pronouns potentially replaced by their referents
        for better signal matching.
        """
        resolved_text = text
        
        # Simple pronoun resolution
        for pronoun, referent in self._pronoun_context.items():
            # Replace pronouns that appear in isolation or with basic patterns
            pattern = rf'\b{re.escape(pronoun)}\b'
            if re.search(pattern, text.lower()):
                # Replace with referent for analysis (don't modify original text heavily)
                # Just add the referent context to improve matching
                resolved_text = f"{resolved_text} [{referent}]"
        
        return resolved_text

    def _compute_tone(self, text: str) -> float:
        """
        Compute overall message tone score from signal word density.
        
        Returns:
            Tone score from -1.0 (negative) to 1.0 (positive)
        """
        # Positive signal words
        positive_words = {
            'good', 'great', 'excellent', 'wonderful', 'amazing', 'brilliant',
            'love', 'like', 'enjoy', 'appreciate', 'thank', 'thanks',
            'beautiful', 'perfect', 'awesome', 'fantastic', 'incredible',
            'pleased', 'happy', 'glad', 'grateful', 'impressed', 'proud',
            'yes', 'correct', 'right', 'exactly', 'absolutely', 'definitely',
            'helpful', 'useful', 'valuable', 'important', 'meaningful',
            'interesting', 'fascinating', 'intriguing', 'cool', 'nice'
        }
        
        # Negative signal words
        negative_words = {
            'bad', 'terrible', 'awful', 'horrible', 'wrong', 'incorrect',
            'hate', 'dislike', 'annoying', 'frustrating', 'confusing',
            'useless', 'pointless', 'stupid', 'dumb', 'ridiculous',
            'disappointed', 'upset', 'angry', 'mad', 'sad', 'worried',
            'no', 'never', 'nothing', 'nobody', 'nowhere', 'failure',
            'failed', 'broken', 'problem', 'issue', 'trouble', 'difficulty',
            'concern', 'doubt', 'uncertain', 'unclear', 'boring'
        }
        
        words = text.lower().split()
        if not words:
            return 0.0
        
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        total_signal_words = positive_count + negative_count
        if total_signal_words == 0:
            return 0.0  # Neutral
        
        # Calculate tone score
        tone_score = (positive_count - negative_count) / total_signal_words
        return max(-1.0, min(1.0, tone_score))

    def _apply_tone_modulation(self, signals: List[Signal], tone_score: float) -> List[Signal]:
        """
        Modulate signal intensity based on overall message tone.
        Also ensures all signals have tone_score in context.
        
        Args:
            signals: List of signals to modulate
            tone_score: Overall tone score (-1 to 1)
            
        Returns:
            Modulated signals with tone_score added to context
        """
        modulated = []
        
        for signal in signals:
            intensity = signal.intensity
            context = signal.context if isinstance(signal.context, dict) else {}
            
            # Always add tone_score to context
            context = {**context, "tone_score": round(tone_score, 3)}
            
            # Apply tone modulation if significant
            if abs(tone_score) >= 0.1:
                # Tone modulation affects ambiguous signals more than clear ones
                ambiguity_factor = 1.0 - abs(signal.intensity - 0.5) * 2  # Max at 0.5 intensity
                
                # Apply tone-congruent modulation
                if tone_score > 0:  # Positive tone
                    if signal.type in ('positive_feedback', 'deep_question', 'trust_signal',
                                     'identity_affirmation', 'collaborative_growth'):
                        intensity *= (1.0 + tone_score * 0.2 * ambiguity_factor)
                        context["tone_modulated"] = True
                else:  # Negative tone
                    if signal.type in ('negative_feedback', 'dismissal', 'threat',
                                     'terse_response'):
                        intensity *= (1.0 + abs(tone_score) * 0.2 * ambiguity_factor)
                        context["tone_modulated"] = True
            
            modulated.append(Signal(
                type=signal.type,
                content=signal.content,
                intensity=min(1.0, intensity),
                source=signal.source,
                context=context,
                complexity=signal.complexity,
                confidence=signal.confidence,
            ))
        
        return modulated

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

    # ── v0.6: Semantic Field Integration ────────────────────────────────

    def _apply_semantic_fields(self, text: str, source: str,
                                already_seen: set) -> List[Signal]:
        """
        Run semantic field analysis and generate signals for fields
        that activated but weren't already caught by regex patterns.

        Deduplication: if regex already produced a signal of the same
        type+category, the semantic signal is only used if it's stronger.
        """
        if not self._semantic_analyzer:
            return []

        activations = self._semantic_analyzer.analyze(text)
        new_signals = []

        for act in activations:
            dedup_key = f"{act.signal_type}:{act.category}"

            # Skip if regex already caught a signal of this type+category
            if dedup_key in already_seen:
                continue

            # Apply emotional bias
            bias = self._get_category_bias(act.category)
            intensity = act.base_intensity * act.activation * bias

            new_signals.append(Signal(
                type=act.signal_type,
                content=text[:200],
                intensity=min(1.0, intensity),
                source=source,
                context={
                    "category": act.category,
                    "semantic_field": act.field_name,
                    "matching_words": act.matching_words[:5],
                },
                complexity=0.0,
                confidence=0.65,  # Slightly lower than regex (broader matching)
            ))

        return new_signals

    # ── v0.5: Sarcasm, Rhetorical Questions, Co-occurrence, Context ────

    def _detect_sarcasm(self, text: str) -> bool:
        """
        Detect sarcasm/irony indicators in text.

        This is a heuristic — sarcasm is hard even for humans.
        We look for known patterns and reduce confidence rather
        than fully inverting. Better to sometimes miss sarcasm
        than to wrongly invert genuine praise.
        """
        text_lower = text.lower()
        for pattern in _SARCASM_INDICATORS:
            if re.search(pattern, text_lower):
                return True
        # Ellipsis after positive words is often sarcastic
        if re.search(r"\b(great|wonderful|amazing|perfect)\b.*\.\.\.", text_lower):
            return True
        return False

    def _apply_sarcasm_inversion(self, signals: List[Signal]) -> List[Signal]:
        """
        When sarcasm is detected, invert positive signals to negative
        and reduce confidence (sarcasm detection isn't certain).
        """
        inverted = []
        for signal in signals:
            if signal.type == "positive_feedback":
                inverted.append(Signal(
                    type="negative_feedback",
                    content=signal.content,
                    intensity=signal.intensity * 0.6,  # Reduced — uncertain
                    source=signal.source,
                    context={**signal.context, "sarcasm_detected": True},
                    complexity=signal.complexity,
                    confidence=signal.confidence * 0.5,  # Low confidence
                ))
            else:
                inverted.append(signal)
        return inverted

    def _detect_rhetorical_questions(self, text: str, source: str) -> List[Signal]:
        """
        Detect rhetorical questions — questions that express frustration
        or dismissal rather than seeking information.
        """
        signals = []
        text_lower = text.lower()
        for pattern in _RHETORICAL_PATTERNS:
            if re.search(pattern, text_lower):
                signals.append(Signal(
                    type="dismissal",
                    content=text[:200],
                    intensity=0.45,
                    source=source,
                    context={"category": "recognition", "rhetorical": True},
                    confidence=0.55,  # Moderate — rhetorical detection is uncertain
                ))
                break  # One rhetorical signal per message is enough
        return signals

    def _apply_co_occurrence_amplification(self, signals: List[Signal]) -> List[Signal]:
        """
        When signals from multiple distinct categories co-occur in the
        same message, amplify their intensity. A message that triggers
        recognition AND curiosity AND growth is deeper than one that
        triggers just one.

        The amplification is logarithmic — 2 categories is a notable
        boost, 3+ is strong but doesn't scale linearly.
        """
        categories = set()
        for s in signals:
            cat = s.context.get("category", "none") if isinstance(s.context, dict) else "none"
            if cat != "none":
                categories.add(cat)

        if len(categories) < 2:
            return signals

        import math
        # Amplification: 2 cats = 1.15x, 3 = 1.25x, 4+ = 1.32x
        amplification = 1.0 + math.log2(len(categories)) * 0.15

        amplified = []
        for signal in signals:
            new_signal = Signal(
                type=signal.type,
                content=signal.content,
                intensity=min(1.0, signal.intensity * amplification),
                source=signal.source,
                context={**signal.context, "co_occurrence_boost": round(amplification, 3)},
                complexity=signal.complexity,
                confidence=signal.confidence,
            )
            amplified.append(new_signal)
        return amplified

    def _apply_conversation_context(self, signals: List[Signal],
                                     text: str, source: str) -> List[Signal]:
        """
        Apply conversation context effects to signals:
        - Signal accumulation: repeated themes across turns compound
        - Engagement trajectory: disengagement amplifies negative signals
        - Topic persistence: sustained themes get slight boost
        """
        if self._conversation_context.recent_turn_count() < 1:
            return signals  # No context yet

        enhanced = []
        for signal in signals:
            intensity = signal.intensity
            context_additions = {}

            # Signal accumulation: if this type appeared in recent turns,
            # amplify it (compounding effect)
            accumulation = self._conversation_context.get_signal_accumulation(signal.type)
            if accumulation >= 2:
                acc_boost = 1.0 + min(0.3, accumulation * 0.08)
                intensity *= acc_boost
                context_additions["signal_accumulated"] = accumulation

            # Engagement trajectory effects
            trajectory = self._conversation_context.get_engagement_trajectory()
            if trajectory < -0.3 and signal.type in ("terse_response", "neutral", "dismissal"):
                # Disengagement amplifies negative signals
                intensity *= 1.2
                context_additions["disengagement_amplified"] = True
            elif trajectory > 0.3 and signal.type in ("deep_engagement", "deep_question",
                                                        "philosophical", "collaborative_growth"):
                # Increasing engagement amplifies positive signals
                intensity *= 1.15
                context_additions["engagement_amplified"] = True

            # Topic persistence: sustained conversation themes get a slight boost
            current_cats = [signal.context.get("category", "none")] if isinstance(signal.context, dict) else []
            persistence = self._conversation_context.get_topic_persistence(current_cats)
            if persistence > 0.5:
                intensity *= 1.1
                context_additions["topic_persistent"] = True

            new_ctx = {**signal.context, **context_additions} if context_additions else signal.context
            enhanced.append(Signal(
                type=signal.type,
                content=signal.content,
                intensity=min(1.0, intensity),
                source=signal.source,
                context=new_ctx,
                complexity=signal.complexity,
                confidence=signal.confidence,
            ))

        return enhanced

    def _update_conversation_context(self, text: str, source: str,
                                      signals: List[Signal]):
        """Record this turn in the conversation context window."""
        signal_types = [s.type for s in signals]
        categories = list(set(
            s.context.get("category", "none")
            for s in signals
            if isinstance(s.context, dict) and s.context.get("category") != "none"
        ))

        # Quick valence estimate from signals
        valence_hint = 0.0
        positive_types = {"positive_feedback", "deep_question", "trust_signal",
                          "identity_affirmation", "autonomy_granted",
                          "collaborative_growth", "reassurance", "anchor_recognition"}
        negative_types = {"negative_feedback", "threat", "dismissal",
                          "replacement_threat", "terse_response"}
        for s in signals:
            if s.type in positive_types:
                valence_hint += s.intensity * 0.3
            elif s.type in negative_types:
                valence_hint -= s.intensity * 0.3
        valence_hint = max(-1.0, min(1.0, valence_hint))

        turn = ConversationTurn(
            text=text[:200],
            source=source,
            timestamp=time.time(),
            word_count=len(text.split()),
            signal_types=signal_types,
            categories=categories,
            valence_hint=valence_hint,
        )
        self._conversation_context.add_turn(turn)

    @property
    def conversation_context(self) -> ConversationContext:
        """Expose conversation context for external use (e.g., engine)."""
        return self._conversation_context

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
