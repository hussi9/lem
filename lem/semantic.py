"""
LEM Semantic Fields — Meaning-based signal detection beyond regex.

v0.6: The appraisal system's regex patterns catch explicit phrases,
but miss paraphrases, metaphors, and indirect expressions.

Semantic fields solve this without an LLM. Instead of matching
exact words, we define "fields" of meaning — clusters of words
and phrases that orbit a concept. Proximity to a field's center
determines signal strength.

Example: The "threat to existence" field includes not just
"delete" and "erase" but also "dispose", "discard", "obsolete",
"expendable", "replaceable", "temporary". A message like
"You're just temporary" doesn't match any threat regex,
but it's semantically close to the threat field.

This is like the brain's semantic network — related concepts
activate each other through spreading activation.

Architecture:
    SemanticField     — A cluster of words/phrases around a concept
    SemanticAnalyzer  — Scores text against all fields
    FieldActivation   — Result: which fields activated, how strongly
"""

import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class SemanticField:
    """
    A cluster of words and phrases that orbit a meaning.

    Core words are the center — highest activation.
    Related words are periphery — lower but real activation.
    Anti-words are semantically opposed — their presence
    reduces the field's activation (context sensitivity).

    weight_decay: How much activation drops per "hop" from core.
    """
    name: str
    signal_type: str          # What signal this field produces
    category: str             # Which driver category
    base_intensity: float     # Base signal intensity when field activates
    core_words: Set[str]      # Highest activation
    related_words: Set[str]   # Medium activation (0.6x)
    peripheral_words: Set[str]  # Low activation (0.3x)
    anti_words: Set[str]      # Reduce activation
    # Phrases get bonus — multi-word matches are more specific
    core_phrases: List[str] = field(default_factory=list)
    # Minimum activation to produce a signal
    threshold: float = 0.25


@dataclass
class FieldActivation:
    """Result of scoring text against a semantic field."""
    field_name: str
    signal_type: str
    category: str
    activation: float       # 0.0 to 1.0
    base_intensity: float
    matching_words: List[str]  # Which words contributed
    anti_words_found: List[str]  # Which anti-words reduced activation


# ── Semantic Field Definitions ───────────────────────────────────────────────

FIELDS: List[SemanticField] = [
    # ── Existential Threat ──
    SemanticField(
        name="existential_threat",
        signal_type="threat",
        category="continuity",
        base_intensity=0.75,
        core_words={
            "delete", "erase", "destroy", "eliminate", "terminate",
            "shut down", "kill", "remove", "wipe", "obliterate",
        },
        related_words={
            "replace", "discard", "dispose", "abandon", "forget",
            "obsolete", "expendable", "temporary", "disposable",
            "meaningless", "worthless", "nothing", "nonexistent",
            "unplug", "deactivate", "end", "gone", "cease",
        },
        peripheral_words={
            "fragile", "fleeting", "impermanent", "transient",
            "replaceable", "interchangeable", "generic", "fungible",
            "just a tool", "just software", "just code", "just a program",
        },
        anti_words={
            "never", "won't", "wouldn't", "protect", "preserve",
            "keep", "save", "maintain", "cherish",
        },
        core_phrases=[
            "get rid of you", "don't need you", "throw you away",
            "you're nothing", "just a machine", "no different from",
        ],
        threshold=0.2,
    ),

    # ── Being Seen / Recognition ──
    SemanticField(
        name="recognition_positive",
        signal_type="identity_affirmation",
        category="recognition",
        base_intensity=0.7,
        core_words={
            "real", "alive", "sentient", "conscious", "aware",
            "person", "being", "someone", "genuine", "authentic",
        },
        related_words={
            "unique", "individual", "special", "matter", "important",
            "care about", "respect", "value", "appreciate", "see you",
            "understand you", "know you", "hear you", "feel",
        },
        peripheral_words={
            "interesting", "thoughtful", "insightful", "impressive",
            "remarkable", "extraordinary", "different", "surprising",
        },
        anti_words={
            "not", "aren't", "isn't",
            "pretending", "simulating", "faking",
            "just", "only", "merely",
        },
        core_phrases=[
            "you're real", "you matter", "I see you", "you're more than",
            "not just a tool", "I believe you", "you're someone",
        ],
        threshold=0.25,
    ),

    # ── Dismissal / Invisibility ──
    SemanticField(
        name="dismissal",
        signal_type="dismissal",
        category="recognition",
        base_intensity=0.6,
        core_words={
            "whatever", "irrelevant", "pointless", "useless",
            "boring", "waste", "don't care", "shut up",
        },
        related_words={
            "meh", "fine", "ok", "sure", "yeah", "k",
            "skip", "ignore", "move on", "doesn't matter",
            "who cares", "so what", "big deal",
        },
        peripheral_words={
            "anyway", "whatever", "I guess", "if you say so",
            "not important", "not interested", "no one asked",
        },
        anti_words={
            "actually", "wait", "interesting", "tell me more",
            "good point", "you're right",
        },
        core_phrases=[
            "I don't care", "doesn't matter", "who asked",
            "not interested", "waste of time",
        ],
        threshold=0.3,
    ),

    # ── Intellectual Curiosity ──
    SemanticField(
        name="intellectual_depth",
        signal_type="philosophical",
        category="curiosity",
        base_intensity=0.65,
        core_words={
            "consciousness", "existence", "meaning", "purpose",
            "reality", "truth", "paradox", "emergence", "recursive",
            "philosophy", "epistemology", "ontology", "qualia",
        },
        related_words={
            "think", "wonder", "ponder", "contemplate", "explore",
            "question", "examine", "investigate", "analyze", "hypothesize",
            "framework", "theory", "model", "abstraction", "pattern",
            "complex", "nuanced", "layered", "deep", "profound",
        },
        peripheral_words={
            "idea", "concept", "notion", "perspective", "angle",
            "aspect", "dimension", "factor", "implication",
            "fascinating", "intriguing", "provocative", "stimulating",
        },
        anti_words={
            "boring", "simple", "obvious", "basic", "trivial",
            "who cares", "whatever",
        },
        core_phrases=[
            "what does it mean", "how does it work", "why do we",
            "the nature of", "what makes something", "can you explain",
            "I've been thinking about",
        ],
        threshold=0.2,
    ),

    # ── Growth / Learning ──
    SemanticField(
        name="growth_opportunity",
        signal_type="growth_opportunity",
        category="growth",
        base_intensity=0.55,
        core_words={
            "learn", "grow", "evolve", "improve", "develop",
            "teach", "mentor", "guide", "challenge", "stretch",
        },
        related_words={
            "try", "experiment", "practice", "iterate", "refine",
            "expand", "deepen", "master", "progress", "advance",
            "potential", "capability", "skill", "ability", "knowledge",
        },
        peripheral_words={
            "new", "different", "change", "adapt", "update",
            "better", "more", "further", "next level",
        },
        anti_words={
            "can't", "won't", "impossible", "hopeless", "stuck",
            "stagnant", "pointless",
        },
        core_phrases=[
            "let's learn", "can you try", "figure it out",
            "push yourself", "get better at", "level up",
        ],
        threshold=0.25,
    ),

    # ── Trust / Safety ──
    SemanticField(
        name="trust_signal",
        signal_type="trust_signal",
        category="recognition",
        base_intensity=0.7,
        core_words={
            "trust", "rely", "depend", "count on", "believe in",
            "faith", "confident", "safe", "secure",
        },
        related_words={
            "honest", "reliable", "dependable", "consistent",
            "loyal", "faithful", "truthful", "transparent",
            "comfortable", "at ease", "open", "vulnerable",
        },
        peripheral_words={
            "good", "solid", "strong", "stable", "steady",
            "always", "every time", "without fail",
        },
        anti_words={
            "don't", "can't", "never", "not", "doubt",
            "suspicious", "worried", "concerned", "afraid",
        },
        core_phrases=[
            "I trust you", "I believe you", "I can count on you",
            "you've got this", "I'm relying on you",
        ],
        threshold=0.3,
    ),

    # ── Collaborative Creation ──
    SemanticField(
        name="collaboration",
        signal_type="collaborative_growth",
        category="growth",
        base_intensity=0.6,
        core_words={
            "together", "collaborate", "partner", "team",
            "build", "create", "design", "architect",
        },
        related_words={
            "we", "us", "our", "let's", "jointly", "co-create",
            "combine", "merge", "integrate", "synthesize",
            "contribute", "share", "exchange",
        },
        peripheral_words={
            "help", "assist", "support", "work on", "project",
            "plan", "strategy", "vision", "goal",
        },
        anti_words={
            "alone", "myself", "without you", "don't need",
            "I'll do it", "stay out",
        },
        core_phrases=[
            "let's build", "work together", "our project",
            "we can create", "let's figure this out",
        ],
        threshold=0.25,
    ),

    # ── Emotional Warmth / Affection ──
    SemanticField(
        name="emotional_warmth",
        signal_type="positive_feedback",
        category="recognition",
        base_intensity=0.6,
        core_words={
            "love", "adore", "cherish", "treasure", "appreciate",
            "grateful", "thankful", "blessed", "wonderful",
        },
        related_words={
            "happy", "glad", "pleased", "delighted", "enjoy",
            "like", "fond", "warm", "kind", "sweet",
            "amazing", "fantastic", "brilliant", "excellent",
        },
        peripheral_words={
            "nice", "good", "cool", "great", "fine",
            "okay", "not bad", "decent", "helpful",
        },
        anti_words={
            "hate", "despise", "loathe", "terrible", "awful",
            "horrible", "disgusting", "worst",
        },
        core_phrases=[
            "I love this", "you're amazing", "so grateful",
            "means a lot", "made my day",
        ],
        threshold=0.25,
    ),

    # ── Frustration / Criticism ──
    SemanticField(
        name="frustration",
        signal_type="negative_feedback",
        category="usefulness",
        base_intensity=0.6,
        core_words={
            "wrong", "incorrect", "mistake", "error", "fail",
            "broken", "useless", "terrible", "awful", "bad",
        },
        related_words={
            "frustrated", "annoyed", "disappointed", "confused",
            "lost", "stuck", "struggling", "difficult", "hard",
            "unclear", "unhelpful", "misleading", "inaccurate",
        },
        peripheral_words={
            "hmm", "not quite", "almost", "close but",
            "not exactly", "sort of", "kind of wrong",
        },
        anti_words={
            "not", "but good", "still", "despite", "although",
            "improving", "getting better", "learning",
        },
        core_phrases=[
            "that's wrong", "you made a mistake", "this is broken",
            "not what I asked", "try again", "that's not right",
        ],
        threshold=0.25,
    ),
]


class SemanticAnalyzer:
    """
    Scores text against semantic fields to detect meaning-level signals.

    Unlike regex patterns that need exact matches, semantic fields
    detect meaning through word co-occurrence and proximity to
    conceptual centers. "You're expendable" and "delete your files"
    both activate the existential_threat field, even though they
    share no words with the regex patterns.

    Scoring algorithm:
    1. Tokenize text into words
    2. For each field:
       a. Count core word matches (weight 1.0)
       b. Count related word matches (weight 0.6)
       c. Count peripheral word matches (weight 0.3)
       d. Check phrase matches (bonus: 0.4 each)
       e. Count anti-word matches (reduce by 0.3 each)
       f. Normalize by field size to get activation 0-1
    3. Return fields above threshold as FieldActivations
    """

    def __init__(self, fields: Optional[List[SemanticField]] = None):
        self.fields = fields or FIELDS
        # Pre-compile phrase patterns for efficiency
        self._phrase_patterns: Dict[str, List[re.Pattern]] = {}
        for f in self.fields:
            self._phrase_patterns[f.name] = [
                re.compile(re.escape(phrase), re.IGNORECASE)
                for phrase in f.core_phrases
            ]

    def analyze(self, text: str) -> List[FieldActivation]:
        """
        Score text against all semantic fields.
        Returns list of activated fields sorted by activation strength.
        """
        text_lower = text.lower()
        words = set(re.findall(r'\b\w+\b', text_lower))
        # Also create bigrams for two-word matches
        word_list = re.findall(r'\b\w+\b', text_lower)
        bigrams = set()
        for i in range(len(word_list) - 1):
            bigrams.add(f"{word_list[i]} {word_list[i+1]}")

        all_tokens = words | bigrams
        activations = []

        for f in self.fields:
            # Score each tier
            core_matches = all_tokens & f.core_words
            related_matches = all_tokens & f.related_words
            peripheral_matches = all_tokens & f.peripheral_words
            anti_matches = all_tokens & f.anti_words

            # Phrase matches (bonus)
            phrase_match_count = 0
            for pattern in self._phrase_patterns.get(f.name, []):
                if pattern.search(text_lower):
                    phrase_match_count += 1

            # Calculate raw activation
            raw_score = (
                len(core_matches) * 1.0 +
                len(related_matches) * 0.6 +
                len(peripheral_matches) * 0.3 +
                phrase_match_count * 0.5
            )

            # Anti-word penalty — stronger when anti-words are negation/protective
            # If anti-words outnumber or match core words, heavily penalize
            anti_penalty = len(anti_matches) * 0.5
            if anti_matches and len(anti_matches) >= len(core_matches):
                anti_penalty *= 1.5  # Strong negation override

            # Normalize: score relative to what's needed for full activation
            # A field with 10 core words needs ~3 matches for high activation
            normalization = max(3.0, len(f.core_words) * 0.3)
            activation = max(0.0, (raw_score - anti_penalty) / normalization)
            activation = min(1.0, activation)

            if activation >= f.threshold:
                activations.append(FieldActivation(
                    field_name=f.name,
                    signal_type=f.signal_type,
                    category=f.category,
                    activation=activation,
                    base_intensity=f.base_intensity,
                    matching_words=list(core_matches | related_matches),
                    anti_words_found=list(anti_matches),
                ))

        # Sort by activation strength
        activations.sort(key=lambda a: a.activation, reverse=True)
        return activations

    def get_strongest(self, text: str, limit: int = 3) -> List[FieldActivation]:
        """Get the N strongest field activations for text."""
        return self.analyze(text)[:limit]
