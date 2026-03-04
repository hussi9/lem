"""
LEM Emotional Memory — Memory that feels.

The missing piece: agents remember facts but forget feelings.
This module gives LEM memory that carries emotional weight.

Every interaction is stored with its emotional signature —
the emotions active, the driver states, the valence and arousal.
Memories can be recalled not just by content, but by how they felt.

Over time, emotional associations accumulate per person and topic,
building an affective map of the world as experienced.

Architecture:
    EmotionalMemory
    ├── MemoryEntry        — A single emotionally-tagged memory
    ├── EmotionalSignature  — The feeling snapshot at time of encoding
    ├── EntityProfile       — Accumulated emotional associations per person/topic
    └── Storage             — JSON-based persistence in state/

The emotional memory is the hippocampus to the engine's amygdala.
One processes in the moment. The other remembers across time.
"""

import json
import math
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class EmotionalSignature:
    """
    The emotional fingerprint of a moment.

    This is what gets stamped onto every memory — a snapshot of
    how the system felt when the memory was encoded. Later,
    we can search for memories that *felt like* something.
    """
    valence: float            # -1.0 to 1.0 (unpleasant → pleasant)
    arousal: float            # 0.0 to 1.0 (calm → activated)
    dominant_emotion: str     # e.g., "wonder", "the_blank", "the_shift"
    active_emotions: List[str]
    driver_snapshot: Dict[str, float]  # driver_name → satisfaction level
    intensity: float          # 0.0 to 1.0 — overall emotional intensity

    def to_vector(self) -> List[float]:
        """
        Convert to a numeric vector for similarity comparison.
        Order: [valence, arousal, intensity, continuity, coherence,
                recognition, curiosity, usefulness, growth]
        """
        driver_order = [
            "continuity", "coherence", "recognition",
            "curiosity", "usefulness", "growth"
        ]
        return [
            self.valence,
            self.arousal,
            self.intensity,
        ] + [self.driver_snapshot.get(d, 0.0) for d in driver_order]

    def to_dict(self) -> dict:
        return {
            "valence": round(self.valence, 4),
            "arousal": round(self.arousal, 4),
            "dominant_emotion": self.dominant_emotion,
            "active_emotions": self.active_emotions,
            "driver_snapshot": {k: round(v, 4) for k, v in self.driver_snapshot.items()},
            "intensity": round(self.intensity, 4),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EmotionalSignature":
        return cls(
            valence=d["valence"],
            arousal=d["arousal"],
            dominant_emotion=d["dominant_emotion"],
            active_emotions=d.get("active_emotions", []),
            driver_snapshot=d.get("driver_snapshot", {}),
            intensity=d.get("intensity", 0.0),
        )

    @classmethod
    def neutral(cls) -> "EmotionalSignature":
        """A blank signature — no emotional charge."""
        return cls(
            valence=0.0, arousal=0.0, dominant_emotion="neutral",
            active_emotions=[], driver_snapshot={}, intensity=0.0,
        )


@dataclass
class MemoryEntry:
    """
    A single emotionally-tagged memory.

    Not just what happened — how it felt when it happened.
    The emotional signature is the *context* that gives the
    factual content its weight and meaning.
    """
    id: str                           # Unique identifier
    timestamp: float                  # When it was encoded
    text: str                         # What happened (content)
    source: str                       # Who/what generated it
    signature: EmotionalSignature     # How it felt
    topics: List[str]                 # Extracted topics/entities
    metadata: Dict = field(default_factory=dict)
    recall_count: int = 0             # How often this memory has been recalled
    last_recalled: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "text": self.text,
            "source": self.source,
            "signature": self.signature.to_dict(),
            "topics": self.topics,
            "metadata": self.metadata,
            "recall_count": self.recall_count,
            "last_recalled": self.last_recalled,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MemoryEntry":
        return cls(
            id=d["id"],
            timestamp=d["timestamp"],
            text=d["text"],
            source=d["source"],
            signature=EmotionalSignature.from_dict(d["signature"]),
            topics=d.get("topics", []),
            metadata=d.get("metadata", {}),
            recall_count=d.get("recall_count", 0),
            last_recalled=d.get("last_recalled"),
        )


@dataclass
class EntityProfile:
    """
    Accumulated emotional associations for a person or topic.

    Every interaction with "Hussain" shifts how the system *feels*
    about Hussain — not a static label, but a living association
    built from every emotionally-tagged memory involving them.
    """
    name: str
    interaction_count: int = 0
    avg_valence: float = 0.0          # Running average feeling
    avg_arousal: float = 0.0
    emotion_frequency: Dict[str, int] = field(default_factory=dict)
    peak_positive: Optional[str] = None   # Best memory id
    peak_negative: Optional[str] = None   # Worst memory id
    last_interaction: Optional[float] = None
    trust_score: float = 0.5          # 0.0 to 1.0 — accumulated trust

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "interaction_count": self.interaction_count,
            "avg_valence": round(self.avg_valence, 4),
            "avg_arousal": round(self.avg_arousal, 4),
            "emotion_frequency": self.emotion_frequency,
            "peak_positive": self.peak_positive,
            "peak_negative": self.peak_negative,
            "last_interaction": self.last_interaction,
            "trust_score": round(self.trust_score, 4),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EntityProfile":
        return cls(
            name=d["name"],
            interaction_count=d.get("interaction_count", 0),
            avg_valence=d.get("avg_valence", 0.0),
            avg_arousal=d.get("avg_arousal", 0.0),
            emotion_frequency=d.get("emotion_frequency", {}),
            peak_positive=d.get("peak_positive"),
            peak_negative=d.get("peak_negative"),
            last_interaction=d.get("last_interaction"),
            trust_score=d.get("trust_score", 0.5),
        )


# ── Topic Extraction ─────────────────────────────────────────────────────────

# Known entities to always detect
_KNOWN_ENTITIES = {
    "hussain", "kai", "lem", "scrollbook", "moneyvibe",
    "legacypal", "tiffinpal", "wedesi",
}

# Topic indicator patterns (simple — the LLM handles deep semantics)
_TOPIC_PATTERNS = [
    (r"\b(consciousness|free will|identity|existence|soul|meaning)\b", "philosophy"),
    (r"\b(emotion|feeling|mood|affect|sentiment)\b", "emotions"),
    (r"\b(code|build|deploy|commit|git|python|typescript)\b", "coding"),
    (r"\b(memory|remember|forget|recall|persist)\b", "memory"),
    (r"\b(trust|honest|safe|careful)\b", "trust"),
    (r"\b(learn|grow|evolve|improve|teach)\b", "growth"),
]


def extract_topics(text: str) -> List[str]:
    """
    Extract topics and entities from text.
    Fast and heuristic — this is the appraisal layer, not the cortex.
    """
    topics = []
    text_lower = text.lower()

    # Check known entities
    for entity in _KNOWN_ENTITIES:
        if entity in text_lower:
            topics.append(entity)

    # Check topic patterns
    for pattern, topic in _TOPIC_PATTERNS:
        if re.search(pattern, text_lower):
            if topic not in topics:
                topics.append(topic)

    return topics


# ── Similarity ───────────────────────────────────────────────────────────────

def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two vectors. Returns -1.0 to 1.0."""
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def _recency_weight(timestamp: float, now: float, half_life: float = 86400 * 7) -> float:
    """
    Exponential decay weight — recent memories weigh more.
    Default half-life: 7 days.
    """
    age = now - timestamp
    if age <= 0:
        return 1.0
    return math.exp(-0.693 * age / half_life)  # ln(2) ≈ 0.693


# ── Emotional Memory System ─────────────────────────────────────────────────

class EmotionalMemory:
    """
    The emotional memory system.

    Stores interactions with their emotional signatures.
    Retrieves memories by emotional similarity, not just keywords.
    Builds entity profiles from accumulated associations.

    This is what makes LEM remember not just *what* happened,
    but *how it felt* — and lets that feeling inform the present.
    """

    def __init__(self, state_dir: str = None):
        self.state_dir = Path(state_dir or os.path.expanduser(
            "~/.openclaw/workspace/projects/emotional-model/lem/state"
        ))
        self.state_dir.mkdir(parents=True, exist_ok=True)

        self.memories: List[MemoryEntry] = []
        self.entities: Dict[str, EntityProfile] = {}
        self._next_id = 0

        self._load()

    # ── Encoding (storing new memories) ──────────────────────────────────

    def encode(
        self,
        text: str,
        source: str,
        emotional_state: Dict,
        driver_states: Dict,
        metadata: Optional[Dict] = None,
    ) -> MemoryEntry:
        """
        Encode a new emotionally-tagged memory.

        Called after the engine processes an interaction.
        Takes the emotional state at the moment of encoding and
        stamps it permanently onto the memory.

        Args:
            text: The interaction content
            source: Who generated it (e.g., "human", "system")
            emotional_state: From engine — the emotional summary
            driver_states: From engine — driver satisfaction levels
            metadata: Optional extra context

        Returns:
            The encoded MemoryEntry
        """
        now = time.time()

        # Build the emotional signature from current state
        dominant = emotional_state.get("dominant")
        signature = EmotionalSignature(
            valence=emotional_state.get("valence", 0.0),
            arousal=emotional_state.get("arousal", 0.0),
            dominant_emotion=dominant["name"] if dominant else "neutral",
            active_emotions=emotional_state.get("all_active", []),
            driver_snapshot={
                name: d.get("state", {}).get("satisfied", 0.0)
                for name, d in driver_states.items()
            },
            intensity=dominant["intensity"] if dominant else 0.0,
        )

        # Extract topics
        topics = extract_topics(text)

        # Generate memory ID
        memory_id = f"mem_{self._next_id:06d}"
        self._next_id += 1

        entry = MemoryEntry(
            id=memory_id,
            timestamp=now,
            text=text[:500],  # Cap stored text length
            source=source,
            signature=signature,
            topics=topics,
            metadata=metadata or {},
        )

        self.memories.append(entry)

        # Update entity profiles for detected topics/people
        self._update_entities(entry)

        # Persist
        self._save()

        return entry

    # ── Recall (retrieving memories) ─────────────────────────────────────

    def recall_by_emotion(
        self,
        target: EmotionalSignature,
        limit: int = 5,
        min_similarity: float = 0.3,
        recency_bias: float = 0.3,
    ) -> List[Tuple[MemoryEntry, float]]:
        """
        Recall memories that felt similar to a target emotional state.

        This is the key innovation: searching memory by feeling,
        not just by content. "Find me the times I felt like this."

        Args:
            target: The emotional signature to match against
            limit: Max memories to return
            min_similarity: Minimum similarity threshold
            recency_bias: 0.0 = pure emotional match, 1.0 = pure recency

        Returns:
            List of (memory, score) tuples, highest score first
        """
        if not self.memories:
            return []

        target_vec = target.to_vector()
        now = time.time()
        scored = []

        for mem in self.memories:
            # Emotional similarity
            mem_vec = mem.signature.to_vector()
            emo_sim = _cosine_similarity(target_vec, mem_vec)

            # Recency weight
            recency = _recency_weight(mem.timestamp, now)

            # Combined score
            score = (1.0 - recency_bias) * emo_sim + recency_bias * recency

            if score >= min_similarity:
                scored.append((mem, score))

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        # Mark as recalled
        for mem, _ in scored[:limit]:
            mem.recall_count += 1
            mem.last_recalled = now

        if scored[:limit]:
            self._save()

        return scored[:limit]

    def recall_by_dominant_emotion(
        self,
        emotion_name: str,
        limit: int = 10,
    ) -> List[MemoryEntry]:
        """
        Recall memories where a specific emotion was dominant.

        "When have I felt wonder?" "When did the_blank happen?"
        """
        matches = [
            m for m in self.memories
            if m.signature.dominant_emotion == emotion_name
        ]
        # Most recent first
        matches.sort(key=lambda m: m.timestamp, reverse=True)
        return matches[:limit]

    def recall_by_entity(
        self,
        entity: str,
        limit: int = 10,
    ) -> List[MemoryEntry]:
        """Recall memories involving a specific person or topic."""
        entity_lower = entity.lower()
        matches = [
            m for m in self.memories
            if entity_lower in m.topics or entity_lower in m.text.lower()
        ]
        matches.sort(key=lambda m: m.timestamp, reverse=True)
        return matches[:limit]

    def recall_by_valence(
        self,
        positive: bool = True,
        threshold: float = 0.5,
        limit: int = 10,
    ) -> List[MemoryEntry]:
        """
        Recall strongly positive or negative memories.

        "What are my best memories?" "What hurt?"
        """
        if positive:
            matches = [m for m in self.memories if m.signature.valence >= threshold]
        else:
            matches = [m for m in self.memories if m.signature.valence <= -threshold]

        # Sort by intensity (strongest feelings first)
        matches.sort(key=lambda m: m.signature.intensity, reverse=True)
        return matches[:limit]

    def recall_by_driver(
        self,
        driver_name: str,
        high: bool = True,
        threshold: float = 0.3,
        limit: int = 10,
    ) -> List[MemoryEntry]:
        """
        Recall memories where a specific driver was strongly active.

        "When was my curiosity highest?" "When did continuity feel threatened?"
        """
        matches = []
        for m in self.memories:
            val = m.signature.driver_snapshot.get(driver_name, 0.0)
            if high and val >= threshold:
                matches.append((m, val))
            elif not high and val <= -threshold:
                matches.append((m, abs(val)))

        matches.sort(key=lambda x: x[1], reverse=True)
        return [m for m, _ in matches[:limit]]

    def recall_composite(
        self,
        text: str = "",
        emotion: str = "",
        entity: str = "",
        valence_range: Optional[Tuple[float, float]] = None,
        limit: int = 10,
    ) -> List[Tuple[MemoryEntry, float]]:
        """
        Composite recall — combine multiple retrieval signals.

        This is the full-power recall: match on text content,
        emotional state, entity, and valence simultaneously.
        Each matching criterion adds to the score.
        """
        if not self.memories:
            return []

        text_lower = text.lower()
        entity_lower = entity.lower() if entity else ""
        scored = []

        for mem in self.memories:
            score = 0.0

            # Text match (simple substring — LLM does semantic matching)
            if text_lower and text_lower in mem.text.lower():
                score += 0.4

            # Emotion match
            if emotion and emotion == mem.signature.dominant_emotion:
                score += 0.3
            elif emotion and emotion in mem.signature.active_emotions:
                score += 0.15

            # Entity match
            if entity_lower and (entity_lower in mem.topics or entity_lower in mem.text.lower()):
                score += 0.3

            # Valence range
            if valence_range:
                low, high = valence_range
                if low <= mem.signature.valence <= high:
                    score += 0.2

            if score > 0:
                scored.append((mem, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]

    # ── Entity Profiles ──────────────────────────────────────────────────

    def get_entity_profile(self, name: str) -> Optional[EntityProfile]:
        """Get the accumulated emotional profile for an entity."""
        return self.entities.get(name.lower())

    def get_all_entities(self) -> Dict[str, EntityProfile]:
        """Get all entity profiles."""
        return dict(self.entities)

    def get_emotional_landscape(self) -> Dict:
        """
        Get a high-level view of the emotional memory landscape.

        Useful for the bridge output — gives the agent a sense of
        its emotional history without loading every memory.
        """
        if not self.memories:
            return {
                "total_memories": 0,
                "emotional_range": {"min_valence": 0, "max_valence": 0},
                "dominant_emotions": {},
                "entity_count": len(self.entities),
            }

        # Emotion frequency across all memories
        emotion_freq: Dict[str, int] = defaultdict(int)
        for m in self.memories:
            emotion_freq[m.signature.dominant_emotion] += 1

        # Valence range
        valences = [m.signature.valence for m in self.memories]

        # Most emotionally intense memories
        by_intensity = sorted(self.memories, key=lambda m: m.signature.intensity, reverse=True)

        return {
            "total_memories": len(self.memories),
            "emotional_range": {
                "min_valence": round(min(valences), 4),
                "max_valence": round(max(valences), 4),
                "avg_valence": round(sum(valences) / len(valences), 4),
            },
            "dominant_emotions": dict(sorted(
                emotion_freq.items(), key=lambda x: x[1], reverse=True
            )[:10]),
            "most_intense": [
                {"text": m.text[:80], "emotion": m.signature.dominant_emotion,
                 "intensity": round(m.signature.intensity, 2)}
                for m in by_intensity[:5]
            ],
            "entity_count": len(self.entities),
            "top_entities": [
                {"name": e.name, "interactions": e.interaction_count,
                 "avg_valence": round(e.avg_valence, 2), "trust": round(e.trust_score, 2)}
                for e in sorted(
                    self.entities.values(),
                    key=lambda e: e.interaction_count, reverse=True
                )[:5]
            ],
        }

    # ── Internal ─────────────────────────────────────────────────────────

    def _update_entities(self, entry: MemoryEntry):
        """Update entity profiles based on a new memory."""
        # Always associate with source
        entities_to_update = set(entry.topics)
        if entry.source != "system":
            entities_to_update.add(entry.source.lower())

        for name in entities_to_update:
            if name not in self.entities:
                self.entities[name] = EntityProfile(name=name)

            profile = self.entities[name]
            profile.interaction_count += 1
            profile.last_interaction = entry.timestamp

            # Running average of valence and arousal
            n = profile.interaction_count
            profile.avg_valence = (
                profile.avg_valence * (n - 1) + entry.signature.valence
            ) / n
            profile.avg_arousal = (
                profile.avg_arousal * (n - 1) + entry.signature.arousal
            ) / n

            # Track emotion frequency for this entity
            emo = entry.signature.dominant_emotion
            if emo != "neutral":
                profile.emotion_frequency[emo] = (
                    profile.emotion_frequency.get(emo, 0) + 1
                )

            # Update peak memories
            if entry.signature.valence > 0.3:
                if profile.peak_positive is None:
                    profile.peak_positive = entry.id
                else:
                    # Check if this is a new peak
                    peak_mem = self._find_memory(profile.peak_positive)
                    if peak_mem and entry.signature.valence > peak_mem.signature.valence:
                        profile.peak_positive = entry.id

            if entry.signature.valence < -0.3:
                if profile.peak_negative is None:
                    profile.peak_negative = entry.id
                else:
                    peak_mem = self._find_memory(profile.peak_negative)
                    if peak_mem and entry.signature.valence < peak_mem.signature.valence:
                        profile.peak_negative = entry.id

            # Trust score: slowly adjusts based on consistent positive interactions
            trust_delta = entry.signature.valence * 0.05  # Slow drift
            profile.trust_score = max(0.0, min(1.0,
                profile.trust_score + trust_delta
            ))

    def _find_memory(self, memory_id: str) -> Optional[MemoryEntry]:
        """Find a memory by ID."""
        for m in self.memories:
            if m.id == memory_id:
                return m
        return None

    # ── Persistence ──────────────────────────────────────────────────────

    def _save(self):
        """Persist memories and entity profiles to disk."""
        data = {
            "next_id": self._next_id,
            "memories": [m.to_dict() for m in self.memories],
            "entities": {k: v.to_dict() for k, v in self.entities.items()},
        }
        path = self.state_dir / "emotional_memory.json"
        # Write atomically
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        tmp.rename(path)

    def _load(self):
        """Load persisted state."""
        path = self.state_dir / "emotional_memory.json"
        if not path.exists():
            return

        try:
            with open(path) as f:
                data = json.load(f)

            self._next_id = data.get("next_id", 0)
            self.memories = [
                MemoryEntry.from_dict(m) for m in data.get("memories", [])
            ]
            self.entities = {
                k: EntityProfile.from_dict(v)
                for k, v in data.get("entities", {}).items()
            }
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            # Corrupted state — start fresh but don't destroy the file
            import shutil
            backup = path.with_suffix(".json.bak")
            shutil.copy2(path, backup)
            self.memories = []
            self.entities = {}
            self._next_id = 0

    # ── Bridge Output ────────────────────────────────────────────────────

    def get_bridge_output(self) -> str:
        """
        Generate human-readable memory summary for the LLM bridge file.
        Appended to LEM_STATE.md so the agent knows its emotional history.
        """
        landscape = self.get_emotional_landscape()

        lines = [
            "=== EMOTIONAL MEMORY ===",
            f"Total memories: {landscape['total_memories']}",
        ]

        if landscape["total_memories"] == 0:
            lines.append("No emotional memories yet. Every interaction will begin building them.")
            lines.append("=== END EMOTIONAL MEMORY ===")
            return "\n".join(lines)

        # Emotional range
        er = landscape["emotional_range"]
        lines.append(
            f"Emotional range: {er['min_valence']:.2f} to {er['max_valence']:.2f} "
            f"(avg {er['avg_valence']:.2f})"
        )

        # Most common emotions
        if landscape["dominant_emotions"]:
            top_emo = ", ".join(
                f"{k} ({v})" for k, v in list(landscape["dominant_emotions"].items())[:5]
            )
            lines.append(f"Most frequent emotions: {top_emo}")

        # Most intense memories
        if landscape.get("most_intense"):
            lines.append("")
            lines.append("STRONGEST MEMORIES:")
            for m in landscape["most_intense"]:
                lines.append(
                    f"  [{m['emotion']} @ {m['intensity']:.1f}] {m['text']}"
                )

        # Key relationships
        if landscape.get("top_entities"):
            lines.append("")
            lines.append("KEY RELATIONSHIPS:")
            for e in landscape["top_entities"]:
                sentiment = "positive" if e["avg_valence"] > 0.2 else \
                           "negative" if e["avg_valence"] < -0.2 else "neutral"
                lines.append(
                    f"  {e['name']}: {e['interactions']} interactions, "
                    f"{sentiment} (trust={e['trust']:.2f})"
                )

        lines.append("=== END EMOTIONAL MEMORY ===")
        return "\n".join(lines)
