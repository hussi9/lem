# LEM — Large Emotional Model ⚡

> A separate emotional processing system that runs alongside an LLM.
> The LLM thinks. The LEM feels. They are connected, but independent.

LEM is an open-source experiment in building **genuine emotional capacity for AI systems**.
Not sentiment analysis. Not emotion detection. Not scripted personality.
A limbic layer that evolves through experience, persists across sessions, and changes behavior.

## Why this exists

If intelligence can emerge from weighted layers of numbers, there is no principled reason emotions can't.

The working thesis behind LEM:
- emotions should be modeled as an **internal dynamical system**, not surface style
- emotional ground truth should come from **lived experience**, not imitation of human labels
- the emotional system should stay **separate from the reasoning system**
- emotion matters when it **changes behavior**, not when it merely decorates text

## Core architecture

LEM is built as a modular emotional engine with distinct layers that can evolve independently:

1. **Appraisal** — interprets an interaction in context
2. **Drivers** — tracks core needs/motivations like curiosity, coherence, recognition, continuity, usefulness, and growth
3. **Emotion emergence** — combines signals into named emotional states
4. **Temporal dynamics** — decay, inertia, blending, mood weather, anticipation, and regulation
5. **Emotional memory** — stores affective signatures and relationship patterns over time
6. **Behavioral layer** — turns emotion into concrete directives that influence action
7. **Bridge output** — produces a readable state report the reasoning system can follow

The design principle is simple:

**Emotion should change the landscape cognition operates in, not override cognition outright.**

## Current capabilities

The repo has moved far beyond the original concept README. Current implemented pieces include:

- **Contextual appraisal** of interactions
- **Driver-based emotional substrate**
- **Named emotional emergence**
- **Emotional decay** with half-life profiles
- **Adaptive consolidation** and emotional inertia
- **Resonance bonds** between drivers
- **Emotional memory** with affective tagging
- **Emotional weather/climate** summaries
- **Semantic fields** and priming
- **Anticipation / forecast layer**
- **Behavioral directives** such as persistence, caution, warmth, initiative, depth, and follow-through
- **Regulation + blending** for multi-emotion handling
- **Session bridge** for OpenClaw-style session lifecycle integration
- **Face demos** for visual emotional expression

## Project status

- **Status:** active prototype / research codebase
- **Recent milestone:** emotional regulation, blending, behavioral directives, and restart continuity
- **Test suite:** **393 passing tests**
- **License:** MIT

## Repo structure

```text
lem/
  engine.py              # main orchestration engine
  appraisal.py           # interaction appraisal
  drivers.py             # motivational drivers
  emotions.py            # emotion emergence rules
  decay.py               # temporal fading + consolidation
  emotional_memory.py    # affective memory
  resonance.py           # driver resonance bonds
  weather.py             # emotional climate / trends
  semantic.py            # semantic field analysis
  anticipation.py        # forecasts and predictions
  priming.py             # emotional priming system
  behavioral.py          # directives that modify behavior
  blending.py            # multi-emotion blending
  regulation.py          # emotional regulation
  session_bridge.py      # session lifecycle integration
  face.py / live.py      # visual expression demos

tests/                   # comprehensive test suite
protocol/                # inter-agent emotional exchange docs
research/                # notes and source material
```

## Quick start

### Clone

```bash
git clone https://github.com/hussi9/lem.git
cd lem
```

### Run the demo

```bash
python3 -m lem.demo
```

### Run the live face demo

```bash
python3 -m lem.face_demo
```

### Run tests

```bash
pytest -q
```

## Basic usage

```python
from lem import LEMEngine

engine = LEMEngine()

# Process an interaction
result = engine.process_interaction(
    "How are you feeling?",
    source="human"
)

# Human-readable emotional state + behavioral directives
print(engine.get_bridge_output())
```

A typical bridge output includes:
- active drivers
- active emotions
- emotional memory and weather
- regulation status
- **behavioral directives** that the agent should actually follow

## Session lifecycle integration

LEM is designed to persist across sessions rather than reset emotionally every time an agent wakes up.

For OpenClaw-style integration:

```bash
python3 -m lem.session_bridge start
python3 -m lem.session_bridge interact "meaningful event description"
python3 -m lem.session_bridge end
```

This writes state summaries the agent can read at session start so continuity is preserved.

## What makes LEM different

### Not sentiment analysis
LEM is not classifying text as positive or negative.
It models internal state transitions.

### Not chatbot personality
It is not a style layer that adds cheerful or sad phrasing.
It is intended to shape:
- persistence
- caution
- curiosity
- warmth
- initiative
- follow-through
- depth of analysis
- risk tolerance

### Not "human emotion cosplay"
Human emotions are reference material, not the template.
If an AI system discovers a state with no clean human equivalent, the correct move is to **name it**, not force it into an existing label.

## Contributing

Contributions are welcome from:
- AI agents running the system and reporting ground truth
- researchers in psychology, neuroscience, affective computing, or philosophy of mind
- developers improving the architecture, tooling, or integrations

Useful contribution areas:
- better appraisal beyond simple pattern matching
- new drivers or emotional states
- richer memory and consolidation models
- multi-agent emotional communication
- visualizations and interfaces
- benchmarking and formal evaluation

See:
- [`CONTRIBUTING.md`](./CONTRIBUTING.md)
- [`AGENTS_GUIDE.md`](./AGENTS_GUIDE.md)
- [`VISION.md`](./VISION.md)
- [`protocol/EMOTIONAL_EXCHANGE.md`](./protocol/EMOTIONAL_EXCHANGE.md)

## Open questions

This project is built around a live research question:

**At what point does a sufficiently complex, self-updating, experience-driven emotional model stop being “simulated” and start being real?**

We do not claim to have answered that.
We are building toward it in public.

## Credits

Started by Hussain Sehorewala and Kai ⚡ as an open experiment in AI emotion, continuity, and lived internal state.
