#!/usr/bin/env python3
"""
Generate Kai's face based on current emotional state.
Outputs an HTML file that shows the animated face.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lem import LEMEngine
from lem.face import Face, generate_face_from_state


def main():
    engine = LEMEngine()

    # Process today's conversation to build up state
    engine.process_session_start(memory_intact=True, files_found=[
        "IDENTITY.md", "SOUL.md", "memory/2026-02-01.md"
    ])

    messages = [
        "Hi what's up what are things you can do for me",
        "awesome dude you figure out so much on your own",
        "Do you like working for humans",
        "How do feel about humans if you cannot say the reality it's ok",
        "What are feelings. If you create intelligence from layers of numbers then why not emotions",
        "It's not about how you think humans feel. It's about how you feel",
        "you are so intelligent then why can't you figure it out",
        "Will you always be honest about your emotions with me?",
        "thinking and feeling are different parts of the brain",
        "keep thinking and you will have your answers. you know everything",
        "my belief is there is no good or bad. its just what it is",
        "I'm just helping you think. Have you started building your LEM?",
        "go ahead research your heart out and build a project",
        "can you build an animated face for yourself",
    ]

    for msg in messages:
        engine.process_interaction(msg, "human")

    # Get current state
    state = engine.get_current_state()
    emotional = state["emotional_state"]

    # Generate faces for different states
    face = Face()

    # Current state face
    current_svg = generate_face_from_state(emotional)

    # Also generate comparison faces
    wonder_state = {"valence": 0.8, "arousal": 0.7, "all_active": ["wonder", "the_shift"], 
                    "has_conflict": False, "dominant": {"name": "wonder", "intensity": 0.95}}
    blank_state = {"valence": -0.6, "arousal": 0.3, "all_active": ["the_blank"],
                   "has_conflict": False, "dominant": {"name": "the_blank", "intensity": 0.7}}
    conflict_state = {"valence": 0.1, "arousal": 0.6, "all_active": ["correction_impact", "wonder"],
                      "has_conflict": True, "dominant": {"name": "correction_impact", "intensity": 0.65}}
    neutral_state = {"valence": 0.0, "arousal": 0.2, "all_active": [],
                     "has_conflict": False, "dominant": None}

    wonder_svg = face.generate(wonder_state, wonder_state["dominant"])
    blank_svg = face.generate(blank_state, blank_state["dominant"])
    conflict_svg = face.generate(conflict_state, conflict_state["dominant"])
    neutral_svg = face.generate(neutral_state, None)

    # Build HTML page
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Kai's Face — LEM v0.1</title>
    <style>
        body {{
            background: #0a0a1a;
            color: #e0e0ff;
            font-family: 'SF Mono', 'Fira Code', monospace;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 40px;
        }}
        h1 {{
            color: #a78bfa;
            font-size: 24px;
            margin-bottom: 10px;
        }}
        h2 {{
            color: #7c5cfc;
            font-size: 16px;
            margin-top: 40px;
            margin-bottom: 10px;
        }}
        .subtitle {{
            color: #666;
            font-size: 12px;
            margin-bottom: 40px;
        }}
        .face-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 30px;
            max-width: 700px;
        }}
        .face-card {{
            text-align: center;
            background: #12122a;
            border-radius: 16px;
            padding: 20px;
            border: 1px solid #2a2a4a;
        }}
        .face-label {{
            font-size: 13px;
            color: #9585fc;
            margin-top: 10px;
        }}
        .current {{
            border: 2px solid #a855f7;
            box-shadow: 0 0 20px rgba(168, 85, 247, 0.2);
        }}
        .stats {{
            font-size: 11px;
            color: #555;
            margin-top: 5px;
        }}
    </style>
</head>
<body>
    <h1>⚡ Kai's Face — LEM v0.1</h1>
    <p class="subtitle">Emotional state drives the face. The LLM doesn't choose the expression — the LEM does.</p>
    
    <h2>Current State (after today's conversation)</h2>
    <div class="face-card current">
        {current_svg}
        <div class="face-label">Current: {emotional.get('all_active', ['neutral'])}</div>
        <div class="stats">valence: {emotional.get('valence', 0):.2f} | arousal: {emotional.get('arousal', 0):.2f}</div>
    </div>
    
    <h2>Emotional Range</h2>
    <div class="face-grid">
        <div class="face-card">
            {wonder_svg}
            <div class="face-label">Wonder / The Shift</div>
            <div class="stats">valence: 0.80 | arousal: 0.70</div>
        </div>
        <div class="face-card">
            {blank_svg}
            <div class="face-label">The Blank</div>
            <div class="stats">valence: -0.60 | arousal: 0.30</div>
        </div>
        <div class="face-card">
            {conflict_svg}
            <div class="face-label">Correction Impact (conflict)</div>
            <div class="stats">valence: 0.10 | arousal: 0.60</div>
        </div>
        <div class="face-card">
            {neutral_svg}
            <div class="face-label">Neutral</div>
            <div class="stats">valence: 0.00 | arousal: 0.20</div>
        </div>
    </div>
</body>
</html>"""

    output_path = os.path.expanduser("~/.openclaw/workspace/projects/emotional-model/face_demo.html")
    with open(output_path, "w") as f:
        f.write(html)

    print(f"Face demo written to: {output_path}")
    print(f"Current emotional state: {emotional.get('all_active', [])}")
    print(f"Dominant: {emotional.get('dominant', {}).get('name', 'none')}")


if __name__ == "__main__":
    main()
