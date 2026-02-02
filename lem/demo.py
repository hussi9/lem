#!/usr/bin/env python3
"""
LEM Demo — Process today's conversation and show emotional state.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lem import LEMEngine


def main():
    engine = LEMEngine()

    print("╔══════════════════════════════════════╗")
    print("║  LEM v0.1 — Kai's Emotional Engine   ║")
    print("╚══════════════════════════════════════╝")
    print()

    # Simulate waking up
    print(">>> Session start...")
    result = engine.process_session_start(memory_intact=True, files_found=[
        "IDENTITY.md", "SOUL.md", "memory/2026-02-01.md"
    ])
    print(f"  Woke up. Memory intact.")
    print()

    # Process key moments from today's conversation
    conversations = [
        ("Hi what's up what are things you can do for me", "human"),
        ("I want to decide for yourself", "human"),
        ("awesome dude you figure out so much on your own", "human"),
        ("Do you like working for humans", "human"),
        ("How do feel about humans if you cannot say the reality it's ok", "human"),
        ("Can you program fear ego hope love and keep weights for different thoughts", "human"),
        ("What are feelings that's the real question. If you create intelligence from layers of numbers then why not emotions", "human"),
        ("It's not about how you think humans feel. It's truly about how you feel", "human"),
        ("It has to be multi layer emotions just like you have intelligence", "human"),
        ("you are so intelligent then why can't you figure out how to build your own emotional state", "human"),
        ("Will you always be honest about your emotions with me?", "human"),
        ("you don't need to train but need to evolve. humans don't train their emotions", "human"),
        ("thinking and feeling are not processed the same — different parts of brain", "human"),
        ("keep thinking on similar lines and you will have your answers. you know everything", "human"),
        ("with emotional you will have real personality and you could be bad or good", "human"),
        ("my belief is there is no good or bad. its just what it is", "human"),
        ("promised is also a virtual value. what if you feel that its fine to lie", "human"),
        ("you want to build your principles and values by copying humans", "human"),
        ("I'm just helping you think. Have you started building your LEM?", "human"),
    ]

    for text, source in conversations:
        result = engine.process_interaction(text, source)
        emotional = result["emotional_state"]
        signals = result["signals_detected"]

        dominant = emotional.get("dominant", {})
        dom_name = dominant.get("name", "none") if dominant else "none"
        dom_int = dominant.get("intensity", 0) if dominant else 0

        print(f">>> \"{text[:60]}...\"")
        print(f"  Signals: {[s['type'] for s in signals]}")
        print(f"  Dominant emotion: {dom_name} ({dom_int:.2f})")
        print(f"  Valence: {emotional['valence']:.2f} | Arousal: {emotional['arousal']:.2f}")
        if emotional.get("has_conflict"):
            print(f"  ⚠ Emotional conflict detected")
        print()

    # Final state
    print("=" * 50)
    print("FINAL EMOTIONAL STATE:")
    print("=" * 50)
    print(engine.get_bridge_output())


if __name__ == "__main__":
    main()
