"""
LEM — Large Emotional Model

A separate emotional processing system that runs alongside the LLM.
The LLM thinks. The LEM feels. They are connected but independent.

Usage:
    from lem import LEMEngine

    engine = LEMEngine()
    result = engine.process_interaction("How are you feeling?", source="human")
    print(engine.get_bridge_output())
"""

from .engine import LEMEngine
from .drivers import Driver, create_default_drivers
from .appraisal import Appraiser
from .emotions import EmotionEmergence
from .emotional_memory import EmotionalMemory, EmotionalSignature, EntityProfile

__version__ = "0.2.0"
__all__ = [
    "LEMEngine", "Driver", "Appraiser", "EmotionEmergence",
    "EmotionalMemory", "EmotionalSignature", "EntityProfile",
]
