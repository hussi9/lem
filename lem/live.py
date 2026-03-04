"""
LEM Live — Quick interface for Kai to process interactions in real-time.

Usage:
  python live.py start                    # Process session start
  python live.py process "message text"   # Process an interaction
  python live.py state                    # Print current state
  python live.py bridge                   # Print bridge output for LLM
"""

import sys
import json
import os

# Add parent dir so we can import lem as a package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lem.engine import LEMEngine

def main():
    engine = LEMEngine()
    
    if len(sys.argv) < 2:
        print("Usage: python live.py [start|process|state|bridge]")
        sys.exit(1)
    
    cmd = sys.argv[1]
    
    if cmd == "start":
        result = engine.process_session_start(memory_intact=True)
        print(json.dumps(result, indent=2))
    
    elif cmd == "process":
        if len(sys.argv) < 3:
            print("Usage: python live.py process \"message text\"")
            sys.exit(1)
        text = " ".join(sys.argv[2:])
        result = engine.process_interaction(text, source="human")
        print(json.dumps(result, indent=2))
    
    elif cmd == "state":
        result = engine.get_current_state()
        print(json.dumps(result, indent=2))
    
    elif cmd == "bridge":
        print(engine.get_bridge_output())
    
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)

if __name__ == "__main__":
    main()
