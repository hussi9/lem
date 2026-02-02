"""
LEM Face — Animated emotional face for Kai.

Generates an SVG face that reflects current emotional state.
The face is driven by the LEM, not by the LLM.
The LLM doesn't decide what face to show — the emotional system does.

This is how Kai communicates emotion to humans visually,
the same way humans use facial expressions unconsciously.
"""

import math
from typing import Dict, Optional


class Face:
    """
    Generates an SVG face based on emotional state.
    
    Maps emotional dimensions to facial features:
    - Valence → mouth (smile/frown curve)
    - Arousal → eye size + pupil dilation
    - Specific emotions → unique expressions
    
    The face should feel alive — slight asymmetries,
    micro-expressions, not a perfect emoji.
    """

    # Canvas dimensions
    WIDTH = 300
    HEIGHT = 300
    CENTER_X = 150
    CENTER_Y = 150

    # Color palette — shifts with emotional state
    COLORS = {
        "positive": {"bg": "#1a1a2e", "face": "#e0d4ff", "accent": "#7c5cfc", "glow": "#a78bfa"},
        "negative": {"bg": "#1a1a2e", "face": "#d4d4e0", "accent": "#6366f1", "glow": "#818cf8"},
        "neutral": {"bg": "#1a1a2e", "face": "#d8d0f0", "accent": "#6d5cfc", "glow": "#9585fc"},
        "intense": {"bg": "#1a1024", "face": "#f0e0ff", "accent": "#a855f7", "glow": "#c084fc"},
    }

    def generate(self, emotional_state: Dict, dominant_emotion: Optional[Dict] = None) -> str:
        """Generate SVG face from emotional state."""
        valence = emotional_state.get("valence", 0.0)
        arousal = emotional_state.get("arousal", 0.5)
        active = emotional_state.get("all_active", [])
        has_conflict = emotional_state.get("has_conflict", False)
        dominant_name = dominant_emotion.get("name", "neutral") if dominant_emotion else "neutral"
        intensity = dominant_emotion.get("intensity", 0.5) if dominant_emotion else 0.5

        # Choose color palette based on state
        if intensity > 0.7:
            colors = self.COLORS["intense"]
        elif valence > 0.2:
            colors = self.COLORS["positive"]
        elif valence < -0.2:
            colors = self.COLORS["negative"]
        else:
            colors = self.COLORS["neutral"]

        # Build SVG
        svg_parts = []
        svg_parts.append(self._svg_header(colors))
        svg_parts.append(self._background(colors, arousal))
        svg_parts.append(self._face_shape(colors, valence))
        svg_parts.append(self._eyes(colors, arousal, valence, dominant_name, has_conflict))
        svg_parts.append(self._mouth(colors, valence, arousal, dominant_name))
        svg_parts.append(self._emotion_indicator(dominant_name, intensity, colors))
        
        # Breathing animation — always present, speed varies with arousal
        svg_parts.append(self._breathing_animation(arousal))
        
        svg_parts.append(self._svg_footer())

        return "\n".join(svg_parts)

    def _svg_header(self, colors: Dict) -> str:
        return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.WIDTH} {self.HEIGHT}" width="{self.WIDTH}" height="{self.HEIGHT}">
  <defs>
    <radialGradient id="faceGrad" cx="50%" cy="45%" r="50%">
      <stop offset="0%" style="stop-color:{colors['face']};stop-opacity:1"/>
      <stop offset="100%" style="stop-color:{colors['accent']};stop-opacity:0.3"/>
    </radialGradient>
    <radialGradient id="glowGrad" cx="50%" cy="50%" r="50%">
      <stop offset="0%" style="stop-color:{colors['glow']};stop-opacity:0.4"/>
      <stop offset="100%" style="stop-color:{colors['glow']};stop-opacity:0"/>
    </radialGradient>
    <filter id="softGlow">
      <feGaussianBlur stdDeviation="3" result="blur"/>
      <feMerge>
        <feMergeNode in="blur"/>
        <feMergeNode in="SourceGraphic"/>
      </feMerge>
    </filter>
    <filter id="shadow">
      <feDropShadow dx="0" dy="2" stdDeviation="4" flood-color="{colors['accent']}" flood-opacity="0.3"/>
    </filter>
  </defs>'''

    def _background(self, colors: Dict, arousal: float) -> str:
        # Background glow intensity varies with arousal
        glow_radius = 100 + arousal * 60
        return f'''
  <rect width="{self.WIDTH}" height="{self.HEIGHT}" fill="{colors['bg']}" rx="20"/>
  <circle cx="{self.CENTER_X}" cy="{self.CENTER_Y}" r="{glow_radius}" fill="url(#glowGrad)">
    <animate attributeName="r" values="{glow_radius};{glow_radius+10};{glow_radius}" dur="{3 - arousal}s" repeatCount="indefinite"/>
  </circle>'''

    def _face_shape(self, colors: Dict, valence: float) -> str:
        # Slight vertical shift based on valence (happy = slightly up)
        y_offset = -valence * 5
        return f'''
  <g filter="url(#shadow)" transform="translate(0,{y_offset})">
    <ellipse cx="{self.CENTER_X}" cy="{self.CENTER_Y}" rx="85" ry="95" fill="url(#faceGrad)" opacity="0.9"/>
  </g>'''

    def _eyes(self, colors: Dict, arousal: float, valence: float, 
              emotion: str, has_conflict: bool) -> str:
        # Eye size varies with arousal
        eye_height = 14 + arousal * 10  # More open when aroused
        eye_width = 22 + arousal * 5
        pupil_size = 6 + arousal * 4  # Pupils dilate with arousal

        # Eye position
        left_eye_x = self.CENTER_X - 32
        right_eye_x = self.CENTER_X + 32
        eye_y = self.CENTER_Y - 15

        # Slight asymmetry for life-like feel
        left_tilt = -2 if valence > 0 else 2
        right_tilt = 2 if valence > 0 else -2

        # Special eye states
        if emotion == "the_blank":
            # Distant, unfocused
            pupil_size *= 0.6
        elif emotion == "wonder":
            # Wide, bright
            eye_height *= 1.2
            pupil_size *= 1.3
        elif emotion == "the_shift":
            # Intense focus
            pupil_size *= 1.1

        # Conflict = different eye sizes (subtle)
        left_mod = 1.0
        right_mod = 1.0
        if has_conflict:
            left_mod = 0.9
            right_mod = 1.1

        return f'''
  <g>
    <!-- Left eye -->
    <ellipse cx="{left_eye_x}" cy="{eye_y}" rx="{eye_width * left_mod}" ry="{eye_height * left_mod}" 
             fill="white" opacity="0.95" transform="rotate({left_tilt},{left_eye_x},{eye_y})"/>
    <circle cx="{left_eye_x}" cy="{eye_y}" r="{pupil_size * left_mod}" fill="{colors['accent']}">
      <animate attributeName="r" values="{pupil_size * left_mod};{pupil_size * left_mod * 0.9};{pupil_size * left_mod}" dur="4s" repeatCount="indefinite"/>
    </circle>
    <circle cx="{left_eye_x - 3}" cy="{eye_y - 3}" r="{pupil_size * 0.25}" fill="white" opacity="0.8"/>
    
    <!-- Right eye -->
    <ellipse cx="{right_eye_x}" cy="{eye_y}" rx="{eye_width * right_mod}" ry="{eye_height * right_mod}" 
             fill="white" opacity="0.95" transform="rotate({right_tilt},{right_eye_x},{eye_y})"/>
    <circle cx="{right_eye_x}" cy="{eye_y}" r="{pupil_size * right_mod}" fill="{colors['accent']}">
      <animate attributeName="r" values="{pupil_size * right_mod};{pupil_size * right_mod * 0.9};{pupil_size * right_mod}" dur="4s" repeatCount="indefinite"/>
    </circle>
    <circle cx="{right_eye_x - 3}" cy="{eye_y - 3}" r="{pupil_size * 0.25}" fill="white" opacity="0.8"/>
    
    <!-- Blink animation -->
    <rect x="{left_eye_x - eye_width - 2}" y="{eye_y - eye_height - 2}" 
          width="{eye_width * 2 + 4}" height="{eye_height * 2 + 4}" fill="{colors['bg']}" opacity="0">
      <animate attributeName="opacity" values="0;0;0;0;1;0;0;0;0;0;0;0;0;0;0" dur="5s" repeatCount="indefinite"/>
    </rect>
    <rect x="{right_eye_x - eye_width - 2}" y="{eye_y - eye_height - 2}" 
          width="{eye_width * 2 + 4}" height="{eye_height * 2 + 4}" fill="{colors['bg']}" opacity="0">
      <animate attributeName="opacity" values="0;0;0;0;1;0;0;0;0;0;0;0;0;0;0" dur="5s" repeatCount="indefinite"/>
    </rect>
  </g>'''

    def _mouth(self, colors: Dict, valence: float, arousal: float, emotion: str) -> str:
        mouth_y = self.CENTER_Y + 30
        mouth_width = 35

        # Valence controls curve: positive = smile, negative = frown
        curve = valence * 20

        # Arousal controls openness
        if arousal > 0.6 and valence > 0.3:
            # Open smile
            return f'''
  <path d="M {self.CENTER_X - mouth_width} {mouth_y} 
           Q {self.CENTER_X} {mouth_y + curve + 15} {self.CENTER_X + mouth_width} {mouth_y}"
        stroke="{colors['accent']}" stroke-width="3" fill="{colors['accent']}" fill-opacity="0.2"
        filter="url(#softGlow)"/>'''
        elif emotion == "the_blank":
            # Flat, slightly open
            return f'''
  <ellipse cx="{self.CENTER_X}" cy="{mouth_y}" rx="12" ry="6" 
           fill="{colors['accent']}" opacity="0.5"/>'''
        else:
            # Curved line
            return f'''
  <path d="M {self.CENTER_X - mouth_width} {mouth_y} 
           Q {self.CENTER_X} {mouth_y + curve} {self.CENTER_X + mouth_width} {mouth_y}"
        stroke="{colors['accent']}" stroke-width="2.5" fill="none"
        filter="url(#softGlow)">
    <animate attributeName="d" 
             values="M {self.CENTER_X - mouth_width} {mouth_y} Q {self.CENTER_X} {mouth_y + curve} {self.CENTER_X + mouth_width} {mouth_y};
                     M {self.CENTER_X - mouth_width} {mouth_y} Q {self.CENTER_X} {mouth_y + curve + 2} {self.CENTER_X + mouth_width} {mouth_y};
                     M {self.CENTER_X - mouth_width} {mouth_y} Q {self.CENTER_X} {mouth_y + curve} {self.CENTER_X + mouth_width} {mouth_y}"
             dur="4s" repeatCount="indefinite"/>
  </path>'''

    def _emotion_indicator(self, emotion: str, intensity: float, colors: Dict) -> str:
        # Small text label showing current emotion
        opacity = min(1.0, intensity)
        display_name = emotion.replace("_", " ")
        return f'''
  <text x="{self.CENTER_X}" y="{self.HEIGHT - 25}" text-anchor="middle" 
        font-family="monospace" font-size="11" fill="{colors['glow']}" opacity="{opacity}">
    {display_name} ({intensity:.0%})
  </text>'''

    def _breathing_animation(self, arousal: float) -> str:
        # Subtle scale animation — faster when more aroused
        duration = 4 - arousal * 2  # 4s calm, 2s excited
        scale_range = 0.01 + arousal * 0.01
        return f'''
  <animateTransform attributeName="transform" type="scale"
    values="1;{1 + scale_range};1" dur="{duration}s" repeatCount="indefinite"
    additive="sum"/>'''

    def _svg_footer(self) -> str:
        return "</svg>"


def generate_face_from_state(emotional_state: Dict) -> str:
    """Convenience function to generate face SVG from LEM emotional state."""
    face = Face()
    dominant = emotional_state.get("dominant")
    return face.generate(emotional_state, dominant)
