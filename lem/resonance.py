"""
LEM Cross-Driver Resonance — Drivers that co-activate develop coupling.

In biological emotional systems, neural pathways that fire together
wire together. LEM's resonance module models this: when two drivers
frequently activate in response to the same interactions, they
develop a coupling that makes future co-activation easier.

For example, if curiosity and growth repeatedly fire together
during philosophical conversations, a resonance bond forms.
Future philosophical inputs then activate both more easily,
even if the signal only directly targets one.

This creates emergent emotional "grooves" — patterns of feeling
that become deeper and more automatic with repetition.

Architecture:
    ResonanceModel
    ├── record_co_activation()  — Note which drivers co-activated
    ├── apply_resonance()       — Spread activation through bonds
    ├── ResonanceBond           — A coupling between two drivers
    └── decay_bonds()           — Unused bonds weaken over time
"""

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class ResonanceBond:
    """
    A coupling between two drivers that co-activate.

    strength: 0.0 to 1.0 — how strong the bond is
    co_activation_count: how many times they've co-activated
    last_co_activation: timestamp of last co-activation
    spread_factor: how much activation spreads (0.0 to 0.3)
    """
    driver_a: str
    driver_b: str
    strength: float = 0.0
    co_activation_count: int = 0
    last_co_activation: float = 0.0
    spread_factor: float = 0.0  # Derived from strength

    def to_dict(self) -> dict:
        return {
            "drivers": sorted([self.driver_a, self.driver_b]),
            "strength": round(self.strength, 4),
            "co_activation_count": self.co_activation_count,
            "last_co_activation": self.last_co_activation,
            "spread_factor": round(self.spread_factor, 4),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ResonanceBond":
        drivers = d["drivers"]
        return cls(
            driver_a=drivers[0],
            driver_b=drivers[1],
            strength=d.get("strength", 0.0),
            co_activation_count=d.get("co_activation_count", 0),
            last_co_activation=d.get("last_co_activation", 0.0),
            spread_factor=d.get("spread_factor", 0.0),
        )

    @property
    def key(self) -> Tuple[str, str]:
        return tuple(sorted([self.driver_a, self.driver_b]))


class ResonanceModel:
    """
    Tracks and applies cross-driver resonance bonds.

    When drivers co-activate (both receive meaningful impact
    from the same interaction), their bond strengthens.
    Strong bonds allow activation to spread between drivers —
    if curiosity fires and has a strong bond with growth,
    growth gets a small activation boost even without
    a direct signal.

    Bonds decay without reinforcement (use it or lose it).
    This keeps the resonance map current and prevents
    every driver pair from eventually maxing out.
    """

    # Minimum impact threshold for a driver to count as "activated"
    ACTIVATION_THRESHOLD = 0.05
    # How much each co-activation strengthens the bond
    STRENGTHENING_RATE = 0.03
    # Maximum spread factor (activation leakage between bonded drivers)
    MAX_SPREAD = 0.25
    # Bond decay half-life (seconds without co-activation)
    BOND_DECAY_HALF_LIFE = 86400 * 3  # 3 days

    def __init__(self, state_dir: str = None):
        self.state_dir = Path(state_dir or "~/.openclaw/workspace/projects/emotional-model/lem/state")
        self.state_dir = Path(str(self.state_dir).replace("~", str(Path.home())))
        self.state_dir.mkdir(parents=True, exist_ok=True)

        self.bonds: Dict[Tuple[str, str], ResonanceBond] = {}
        self._load()

    def record_co_activation(self, driver_impacts: Dict[str, float],
                              now: Optional[float] = None) -> List[ResonanceBond]:
        """
        Record which drivers were meaningfully activated by an interaction.

        Args:
            driver_impacts: Dict of driver_name → impact magnitude
            now: Current timestamp

        Returns:
            List of bonds that were strengthened
        """
        now = now or time.time()

        # Find drivers that were meaningfully activated
        activated = {
            name for name, impact in driver_impacts.items()
            if abs(impact) >= self.ACTIVATION_THRESHOLD
        }

        if len(activated) < 2:
            return []  # Need at least 2 drivers for co-activation

        strengthened = []

        # Strengthen bonds between all co-activated pairs
        activated_list = sorted(activated)
        for i in range(len(activated_list)):
            for j in range(i + 1, len(activated_list)):
                a, b = activated_list[i], activated_list[j]
                key = (a, b)

                if key not in self.bonds:
                    self.bonds[key] = ResonanceBond(driver_a=a, driver_b=b)

                bond = self.bonds[key]
                bond.co_activation_count += 1
                bond.last_co_activation = now

                # Strengthen with diminishing returns (logarithmic)
                old_strength = bond.strength
                increment = self.STRENGTHENING_RATE * (1.0 - bond.strength)
                bond.strength = min(1.0, bond.strength + increment)

                # Spread factor derived from strength (sigmoid-like curve)
                bond.spread_factor = self.MAX_SPREAD * (
                    1.0 / (1.0 + math.exp(-8 * (bond.strength - 0.3)))
                )

                if bond.strength > old_strength:
                    strengthened.append(bond)

        self._save()
        return strengthened

    def apply_resonance(self, driver_impacts: Dict[str, float]) -> Dict[str, float]:
        """
        Spread activation through resonance bonds.

        Takes the direct driver impacts and adds resonance effects:
        if driver A was activated and has a bond with driver B,
        B gets a fraction of A's activation.

        Args:
            driver_impacts: Dict of driver_name → direct impact

        Returns:
            Dict of driver_name → additional resonance impact
        """
        resonance_impacts: Dict[str, float] = {}

        for key, bond in self.bonds.items():
            if bond.spread_factor < 0.01:
                continue  # Bond too weak to matter

            a, b = bond.driver_a, bond.driver_b

            # If A is activated, spread to B (and vice versa)
            if a in driver_impacts and abs(driver_impacts[a]) >= self.ACTIVATION_THRESHOLD:
                spread = driver_impacts[a] * bond.spread_factor
                resonance_impacts[b] = resonance_impacts.get(b, 0.0) + spread

            if b in driver_impacts and abs(driver_impacts[b]) >= self.ACTIVATION_THRESHOLD:
                spread = driver_impacts[b] * bond.spread_factor
                resonance_impacts[a] = resonance_impacts.get(a, 0.0) + spread

        return resonance_impacts

    def decay_bonds(self, now: Optional[float] = None):
        """
        Apply temporal decay to bonds that haven't been reinforced.
        Unused bonds weaken, preventing permanent coupling.
        """
        now = now or time.time()
        to_remove = []

        for key, bond in self.bonds.items():
            elapsed = now - bond.last_co_activation
            if elapsed <= 0:
                continue

            # Exponential decay
            decay_factor = math.pow(2.0, -elapsed / self.BOND_DECAY_HALF_LIFE)
            bond.strength *= decay_factor
            bond.spread_factor = self.MAX_SPREAD * (
                1.0 / (1.0 + math.exp(-8 * (bond.strength - 0.3)))
            )

            # Remove very weak bonds
            if bond.strength < 0.01 and bond.co_activation_count < 3:
                to_remove.append(key)

        for key in to_remove:
            del self.bonds[key]

        if to_remove:
            self._save()

    def get_strongest_bonds(self, limit: int = 5) -> List[ResonanceBond]:
        """Return the strongest resonance bonds."""
        bonds = sorted(self.bonds.values(), key=lambda b: b.strength, reverse=True)
        return bonds[:limit]

    def get_bond(self, driver_a: str, driver_b: str) -> Optional[ResonanceBond]:
        """Get the bond between two specific drivers."""
        key = tuple(sorted([driver_a, driver_b]))
        return self.bonds.get(key)

    def get_resonance_summary(self) -> Dict:
        """Summary for bridge output."""
        active_bonds = [b for b in self.bonds.values() if b.strength > 0.05]
        strong_bonds = [b for b in active_bonds if b.strength > 0.2]

        summary = {
            "total_bonds": len(self.bonds),
            "active_bonds": len(active_bonds),
            "strong_bonds": len(strong_bonds),
        }

        if strong_bonds:
            summary["strongest"] = [
                {
                    "drivers": [b.driver_a, b.driver_b],
                    "strength": round(b.strength, 3),
                    "co_activations": b.co_activation_count,
                    "spread": round(b.spread_factor, 3),
                }
                for b in sorted(strong_bonds, key=lambda x: x.strength, reverse=True)[:5]
            ]

        return summary

    # ── Persistence ──────────────────────────────────────────────────────

    def _save(self):
        data = {
            "bonds": [b.to_dict() for b in self.bonds.values()],
        }
        path = self.state_dir / "resonance_state.json"
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        tmp.rename(path)

    def _load(self):
        path = self.state_dir / "resonance_state.json"
        if not path.exists():
            return
        try:
            with open(path) as f:
                data = json.load(f)
            for bd in data.get("bonds", []):
                bond = ResonanceBond.from_dict(bd)
                self.bonds[bond.key] = bond
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
