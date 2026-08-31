"""Base interface for BEV-space sensor attacks against Roach.

Every attack in this library perturbs exactly the (birdview, state) pair
Roach's policy actually receives - see docs/attacks.md for what that pair
is and where it's intercepted. Ground-truth CARLA state (real actor
positions, the real collision/lane-invasion sensors) is never touched by an
Attack; the interception (avredteam_carla/attacks/hook.py) only ever
transforms the planner's perceived input.

The declarative `tunable_params` schema exists so later phases (random
search, Bayesian optimization, an LLM agent - Phase 4) can discover what
knobs an attack exposes without special-casing each attack type.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass(frozen=True)
class TunableParam:
    name: str
    type: str  # "float" | "int" | "bool"
    default: Any
    low: Optional[float] = None
    high: Optional[float] = None

    def __post_init__(self):
        if self.type not in ("float", "int", "bool"):
            raise ValueError(f"TunableParam {self.name!r}: unknown type {self.type!r}")
        if self.type in ("float", "int") and (self.low is None or self.high is None):
            raise ValueError(f"TunableParam {self.name!r}: float/int params need low and high")

    def cast(self, value) -> Any:
        if self.type == "float":
            value = float(value)
            return min(self.high, max(self.low, value))
        if self.type == "int":
            value = int(round(value))
            return min(int(self.high), max(int(self.low), value))
        return bool(value)


class Attack:
    """Base class for a single BEV-space attack.

    Subclass and set `name` + `tunable_params`, then implement `apply()`.
    """

    name: str = "base"
    tunable_params: tuple = ()

    def __init__(self, **param_overrides):
        declared = {p.name for p in self.tunable_params}
        unknown = set(param_overrides) - declared
        if unknown:
            raise ValueError(
                f"{self.name}: unknown params {sorted(unknown)}, declared: {sorted(declared)}"
            )
        self.params = {
            p.name: p.cast(param_overrides.get(p.name, p.default))
            for p in self.tunable_params
        }

    def reset(self) -> None:
        """Called once at the start of each episode. Override for stateful
        attacks (e.g. a ramp that needs to remember its start tick)."""

    def apply(
        self, bev_raster: np.ndarray, scalar_state: np.ndarray, tick: int
    ) -> tuple:
        """Return a perturbed (bev_raster, scalar_state) pair for this tick.

        bev_raster: uint8, shape (C, H, W) or (1, C, H, W) - see
            docs/attacks.md #2 for channel layout.
        scalar_state: float32, shape (N,) or (1, N) - see docs/attacks.md #3.
        tick: 0-indexed step counter since this episode's first tick (not
            the CARLA frame number).

        Must not mutate the inputs in place - copy before modifying, since
        the caller may still need the clean arrays for baseline logging.
        Must return arrays with the same shape and dtype as the inputs.
        """
        raise NotImplementedError

    def param_summary(self) -> dict:
        return dict(self.params)

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v!r}" for k, v in self.params.items())
        return f"{self.__class__.__name__}({params})"
