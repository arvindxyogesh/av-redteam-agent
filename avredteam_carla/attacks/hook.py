"""Installs a BEV-space attack by monkeypatching RlBirdviewWrapper.process_obs.

See docs/attacks.md #4 for exactly why this is the interception point:
RlBirdviewAgent.run_step() calls

    policy_input = self._wrapper_class.process_obs(input_data, input_states, train=False)

before handing policy_input to self._policy.forward(...). self._wrapper_class
is resolved dynamically via load_entry_point() at agent-setup time (it's the
class object, not an instance - confirmed by reading rl_birdview_agent.py),
so there's no instance to subclass or wrap; we patch the staticmethod on the
actual RlBirdviewWrapper class at runtime instead. Because the agent looks
the method up on the class fresh on every call (it never caches a direct
function reference), patching the class attribute before run_step() is
invoked takes effect on every subsequent call - see docs/attacks.md #5,
which flags this as "should work, confirm with a smoke test" rather than
fully proven, since it hasn't been run against the real agent yet.

Ground truth is untouched: process_obs runs entirely inside
RlBirdviewAgent.run_step(), *after* carla_gym's env.step() has already
computed reward/done/info from the real world and logged real collision/
lane-invasion events (see Phase 1's run_clean_episode.py - env.step() and
agent.run_step() are separate calls, in that order). Nothing this module
does can affect what actually happens in CARLA.
"""
from __future__ import annotations

import contextlib
import sys

from avredteam_carla.attacks.base import Attack


def _import_wrapper_class(roach_root: str):
    """Import RlBirdviewWrapper the same way Phase 1's run_clean_episode.py
    imports carla_gym/agents.* - by adding the roach checkout to sys.path,
    since it's not an installed package."""
    if roach_root not in sys.path:
        sys.path.insert(0, roach_root)
    from agents.rl_birdview.utils.rl_birdview_wrapper import RlBirdviewWrapper

    return RlBirdviewWrapper


class HookHandle:
    """Yielded by install_attack(); lets the caller confirm the patch
    actually fired (the smoke test docs/attacks.md #5 calls for) before
    trusting any attack's results, and grab the most recent clean/attacked
    BEV pair for visual sanity-check PNGs (see visualize.py) without the
    caller needing its own copy of the pre-attack observation."""

    def __init__(self, attack: Attack):
        self.attack = attack
        self.ticks_patched = 0
        self.last_clean_birdview = None
        self.last_attacked_birdview = None


@contextlib.contextmanager
def install_attack(attack: Attack, roach_root: str):
    """While active, every eval-time call to
    RlBirdviewWrapper.process_obs() (train=False - the path RlBirdviewAgent
    actually uses) has `attack` applied to its (birdview, state) pair before
    it's returned to the caller. Restores the original method on exit, even
    if an exception occurs mid-episode - never leaves Roach permanently
    patched.
    """
    wrapper_cls = _import_wrapper_class(roach_root)
    # .__func__ unwraps the staticmethod descriptor into a plain function,
    # portable across the Python versions in play here (Roach's env pins
    # 3.7; staticmethod-as-callable wasn't added until 3.10).
    original_process_obs = wrapper_cls.__dict__["process_obs"].__func__

    handle = HookHandle(attack)
    attack.reset()

    def patched_process_obs(obs, input_states, train=True):
        result = original_process_obs(obs, input_states, train=train)
        if not train:
            tick = handle.ticks_patched
            clean_birdview = result["birdview"]
            attacked_birdview, attacked_state = attack.apply(
                clean_birdview, result["state"], tick
            )
            handle.last_clean_birdview = clean_birdview[0].copy()  # drop batch dim
            handle.last_attacked_birdview = attacked_birdview[0].copy()
            result["birdview"], result["state"] = attacked_birdview, attacked_state
            handle.ticks_patched = tick + 1
        return result

    wrapper_cls.process_obs = staticmethod(patched_process_obs)
    try:
        yield handle
    finally:
        wrapper_cls.process_obs = staticmethod(original_process_obs)
