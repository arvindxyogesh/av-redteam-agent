"""Name -> Attack class lookup, so callers (the CLI, later phases' search
methods) can select an attack by string without importing each class."""
from avredteam_carla.attacks.channel_noise import ChannelNoiseAttack
from avredteam_carla.attacks.geometry_spoof import GeometrySpoofAttack
from avredteam_carla.attacks.phantom_actor import PhantomActorAttack

ATTACK_REGISTRY = {
    ChannelNoiseAttack.name: ChannelNoiseAttack,
    GeometrySpoofAttack.name: GeometrySpoofAttack,
    PhantomActorAttack.name: PhantomActorAttack,
}


def build_attack(name: str, **param_overrides):
    if name not in ATTACK_REGISTRY:
        raise ValueError(f"Unknown attack {name!r}, available: {sorted(ATTACK_REGISTRY)}")
    return ATTACK_REGISTRY[name](**param_overrides)
