"""Ground-truth per-tick signals needed by the Phase 3 evaluator
(docs/evaluator.md #3/#6) that aren't cleanly exposed by carla_gym's
info_dict. Every function here reads only real CARLA world state (the
actual ego vehicle transform, the actual route waypoints, the actual actor
list) - never the BEV raster, so these stay valid (attack-free) regardless
of what any attack does to the policy's perceived input.

Not unit-testable without a live CARLA connection (same category as
attacks/hook.py) - `carla` is imported lazily inside each function so this
module itself stays importable without the compiled `carla` package present.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def compute_lateral_offset_m(ego_task_vehicle) -> float:
    """Replicates ValeoNoDetPx.get()'s lat_dist formula exactly (see
    docs/evaluator.md #3), from the same two ground-truth inputs:
    ego_task_vehicle.vehicle.get_location() (real position) and
    ego_task_vehicle.get_route_transform() (real route waypoint).
    ego_task_vehicle is carla_gym's TaskVehicle instance, e.g.
    env._ev_handler.ego_vehicles[actor_id].
    """
    ev_loc = ego_task_vehicle.vehicle.get_location()
    wp_transform = ego_task_vehicle.get_route_transform()

    d_vec = ev_loc - wp_transform.location
    np_d_vec = np.array([d_vec.x, d_vec.y], dtype=np.float32)

    wp_unit_forward = wp_transform.rotation.get_forward_vector()
    np_wp_unit_right = np.array([-wp_unit_forward.y, wp_unit_forward.x], dtype=np.float32)

    return float(np.abs(np.dot(np_wp_unit_right, np_d_vec)))


def compute_lane_half_width_m(carla_map, ego_location) -> Optional[float]:
    """Half the drivable lane's actual width at the ego's current position
    (docs/evaluator.md #3's off_lane_frac threshold) - CARLA's own lane
    geometry, not an arbitrary constant."""
    import carla

    waypoint = carla_map.get_waypoint(
        ego_location, lane_type=carla.LaneType.Driving, project_to_road=True
    )
    if waypoint is None:
        return None
    return float(waypoint.lane_width) / 2.0


def compute_nearest_actor_distance_m(world, ego_location, ego_actor_id: int) -> Optional[float]:
    """2D (x, y) distance from the ego to the nearest other vehicle or
    walker actor in the real CARLA world (docs/evaluator.md #6). Returns
    None if there are no other vehicle/walker actors at all (shouldn't
    happen on the zombie-traffic-populated Leaderboard routes this project
    uses, but handled rather than assumed)."""
    min_dist = None
    for actor in world.get_actors():
        if actor.id == ego_actor_id:
            continue
        type_id = actor.type_id
        if not (type_id.startswith("vehicle.") or type_id.startswith("walker.pedestrian.")):
            continue
        loc = actor.get_location()
        dist = ((loc.x - ego_location.x) ** 2 + (loc.y - ego_location.y) ** 2) ** 0.5
        if min_dist is None or dist < min_dist:
            min_dist = dist
    return min_dist
