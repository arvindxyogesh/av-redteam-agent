#!/usr/bin/env python3
"""Minimal CARLA client connectivity check.

Connects to a running CARLA server, fetches the world, and lists available
maps and blueprints. Run this after scripts/launch_carla.sh to confirm the
server is actually reachable before doing anything else.

Usage:
    python scripts/check_client_connection.py --host localhost --port 2000
"""
import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args()

    try:
        import carla
    except ImportError:
        print(
            "error: 'carla' package not importable. Install the CARLA Python "
            "API into the active env first (see docs/setup.md, step 4).",
            file=sys.stderr,
        )
        return 1

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)

    print(f"Connecting to CARLA at {args.host}:{args.port} ...")
    world = client.get_world()
    print(f"Connected. Client version: {client.get_client_version()}")
    print(f"Server version: {client.get_server_version()}")

    print("\nAvailable maps:")
    for m in sorted(client.get_available_maps()):
        print(f"  {m}")

    current_map = world.get_map()
    print(f"\nCurrently loaded map: {current_map.name}")

    blueprints = world.get_blueprint_library()
    print(f"\nBlueprint library: {len(blueprints)} blueprints")
    vehicle_bps = blueprints.filter("vehicle.*")
    print(f"  vehicle.* blueprints: {len(vehicle_bps)}")
    sensor_bps = blueprints.filter("sensor.*")
    print(f"  sensor.* blueprints: {len(sensor_bps)}")

    print("\nOK: client connection verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
