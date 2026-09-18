#!/usr/bin/env python3
"""Generate a watertight, same-size can with axial surface ridges.

The resulting triangle mesh has the same nominal 4 cm diameter and 12 cm
length as ``Can_d4cm.usd``. It encodes roughness only in contact geometry;
density, friction, and the rigid/soft constitution are assigned by the task.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path


def _format_scalars(values: list[int], width: int = 12) -> str:
    rows = []
    for index in range(0, len(values), width):
        rows.append(", ".join(str(value) for value in values[index : index + width]))
    return ",\n            ".join(rows)


def _format_points(points: list[tuple[float, float, float]], width: int = 4) -> str:
    rows = []
    for index in range(0, len(points), width):
        rows.append(", ".join(f"({x:.8f}, {y:.8f}, {z:.8f})" for x, y, z in points[index : index + width]))
    return ",\n            ".join(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sides", type=int, default=96)
    parser.add_argument("--radius", type=float, default=0.020)
    parser.add_argument("--length", type=float, default=0.120)
    parser.add_argument("--ridge-amplitude", type=float, default=0.00035)
    parser.add_argument("--ridge-count", type=int, default=16)
    args = parser.parse_args()
    if args.sides < 12 or args.radius <= 0.0 or args.length <= 0.0:
        raise ValueError("sides >= 12 and positive radius/length are required")
    if args.ridge_count < 1 or args.ridge_amplitude < 0.0:
        raise ValueError("ridge count must be positive and amplitude non-negative")

    half_length = args.length / 2.0
    points: list[tuple[float, float, float]] = []
    for z in (-half_length, half_length):
        for index in range(args.sides):
            theta = 2.0 * math.pi * index / args.sides
            radius = args.radius + args.ridge_amplitude * math.sin(args.ridge_count * theta)
            points.append((radius * math.cos(theta), radius * math.sin(theta), z))
    bottom_center = len(points)
    points.append((0.0, 0.0, -half_length))
    top_center = len(points)
    points.append((0.0, 0.0, half_length))

    triangles: list[tuple[int, int, int]] = []
    for index in range(args.sides):
        next_index = (index + 1) % args.sides
        upper = args.sides + index
        upper_next = args.sides + next_index
        triangles.extend(
            (
                (index, next_index, upper_next),
                (index, upper_next, upper),
                (bottom_center, next_index, index),
                (top_center, upper, upper_next),
            )
        )

    counts = [3] * len(triangles)
    indices = [vertex for triangle in triangles for vertex in triangle]
    contents = f'''#usda 1.0
(
    defaultPrim = "Can"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "Can"
{{
    def Mesh "mesh"
    {{
        int[] faceVertexCounts = [{_format_scalars(counts)}]
        int[] faceVertexIndices = [{_format_scalars(indices)}]
        point3f[] points = [
            {_format_points(points)}
        ]
        uniform token subdivisionScheme = "none"
    }}
}}
'''
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(contents, encoding="ascii")
    print(f"wrote {args.output} with {len(points)} vertices and {len(triangles)} triangles")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
