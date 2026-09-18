#!/usr/bin/env python3
"""Generate a watertight, same-size can with axial surface ridges.

``Can_d4cm.usd`` is an 8 cm long horizontal can: its axis is world X, its
origin sits at the positive-X end, and its circular cross-section is in Y/Z.
This generator preserves that convention exactly, so the task's existing
poses and side-grasp protocol remain valid. It encodes roughness only in
contact geometry; density, friction, and constitution are assigned by task.
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


def _format_uvs(values: list[tuple[float, float]], width: int = 6) -> str:
    rows = []
    for index in range(0, len(values), width):
        rows.append(", ".join(f"({u:.8f}, {v:.8f})" for u, v in values[index : index + width]))
    return ",\n            ".join(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sides", type=int, default=96)
    parser.add_argument("--radius", type=float, default=0.020)
    parser.add_argument("--length", type=float, default=0.080)
    parser.add_argument("--ridge-amplitude", type=float, default=0.00035)
    parser.add_argument("--ridge-count", type=int, default=16)
    args = parser.parse_args()
    if args.sides < 12 or args.radius <= 0.0 or args.length <= 0.0:
        raise ValueError("sides >= 12 and positive radius/length are required")
    if args.ridge_count < 1 or args.ridge_amplitude < 0.0:
        raise ValueError("ridge count must be positive and amplitude non-negative")

    points: list[tuple[float, float, float]] = []
    # Match the source asset's local frame: X in [-length, 0], circular Y/Z.
    for x in (-args.length, 0.0):
        for index in range(args.sides):
            theta = 2.0 * math.pi * index / args.sides
            radius = args.radius + args.ridge_amplitude * math.sin(args.ridge_count * theta)
            points.append((x, radius * math.cos(theta), radius * math.sin(theta)))
    start_center = len(points)
    points.append((-args.length, 0.0, 0.0))
    end_center = len(points)
    points.append((0.0, 0.0, 0.0))

    triangles: list[tuple[int, int, int]] = []
    uvs: list[tuple[float, float]] = []
    for index in range(args.sides):
        next_index = (index + 1) % args.sides
        end = args.sides + index
        end_next = args.sides + next_index
        u0 = index / args.sides
        u1 = (index + 1) / args.sides
        triangles.extend(
            (
                (index, next_index, end_next),
                (index, end_next, end),
                (start_center, next_index, index),
                (end_center, end, end_next),
            )
        )
        uvs.extend(
            (
                (u0, 0.0), (u1, 0.0), (u1, 1.0),
                (u0, 0.0), (u1, 1.0), (u0, 1.0),
                (0.5, 0.5),
                (0.5 + points[next_index][1] / (2.0 * args.radius), 0.5 + points[next_index][2] / (2.0 * args.radius)),
                (0.5 + points[index][1] / (2.0 * args.radius), 0.5 + points[index][2] / (2.0 * args.radius)),
                (0.5, 0.5),
                (0.5 + points[end][1] / (2.0 * args.radius), 0.5 + points[end][2] / (2.0 * args.radius)),
                (0.5 + points[end_next][1] / (2.0 * args.radius), 0.5 + points[end_next][2] / (2.0 * args.radius)),
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
        texCoord2f[] primvars:st = [
            {_format_uvs(uvs)}
        ] (
            interpolation = "faceVarying"
        )
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
