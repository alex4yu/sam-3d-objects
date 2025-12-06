#!/usr/bin/env python3
"""
Convert PLY meshes to OBJ and generate a simple MuJoCo XML (MJCF) to reference them.

Usage:
    python scripts/ply_to_mujoco.py path/to/splat.ply
    python scripts/ply_to_mujoco.py --dir path/to/plys --scale 0.01

Outputs for each input `name.ply`:
 - `name.obj` (triangulated, normals computed)
 - `name.xml` (MuJoCo XML referencing the OBJ)

Requirements:
    pip install trimesh numpy

Optional (for viewing with MuJoCo):
    pip install mujoco mujoco-viewer
    export MUJOCO_GL=egl   # or osmesa / glfw depending on environment

The generated XML is minimal and places the mesh at the origin with a simple plane floor
and a camera. Use MuJoCo viewer to open the XML, for example with mujoco or mujoco-viewer.
"""

import argparse
import os
import sys
from pathlib import Path

try:
    import trimesh
    import numpy as np
except Exception as e:
    print("Missing dependency. Run: pip install trimesh numpy")
    raise


MJCF_TEMPLATE = r'''<?xml version="1.0" ?>
<mujoco model="{name}">
  <asset>
    <mesh name="{mesh_name}" file="{mesh_file}" scale="{scale_x} {scale_y} {scale_z}" />
  </asset>

  <worldbody>
    <geom type="mesh" mesh="{mesh_name}" pos="0 0 0" quat="1 0 0 0" rgba="1 1 1 1" />
    <geom type="plane" pos="0 0 -0.02" size="5 5 0.01" rgba="0.8 0.8 0.8 1"/>
    <camera name="camera" pos="{cam_pos}" euler="{cam_euler}"/>
  </worldbody>
</mujoco>
'''


def ensure_trimesh(mesh_or_scene):
    # Accept Trimesh or Scene, return single Trimesh
    if isinstance(mesh_or_scene, trimesh.Trimesh):
        mesh = mesh_or_scene
    elif isinstance(mesh_or_scene, trimesh.Scene):
        # combine geometry into a single mesh
        geom_list = []
        for geom in mesh_or_scene.geometry.values():
            geom_list.append(geom)
        if len(geom_list) == 0:
            raise ValueError("Loaded scene has no geometry")
        mesh = trimesh.util.concatenate(geom_list)
    else:
        raise TypeError("Unsupported mesh type: {}".format(type(mesh_or_scene)))
    return mesh


def process_file(path: Path, out_dir: Path, center: bool, scale_to: float, scale_tuple=None):
    print(f"Processing {path}")
    mesh = trimesh.load(path, force='mesh')
    mesh = ensure_trimesh(mesh)

    # Remove degenerate faces
    try:
        mesh.remove_degenerate_faces()
    except Exception:
        pass

    # Ensure triangle faces (trimesh should handle this on export)
    # Some PLY files may be point clouds and have no faces. Handle that gracefully.
    if mesh.faces is None or len(mesh.faces) == 0:
        # Try to construct a convex hull as a fallback surface mesh
        try:
            hull = mesh.convex_hull
            if isinstance(hull, trimesh.Trimesh) and hull.faces is not None and len(hull.faces) > 0:
                mesh = hull
                print("Input had no faces — using convex hull as a fallback mesh.")
            else:
                raise RuntimeError("Convex hull did not produce faces")
        except Exception:
            raise RuntimeError(
                "Input PLY appears to be a point cloud (no faces).\n"
                "Please provide a mesh with faces, or preprocess the point cloud to generate faces (e.g., Poisson reconstruction or surface reconstruction)."
            )
    else:
        if mesh.faces.shape[1] != 3:
            mesh = mesh.triangulate()

    # Ensure normals
    if not hasattr(mesh, 'vertex_normals') or mesh.vertex_normals is None or len(mesh.vertex_normals) == 0:
        mesh.rezero()
        mesh.generate_vertex_normals()

    # Optional centering
    if center:
        mesh.vertices -= mesh.centroid

    # Optional scaling: scale_to is the max dimension desired in meters (e.g., 1.0)
    if scale_tuple is not None:
        sx, sy, sz = scale_tuple
        mesh.apply_scale([sx, sy, sz])
        scale_x = sx
        scale_y = sy
        scale_z = sz
    elif scale_to is not None:
        bbox = mesh.bounding_box.extents
        max_dim = float(bbox.max())
        if max_dim > 0:
            factor = float(scale_to) / max_dim
            mesh.apply_scale(factor)
            scale_x = scale_y = scale_z = 1.0
        else:
            scale_x = scale_y = scale_z = 1.0
    else:
        # keep as-is; set mesh scale to 1 in mjcf
        scale_x = scale_y = scale_z = 1.0

    out_stem = path.stem
    out_obj = out_dir / f"{out_stem}.obj"
    out_xml = out_dir / f"{out_stem}.xml"

    # Export OBJ (ensure ascii for readability)
    mesh.export(out_obj, file_type='obj')
    print(f"Wrote OBJ: {out_obj}")

    # Write a simple MJCF/XML referencing the OBJ
    cam_pos = "1.5 0.5 0.8"
    cam_euler = "-20 0 110"
    xml_text = MJCF_TEMPLATE.format(
        name=out_stem,
        mesh_name=out_stem + "_mesh",
        mesh_file=out_obj.name,
        scale_x=scale_x,
        scale_y=scale_y,
        scale_z=scale_z,
        cam_pos=cam_pos,
        cam_euler=cam_euler,
    )
    out_xml.write_text(xml_text)
    print(f"Wrote MuJoCo XML: {out_xml}")

    return out_obj, out_xml


def main():
    parser = argparse.ArgumentParser(description="Convert PLY -> OBJ for MuJoCo and generate XML")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("file", nargs='?', help="Path to a single PLY file")
    group.add_argument("--dir", help="Directory containing PLY files to convert")

    parser.add_argument("--out", default="./mujoco_meshes", help="Output directory")
    parser.add_argument("--center", action='store_true', help="Center mesh at origin")
    parser.add_argument("--scale-to", type=float, default=None, help="Scale mesh so its max dimension equals this value (meters)")
    parser.add_argument("--scale", type=float, nargs=3, default=None, help="Directly scale mesh by (sx sy sz) before exporting")

    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    inputs = []
    if args.dir:
        p = Path(args.dir)
        for f in sorted(p.glob('*.ply')):
            inputs.append(f)
    else:
        if args.file is None:
            parser.error("Provide a file or --dir")
        inputs.append(Path(args.file))

    if len(inputs) == 0:
        print("No PLY files found to process")
        sys.exit(1)

    for p in inputs:
        try:
            process_file(p, out_dir, args.center, args.scale_to, args.scale)
        except Exception as e:
            print(f"Error processing {p}: {e}")

    print("Done. To view in MuJoCo, run:")
    print("  pip install mujoco mujoco-viewer")
    print("  export MUJOCO_GL=egl   # or osmesa/glfw")
    print("  python -c \"from mujoco import MjModel, MjData; from mujoco.viewer import launch; m=MjModel.from_xml_path('mujoco_meshes/NAME.xml'); d=MjData(m); with launch(m,d) as v: v.render()\"")


if __name__ == '__main__':
    main()
