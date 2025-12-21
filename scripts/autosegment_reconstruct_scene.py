"""CLI: Run SAM automatic segmentation and reconstruct each object with sam3d, then merge into full scene.

Usage example:
  python scripts/autosegment_reconstruct_scene.py /path/to/image.png \
      --sam-checkpoint checkpoints/sam/sam_vit_h.pth \
      --pipeline checkpoints/hf/pipeline.yaml \
      --outdir outputs --max-masks 30

The script will save per-object masks, metadata, per-object meshes (if available),
a merged gaussian scene `splat_all_objects.ply`, and a full scene mesh `scene_all_objects.ply`
with all objects transformed to their original scale and orientation in the output directory.
"""

import sys
import os
import argparse
import json
from typing import List

# ensure repository root and notebook are on sys.path so imports work
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
notebook_path = os.path.join(repo_root, "notebook")
if notebook_path not in sys.path:
    sys.path.insert(0, notebook_path)

from inference import Inference, load_image, make_scene
import numpy as np
import torch
import trimesh
from pytorch3d.transforms import quaternion_to_matrix

try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
except Exception as e:
    raise ImportError(
        "Unable to import `segment_anything`. Install it with: pip install git+https://github.com/facebookresearch/segment-anything.git"
    ) from e


def save_mask_png(mask_bool: np.ndarray, path: str):
    from PIL import Image

    im = Image.fromarray((mask_bool.astype(np.uint8) * 255))
    im.save(path)


def bbox_from_mask(mask: np.ndarray):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return [0, 0, 0, 0]
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return [x0, y0, x1, y1]


def iou(mask_a: np.ndarray, mask_b: np.ndarray):
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return 0.0 if union == 0 else float(inter) / float(union)


def transform_mesh_to_scene(vertices, rotation, translation, scale):
    """
    Transform mesh vertices to scene coordinates using pose information.
    
    Args:
        vertices: (N, 3) numpy array of vertex positions
        rotation: (4,) tensor quaternion [w, x, y, z] or (1, 4) tensor
        translation: (3,) tensor or (1, 3) tensor
        scale: (3,) tensor or (1, 3) tensor, or scalar
    
    Returns:
        Transformed vertices as numpy array
    """
    # Convert to tensors and ensure correct shapes
    if isinstance(vertices, np.ndarray):
        verts = torch.from_numpy(vertices).float()
    else:
        verts = vertices.float()
    # Ensure verts lives on the same device as the pose tensors to avoid
    # CUDA/CPU mismatches when doing matrix ops below. Determine device from
    # rotation/translation/scale if available.
    device = None
    for t in (rotation, translation, scale):
        if t is not None:
            device = t.device
            break
    if device is None:
        device = torch.device("cpu")
    verts = verts.to(device)
    
    # Handle quaternion shape
    if rotation.dim() == 2:
        quat = rotation.squeeze(0)
    else:
        quat = rotation
    
    # Handle translation shape
    if translation.dim() == 2:
        trans = translation.squeeze(0)
    else:
        trans = translation
    
    # Handle scale shape
    if scale.dim() == 2:
        scale_val = scale.squeeze(0)
    else:
        scale_val = scale
    
    # If scale is uniform (all values same), use scalar
    if scale_val.numel() == 3:
        if torch.allclose(scale_val, scale_val[0]):
            scale_val = scale_val[0]
        else:
            # Non-uniform scale - apply per-axis
            scale_val = scale_val.unsqueeze(0)
    
    # Convert quaternion to rotation matrix
    R = quaternion_to_matrix(quat.unsqueeze(0)).squeeze(0)
    
    # Apply transformation: scale -> rotate -> translate
    if scale_val.dim() == 0:
        # Uniform scale
        verts_scaled = verts * scale_val.item()
    else:
        # Non-uniform scale
        verts_scaled = verts * scale_val
    
    verts_rotated = verts_scaled @ R.T
    verts_transformed = verts_rotated + trans.unsqueeze(0)
    
    # Ensure tensor is on CPU before converting to numpy to avoid
    # the "can't convert cuda device tensor to numpy" error.
    return verts_transformed.cpu().numpy()


def extract_vertex_colors(mesh):
    """Extract vertex colors from mesh, matching demo_with_colors.py behavior."""
    vcols = None
    if getattr(mesh, "vertex_attrs", None) is not None:
        va = mesh.vertex_attrs
        va_np = va.detach().cpu().numpy()
        if va_np.shape[1] >= 3:
            # Take first 3 channels as RGB and scale to 0-255 if needed
            cols = va_np[:, :3].copy()
            if cols.max() <= 1.001:
                cols = (cols * 255).astype(np.uint8)
            else:
                cols = cols.astype(np.uint8)
            vcols = cols
    return vcols


def main(argv: List[str]):
    p = argparse.ArgumentParser()
    p.add_argument("image", type=str, help="Path to input image")
    p.add_argument("--sam-checkpoint", type=str, default="checkpoints/sam/sam_vit_h.pth")
    p.add_argument("--pipeline", type=str, default="checkpoints/hf/pipeline.yaml")
    p.add_argument("--min-mask-area", type=int, default=2000)
    p.add_argument("--max-masks", type=int, default=10)
    p.add_argument("--outdir", type=str, default="outputs")
    p.add_argument("--device", type=str, default=None, help="cuda or cpu (default: auto detect)")
    p.add_argument("--overlap-thresh", type=float, default=0.7, help="IoU threshold to skip highly overlapping masks")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    image_path = args.image
    sam_checkpoint = args.sam_checkpoint
    pipeline_config = args.pipeline
    min_mask_area = args.min_mask_area
    max_masks = args.max_masks
    outdir = args.outdir
    overlap_thresh = args.overlap_thresh
    seed = args.seed

    os.makedirs(outdir, exist_ok=True)
    # create a run-specific directory named after the image (basename without extension)
    image_basename = os.path.splitext(os.path.basename(image_path))[0]
    run_dir = os.path.join(outdir, image_basename)
    # if the run directory already exists, append a numeric suffix to avoid overwriting
    if os.path.exists(run_dir):
        suffix = 1
        while True:
            candidate = f"{run_dir}_{suffix}"
            if not os.path.exists(candidate):
                run_dir = candidate
                break
            suffix += 1
    os.makedirs(run_dir, exist_ok=True)

    masks_outdir = os.path.join(run_dir, "masks")
    meshes_outdir = os.path.join(run_dir, "meshes")
    os.makedirs(masks_outdir, exist_ok=True)
    os.makedirs(meshes_outdir, exist_ok=True)

    device = args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading SAM checkpoint and building mask generator...")
    sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
    sam.to(device)
    mask_generator = SamAutomaticMaskGenerator(sam)

    print("Loading image:", image_path)
    full_image = load_image(image_path)
    sam_image = full_image[..., :3]

    print("Generating masks with SAM (may take a while)...")
    sam_masks = mask_generator.generate(sam_image)
    print(f"SAM produced {len(sam_masks)} masks")

    masks_bool = [m["segmentation"].astype(bool) for m in sam_masks]
    areas = [m.sum() for m in masks_bool]
    order = np.argsort(areas)[::-1]

    # Filter by area and deduplicate by IoU
    accepted_idxs = []
    accepted_masks = []
    for idx in order:
        if areas[idx] < min_mask_area:
            continue
        m = masks_bool[idx]
        skip = False
        for am in accepted_masks:
            if iou(m, am) >= overlap_thresh:
                skip = True
                break
        if skip:
            continue
        accepted_idxs.append(idx)
        accepted_masks.append(m)
        if len(accepted_idxs) >= max_masks:
            break

    print(f"Keeping {len(accepted_idxs)} masks after filtering (min_area={min_mask_area}, overlap<{overlap_thresh})")

    # Instantiate the reconstruction pipeline once
    print("Loading sam3d pipeline from:", pipeline_config)
    inference = Inference(pipeline_config, compile=False)

    outputs = []
    metadata = []
    scene_meshes = []  # Store meshes for scene reconstruction
    
    for j, idx in enumerate(accepted_idxs):
        mask = masks_bool[idx]
        area = int(areas[idx])
        bbox = bbox_from_mask(mask)
        print(f"[{j+1}/{len(accepted_idxs)}] Running reconstruction (area={area}, bbox={bbox})")
        try:
            out = inference(full_image, mask, seed=seed + j)
        except Exception as e:
            print(f"Reconstruction failed for mask {j}: {e}")
            continue
        outputs.append(out)

        # save mask png
        mask_path = os.path.join(masks_outdir, f"mask_{j:03d}.png")
        save_mask_png(mask, mask_path)

        # save metadata
        meta = {"index": j, "area": area, "bbox": bbox, "mask_path": mask_path}
        metadata.append(meta)

        # Extract pose information for scene reconstruction
        rotation = out.get("rotation", None)
        translation = out.get("translation", None)
        scale = out.get("scale", None)
        
        if rotation is not None:
            meta["rotation"] = rotation.detach().cpu().numpy().tolist()
        if translation is not None:
            meta["translation"] = translation.detach().cpu().numpy().tolist()
        if scale is not None:
            meta["scale"] = scale.detach().cpu().numpy().tolist()

        # export mesh if present
        mesh_res = out.get("mesh", None)
        if mesh_res and len(mesh_res) > 0:
            m = mesh_res[0]
            verts = m.vertices.detach().cpu().numpy()
            faces = m.faces.detach().cpu().numpy().astype(np.int64)
            
            # Extract vertex colors using the same method as demo_with_colors.py
            vcols = extract_vertex_colors(m)

            # Create trimesh with vertex colors
            tri = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
            if vcols is not None:
                tri.visual.vertex_colors = vcols
                print(f"Extracted vertex colors: shape={vcols.shape}, range=[{vcols.min()}, {vcols.max()}]")
            
            # Export PLY, OBJ, and GLB (all formats support vertex colors)
            ply_out = os.path.join(meshes_outdir, f"object_{j:03d}.ply")
            obj_out = os.path.join(meshes_outdir, f"object_{j:03d}.obj")
            glb_out = os.path.join(meshes_outdir, f"object_{j:03d}.glb")
            
            tri.export(ply_out)
            try:
                tri.export(obj_out)
                meta["mesh_path_obj"] = obj_out
            except Exception:
                meta["mesh_path_obj"] = None
            try:
                tri.export(glb_out)
                meta["glb_path"] = glb_out
            except Exception:
                meta["glb_path"] = None
            
            meta["mesh_path"] = ply_out
            print(f"Exported mesh: {ply_out}, {obj_out}, {glb_out}")
            
            # Store mesh and pose for scene reconstruction
            if rotation is not None and translation is not None and scale is not None:
                scene_meshes.append({
                    "mesh": tri,
                    "rotation": rotation,
                    "translation": translation,
                    "scale": scale,
                    "vertex_colors": vcols,
                })

        # try to free GPU memory if needed
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    # save metadata file inside the run directory
    meta_path = os.path.join(run_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print("Saved metadata to", meta_path)

    # merge outputs into a combined gaussian scene if we have any
    if len(outputs) > 0:
        print("Merging outputs into a single scene and saving...")
        scene_gs = make_scene(*outputs)
        scene_out_path = os.path.join(run_dir, "splat_all_objects.ply")
        scene_gs.save_ply(scene_out_path)
        print("Saved merged gaussian scene to:", scene_out_path)

        # Reconstruct full scene mesh with correct scale and orientation
        if len(scene_meshes) > 0:
            print("Reconstructing full scene mesh with original scale and orientation...")
            transformed_meshes = []
            
            for scene_data in scene_meshes:
                mesh = scene_data["mesh"]
                rotation = scene_data["rotation"]
                translation = scene_data["translation"]
                scale = scene_data["scale"]
                vcols = scene_data.get("vertex_colors", None)
                
                # Transform vertices to scene coordinates
                verts_scene = transform_mesh_to_scene(
                    mesh.vertices,
                    rotation,
                    translation,
                    scale
                )
                
                # Create new mesh with transformed vertices and colors
                scene_mesh = trimesh.Trimesh(
                    vertices=verts_scene,
                    faces=mesh.faces,
                    vertex_colors=vcols if vcols is not None else None,
                    process=False
                )
                transformed_meshes.append(scene_mesh)
            
            # Combine all transformed meshes into a single scene
            scene_combined = trimesh.util.concatenate(transformed_meshes)
            scene_ply = os.path.join(run_dir, "scene_all_objects.ply")
            scene_obj = os.path.join(run_dir, "scene_all_objects.obj")
            scene_glb = os.path.join(run_dir, "scene_all_objects.glb")
            
            scene_combined.export(scene_ply)
            try:
                scene_combined.export(scene_obj)
                print("Saved full scene mesh:", scene_ply, scene_obj)
            except Exception:
                print("Warning: failed to export scene OBJ")
                print("Saved full scene mesh:", scene_ply)
            try:
                scene_combined.export(scene_glb)
                print("Also saved scene GLB:", scene_glb)
            except Exception:
                pass
        else:
            print("No meshes with pose information available for scene reconstruction")
    else:
        print("No successful outputs to merge.")


if __name__ == "__main__":
    main(sys.argv[1:])

