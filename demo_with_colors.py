import sys
import numpy as np
import trimesh
import torch
# import inference code
sys.path.append("notebook")
from inference import Inference, load_image, load_single_mask

# load model
tag = "hf"
config_path = f"checkpoints/{tag}/pipeline.yaml"
inference = Inference(config_path, compile=False)

# load image (RGBA only, mask is embedded in the alpha channel)
image = load_image("notebook/images/shutterstock_stylish_kidsroom_1640806567/image.png")
mask = load_single_mask("notebook/images/shutterstock_stylish_kidsroom_1640806567", index=14)

# run model
output = inference(image, mask, seed=42)
print(output)
# export gaussian splat
output["gs"].save_ply(f"splat.ply")
print("Your reconstruction has been saved to splat.ply")


m = output["mesh"][0]  # MeshExtractResult

# move to CPU numpy
verts = m.vertices.detach().cpu().numpy()
faces = m.faces.detach().cpu().numpy().astype(np.int64)

# Extract vertex colors: inspect and extract RGB if present
vcols = None
if getattr(m, "vertex_attrs", None) is not None:
    va = m.vertex_attrs
    va_np = va.detach().cpu().numpy()
    print("vertex_attrs shape:", va_np.shape)
    # Try to find RGB columns — often first 3 or last 3 columns; inspect to pick correct slice:
    if va_np.shape[1] >= 3:
        # heuristic: take first 3 channels as RGB and scale to 0-255 if needed
        cols = va_np[:, :3].copy()
        if cols.max() <= 1.001:
            cols = (cols * 255).astype(np.uint8)
        else:
            cols = cols.astype(np.uint8)
        vcols = cols
        print(f"Extracted vertex colors: shape={vcols.shape}, range=[{vcols.min()}, {vcols.max()}]")
else:
    print("No vertex colors found in mesh")

# create trimesh with vertex colors
tri = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
if vcols is not None:
    # attach as vertex colors (trimesh expects (N,3) uint8)
    tri.visual.vertex_colors = vcols
    print("Attached vertex colors to mesh")
else:
    print("Warning: No vertex colors available - exporting mesh without colors")

# export with colors
print("\nExporting meshes with colors...")
tri.export("splat_from_meshextract.obj")   # OBJ (will include vertex colors if present)
tri.export("splat_from_meshextract.ply")   # PLY (will include vertex colors if present)
tri.export("splat_from_meshextract.glb")   # GLB (binary glTF, will include vertex colors if present)
print("Exported splat_from_meshextract.{obj,ply,glb}")

# Also export a version explicitly named with colors for clarity
if vcols is not None:
    tri_colored = trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=vcols, process=False)
    tri_colored.export("splat_from_meshextract_colored.obj")
    tri_colored.export("splat_from_meshextract_colored.ply")
    print("Also exported splat_from_meshextract_colored.{obj,ply}")

print("\nDone! Check the exported files for vertex colors.")
print("Note: OBJ format supports vertex colors via 'v x y z r g b' format")
print("      PLY format supports vertex colors in the vertex data")
print("      GLB format supports vertex colors in the glTF structure")

