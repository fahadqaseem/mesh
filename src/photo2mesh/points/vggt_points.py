"""Multi-view photos → dense coloured point cloud with VGGT.

Requires the VGGT repo installed (see README): `pip install -e third_party/vggt`.
Weights (facebook/VGGT-1B) download from Hugging Face on first use.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..device import pick_device


def run_vggt(image_paths: list[str], use_depth_unprojection: bool = True) -> dict[str, np.ndarray]:
    """Run VGGT on a list of images of one object/scene.

    Returns a dict of numpy arrays:
        points (S, H, W, 3), conf (S, H, W), colors (S, H, W, 3) in [0, 1],
        extrinsic (S, 3, 4) camera-from-world (OpenCV), intrinsic (S, 3, 3),
        cam_centres (S, 3) in world coordinates.
    """
    import torch
    from vggt.models.vggt import VGGT
    from vggt.utils.geometry import unproject_depth_map_to_point_map
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    device = pick_device()
    # bf16 only on recent CUDA GPUs; fp32 elsewhere (MPS / CPU)
    use_amp = device == "cuda" and torch.cuda.get_device_capability()[0] >= 8
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device).eval()
    images = load_and_preprocess_images(image_paths).to(device)    # (S, 3, H, W) in [0, 1]

    with torch.no_grad():
        if use_amp:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = model(images)
        else:
            pred = model(images)

    extrinsic, intrinsic = pose_encoding_to_extri_intri(pred["pose_enc"], images.shape[-2:])
    extrinsic, intrinsic = extrinsic[0].float().cpu().numpy(), intrinsic[0].float().cpu().numpy()
    if use_depth_unprojection:
        # VGGT authors: unprojected depth is usually more accurate than the point head
        depth = pred["depth"][0].float().cpu().numpy()
        points = unproject_depth_map_to_point_map(depth, extrinsic, intrinsic)
        conf = pred["depth_conf"][0].float().cpu().numpy()
    else:
        points = pred["world_points"][0].float().cpu().numpy()
        conf = pred["world_points_conf"][0].float().cpu().numpy()

    rot, trans = extrinsic[:, :, :3], extrinsic[:, :, 3]
    cam_centres = -np.einsum("sji,sj->si", rot, trans)              # C = -R^T t
    colors = images.permute(0, 2, 3, 1).float().cpu().numpy()
    return {"points": np.asarray(points, np.float32), "conf": np.asarray(conf, np.float32),
            "colors": colors, "extrinsic": extrinsic, "intrinsic": intrinsic,
            "cam_centres": cam_centres.astype(np.float32)}


def flatten_views(pred: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Flatten per-view maps to per-point arrays, keeping which camera saw each point."""
    s, h, w, _ = pred["points"].shape
    view_id = np.repeat(np.arange(s), h * w)
    return {"points": pred["points"].reshape(-1, 3), "conf": pred["conf"].reshape(-1),
            "colors": pred["colors"].reshape(-1, 3), "viewpoints": pred["cam_centres"][view_id]}


def list_images(folder: str | Path) -> list[str]:
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    return sorted(str(p) for p in Path(folder).iterdir() if p.suffix.lower() in exts)
