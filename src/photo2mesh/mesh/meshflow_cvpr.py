"""Adapter for CVPR 2026 MeshFlow (facebookresearch/meshflow): point cloud in, artist-style mesh out.

Setup (see README):
    git clone https://github.com/facebookresearch/meshflow.git third_party/meshflow
    hf download facebook/meshflow --local-dir ckpt/meshflow   (files land in ckpt/meshflow/meshflow/)

What this adapter adds on top of MeshFlowPipeline (third_party is never edited):
- Device split. MeshFlow's from_pretrained silently uses CPU whenever CUDA is absent. On the Mac we put the DiT on MPS
  (flow sampling is ~1.8x faster there) and keep the VAE decoder + mesh extraction on CPU (extraction is ~20x slower
  on MPS). device="hybrid" is the default on Apple silicon; "cuda", "mps", "cpu" put everything on one device.
- guidance_scale=1.0 without a reference image: CFG would compare two identical empty conditions (2x compute, same result).
- Output frame. MeshFlow normalises the input cloud itself (bbox centre, max radius 1) and generates in that frame.
  We map the mesh back with the exact inverse, computed from the input cloud (no fitting), so the mesh lands in the
  frame of the input points.
MeshFlow only uses xyz: normals and colours in the .ply are ignored at this stage.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import trimesh

from ..device import pick_device
from .base import Mesher

REPO_ROOT = Path(__file__).resolve().parents[3]


def _sync(device: str) -> None:
    import torch
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def meshflow_input_normalisation(points: np.ndarray) -> tuple[np.ndarray, float]:
    """(centre, radius) MeshFlow uses for a point cloud: bbox centre, max distance from it ("bsphere", size 2)."""
    centre = (points.min(axis=0) + points.max(axis=0)) / 2.0
    return centre, float(np.linalg.norm(points - centre, axis=1).max())


class MeshFlowCVPR(Mesher):
    name = "meshflow_cvpr"

    def __init__(self, ckpt_dir: str = "ckpt/meshflow/meshflow", steps: int = 28, seed: int = 42,
                 dtype: str = "fp16", num_verts: int | None = None, device: str | None = None):
        self.ckpt_dir = str(ckpt_dir)
        self.steps, self.seed, self.num_verts = steps, seed, num_verts
        picked = pick_device(None if device in (None, "hybrid") else device)
        self.device = "hybrid" if (device == "hybrid" or (device is None and picked == "mps")) else picked
        self.sample_device = "mps" if self.device == "hybrid" else self.device
        self.decode_device = "cpu" if self.device == "hybrid" else self.device
        # half precision is only reliable on CUDA
        self.dtype = dtype if self.device == "cuda" else "fp32"
        self._pipeline = None
        self.last_timing: dict[str, float] = {}

    def _load(self):
        if self._pipeline is None:
            third_party = REPO_ROOT / "third_party" / "meshflow"
            if third_party.exists() and str(third_party) not in sys.path:
                sys.path.insert(0, str(third_party))
            import torch
            from meshflow.pipelines import MeshFlowPipeline
            pipe = MeshFlowPipeline.from_pretrained(self.ckpt_dir, device="cuda" if self.device == "cuda" else "cpu",
                                                    dtype=self.dtype)
            if self.sample_device != pipe.device.type:
                pipe._device = torch.device(self.sample_device)                 # tensors for sampling go here
                pipe.models["denoiser"].to(self.sample_device)
            pipe.models["mesh_model"].to(self.decode_device)
            self._pipeline = pipe
        return self._pipeline

    def generate(self, points_ply: str | Path, seed: int | None = None, steps: int | None = None):
        """Run MeshFlow on a point cloud; returns (trimesh in MeshFlow's normalised frame, latents, timing dict)."""
        import torch
        pipe = self._load()
        seed = self.seed if seed is None else int(seed)
        steps = self.steps if steps is None else int(steps)
        torch.manual_seed(seed)                       # MeshFlow's input resampling uses the global torch RNG
        timing = {}
        t0 = time.time()
        surface = pipe.sample_surface_points(str(points_ply))
        rope = pipe.get_rope_cond(surface)
        visual = pipe.get_visual_cond(None, guidance_scale=1.0)
        kwargs = {} if self.num_verts is None else {"num_verts": self.num_verts}
        latents = pipe.sample_latent(visual, rope, steps=steps, guidance_scale=1.0, seed=seed, disable_prog=True, **kwargs)
        _sync(self.sample_device)
        timing["sample_s"] = time.time() - t0
        t0 = time.time()
        vae = pipe.models["mesh_model"]
        with torch.no_grad():
            lat = latents.to(self.decode_device, torch.float32)
            out = vae.extract_mesh(vae.decode(vae.denormalize(lat)), fill_holes=False)[0]
        _sync(self.decode_device)
        timing["extract_s"] = time.time() - t0
        timing["total_s"] = timing["sample_s"] + timing["extract_s"]
        return out.to_trimesh(), latents, timing

    def run(self, points_ply: str | Path, out_path: str | Path, seed: int | None = None, **kwargs) -> trimesh.Trimesh:
        """Mesh the cloud and save it in the frame of the input points (exact inverse of MeshFlow's normalisation)."""
        from ..io import load_points_ply
        mesh, _, timing = self.generate(points_ply, seed=seed, steps=kwargs.get("steps"))
        centre, radius = meshflow_input_normalisation(load_points_ply(points_ply)["points"].astype(np.float64))
        mesh = trimesh.Trimesh(mesh.vertices * radius + centre, mesh.faces, process=False)
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(str(out_path))
        self.last_timing = timing
        return mesh
