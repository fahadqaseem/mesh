"""Adapter for CVPR 2026 MeshFlow (facebookresearch/meshflow): point cloud in, artist-style mesh out.

Setup (see README):
    git clone https://github.com/facebookresearch/meshflow.git third_party/meshflow
    pip install -e third_party/meshflow   (or add it to PYTHONPATH)
    download config.yaml + model.pth from huggingface.co/facebook/meshflow (hf download --local-dir ckpt/meshflow puts them in ckpt/meshflow/meshflow/)

MeshFlow normalises point-cloud input itself (bounding sphere) and only uses xyz,
so normals/colours in the .ply are ignored at this stage.
"""
from __future__ import annotations

import sys
from pathlib import Path

import trimesh

from ..device import pick_device
from .base import Mesher

REPO_ROOT = Path(__file__).resolve().parents[3]


class MeshFlowCVPR(Mesher):
    name = "meshflow_cvpr"

    def __init__(self, ckpt_dir: str = "ckpt/meshflow/meshflow", steps: int = 28, seed: int = 42,
                 dtype: str = "fp16", num_verts: int | None = None, device: str | None = None):
        self.ckpt_dir = str(ckpt_dir)
        self.steps, self.seed, self.num_verts = steps, seed, num_verts
        self.device = pick_device(device)
        # half precision is only reliable on CUDA
        self.dtype = dtype if self.device == "cuda" else "fp32"
        self._pipeline = None

    def _load(self):
        if self._pipeline is None:
            third_party = REPO_ROOT / "third_party" / "meshflow"
            if third_party.exists() and str(third_party) not in sys.path:
                sys.path.insert(0, str(third_party))
            from meshflow.pipelines import MeshFlowPipeline
            # MeshFlow's from_pretrained silently falls back to CPU whenever CUDA is absent
            # (so "mps" would be ignored). Load there, then move to the device we picked.
            self._pipeline = MeshFlowPipeline.from_pretrained(self.ckpt_dir, device=self.device, dtype=self.dtype)
            if self._pipeline.device.type != self.device:
                self._pipeline.to(self.device)
        return self._pipeline

    def run(self, points_ply: str | Path, out_path: str | Path, **kwargs) -> trimesh.Trimesh:
        pipeline = self._load()
        run_kwargs = {"mesh": str(points_ply), "image": None, "steps": self.steps, "seed": self.seed}
        if self.num_verts:
            run_kwargs["num_verts"] = self.num_verts   # rough target vertex count
        run_kwargs.update(kwargs)
        mesh = pipeline.run(**run_kwargs).to_trimesh()
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(str(out_path))
        return mesh
