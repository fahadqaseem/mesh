"""Common interface for swappable point-cloud-to-mesh stages."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import trimesh


class Mesher(ABC):
    """A mesher reads a point cloud .ply (xyz, optional normals/colours) and writes a mesh."""

    name: str = "base"

    @abstractmethod
    def run(self, points_ply: str | Path, out_path: str | Path, **kwargs) -> trimesh.Trimesh:
        """Mesh the point cloud in `points_ply`, save it to `out_path`, and return it."""


def get_mesher(name: str, **cfg) -> Mesher:
    """Factory so configs can switch meshers by name."""
    if name == "meshflow_cvpr":
        from .meshflow_cvpr import MeshFlowCVPR
        return MeshFlowCVPR(**cfg)
    if name == "hdpea":
        from .hdpea import HDPEA
        return HDPEA(**cfg)
    raise ValueError(f"Unknown mesher '{name}' (expected 'meshflow_cvpr' or 'hdpea')")
