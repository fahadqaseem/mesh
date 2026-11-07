"""Placeholder adapter for HD-PEA (Hongbo Li / Zhong lab): anisotropic meshing from points + normals.

Status: code not public yet; Prof. Zhong will share access. Expected input is
6 values per point (x, y, z, nx, ny, nz). Internally it embeds points in 8-D
(learned features), samples uniformly there, and maps back to an anisotropic mesh.

When the code arrives: clone it to third_party/hdpea, then implement run() by
calling its inference entry point on `points_ply` (which already contains normals).
"""
from __future__ import annotations

from pathlib import Path

import trimesh

from .base import Mesher


class HDPEA(Mesher):
    name = "hdpea"

    def __init__(self, repo_dir: str = "third_party/hdpea", **kwargs):
        self.repo_dir = repo_dir
        self.kwargs = kwargs

    def run(self, points_ply: str | Path, out_path: str | Path, **kwargs) -> trimesh.Trimesh:
        raise NotImplementedError(
            "HD-PEA code is not available yet. Ask Prof. Zhong / Hongbo for access, "
            "clone it into third_party/hdpea, then implement this adapter."
        )
