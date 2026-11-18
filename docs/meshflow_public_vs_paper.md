# MeshFlow: paper vs released checkpoint

Fahad Qaseem Khawar, 17 Nov 2026. For Prof. Zhong (email of 17 Nov, item 4). Sources:
- **Supp. A**: CVPR 2026 supplementary, Section A "Additional Implementation Details", page 1 of 5 (local PDF in
  `docs/papers/`, not in git).
- **arXiv §**: arXiv:2606.04621v2. The HTML was read verbatim on 17 Nov.
- **Release**: facebookresearch/meshflow (`third_party/meshflow`) and the `facebook/meshflow` checkpoint
  (`ckpt/meshflow/meshflow/`).

Full quotes and context are in `docs/methods.md` section 8.

| item | paper says | released checkpoint has | match? |
|---|---|---|---|
| **Point count** | Supp. A: "We uniformly sampled 32,768 points for shape conditioning". arXiv §3.4 says the same, but as the "initial design": "In our initial design, we uniformly sampled 32768 points for shape conditioning". | 4,096 points: `data.n_verts: 4096` (`ckpt/meshflow/meshflow/config.yaml:2`), `denoiser_model.input_size: 4096` (`config.yaml:47`). The pipeline draws 4,096 at random (`third_party/meshflow/meshflow/utils/mesh.py:67-75`, `meshflow_pipeline.py:275-282`). | **No** |
| **Encoder / how the shape enters** | Supp. A: "fed into a similarly pre-trained shape encoder with a structure similar to Shape2VecSet [7] to obtain 2048 tokens". arXiv §3.4 adds "We then leveraged cross-attention (CA) to condition our DiT on these tokens" for the initial design. It then describes coarsely voxelised vertices as RoPE positions: "a voxel resolution of 32 strikes a reasonable trade-off". Supp. A: "The results presented in the paper therefore employ indices for RoPE." | No shape encoder and no 2,048-token cross-attention input. `model.pth` holds only `mesh_model` (the MeshVAE) and `denoiser_model` (the DiT); keys listed 17 Nov. The points are voxelised at 32^3 and their cell indices are the RoPE positions: `voxelize_resolution: 32` (`config.yaml:13`), `voxelize_pc` (`meshflow/models/meshflow_dit.py:19-36`), `get_rope_cond` (`meshflow_pipeline.py:293-296`). | **Partly**: RoPE on voxel indices matches; the encoder is absent |
| **Vertex-count control** | Supp. A: "The resulting tokens and the vertex count serve as conditional inputs for our Diffusion Transformer." arXiv §3.4: "We also provide the number of vertices concatenated with the time embedding". | Not trained. It needs `denoiser_model.use_proj_cond_on_temb: true` (`meshflow_pipeline.py:119-132`; `meshflow_dit.py:697`, `:721`, `:729`). The released config does not set it (`config.yaml:46-60`), and `model.pth` has no `t_embedder.cond_proj` weights (checked 15 Nov). README: "use the **num_verts** slider ... When the model config sets `denoiser_model.use_proj_cond_on_temb: true`" (`third_party/meshflow/README.md:131`). | **No** |
| **Training data** | Supp. A: "We utilize a proprietary dataset of approximately 600k high-quality 3D models from professional artists." | Not stated. Neither the code README nor the model card (`ckpt/meshflow/README.md`) names the training data, and there is no training code. Licence: FAIR Noncommercial Research License (`ckpt/meshflow/README.md:2`). | **Unknown** |
| **DiT size** | Supp. A and arXiv §4.1: "18 blocks and a 1024 hidden size, amounting to 427M parameters". arXiv appendix §F: "16 blocks and a 1536 hidden size, amounting to 895M parameters". | `layers: 16`, `width: 1536` (`config.yaml:51-52`); 895.6M parameters, counted from `model.pth` (17 Nov). This is the arXiv §F model. | **No** for Supp. A / §4.1; **yes** for arXiv §F |
| *Image condition (extra)* | Optional reference image | Optional: `visual_condition_dim: 1024` (`config.yaml:50`); needs DINOv3 weights. We pass a zero visual condition (`src/photo2mesh/mesh/meshflow_cvpr.py:89`). | n/a (unused) |

## What this means for our pipeline

1. **The paper's numbers do not describe the model we run.** Every MeshFlow number of ours is for the released
   895.6M checkpoint. It takes 4,096 points as 32^3 voxel indices, with no encoder and no vertex-count input.
   Whenever we put our numbers next to the paper's tables, we say so.
2. **MeshFlow sees only a coarse, sparse version of our point cloud.** It keeps 4,096 random points and quantises each
   to a 32^3 cell (3.2–4.8% of the object's size on our shapes). Point-cloud improvements below that scale cannot reach
   the mesh. This is why better clouds (gap fill, 1M points) did not give better meshes (14 and 15 Nov). It is also why
   we score the point cloud directly.
3. **No vertex budget.** The vertex count is the mask head's decision and depends on how the 4,096 input points are
   spread: about 250 on clean GT samples, about 900 on TSDF points, about 3,800 on FPS input. We cannot request a
   lightweight or a dense mesh. For an anisotropic, budgeted mesh this is a real limit, and it is one argument for
   HD-PEA or our own generator at the mesh stage.
4. **The 32,768-point encoder of the paper would use more of our cloud**, but it is not released. If it appears with the
   training code, it is worth rerunning the mesh rows.
5. **Unknown training data.** We cannot check for overlap with Thingi10K, so MeshFlow results on Thingi10K shapes carry
   that caveat.
