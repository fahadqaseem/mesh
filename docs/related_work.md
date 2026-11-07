# Related work

Short notes per paper: what it does, what it takes in, what it outputs, and how we differ.

| Paper | Input | Output | One-line takeaway | How we differ |
|---|---|---|---|---|
| VGGT (CVPR 2025, Meta) | multi-view images | cameras, depth, point maps | feed-forward 3D from photos in one pass | we use it as stage 1, not a contribution |
| MeshFlow (CVPR 2026, Meta) | point cloud (xyz) | artist-style triangle mesh | flow model over mesh tokens | baseline mesher; ignores normals and colour |
| HD-PEA (Zhong lab, Hongbo Li) | points + normals (6-D) | anisotropic mesh | embed in 8-D, sample uniformly, map back | our target mesher |
| LATO | TODO | TODO | TODO | unified model vs our two stages |
| MeshCraft | TODO | TODO | TODO | unified model vs our two stages |

## Questions to answer before reviewers ask
- Why two stages instead of one end-to-end model? (modularity, swap any mesher, reuse strong pretrained parts)
- What do we get that LATO / MeshCraft cannot? (anisotropy, fewer faces for the same error, texture)
