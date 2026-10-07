# Source assets for twin/build_assets.py

- `soma/`: NVIDIA SOMA-X (Apache-2.0), from Hugging Face `nvidia/SOMA-X` (tag v0.3.3): `SOMA_neutral.npz`,
  `SOMA_template_rig.usda`, `SOMA_procedural_transforms.json`, plus LICENSE and README. Python package: `py-soma-x`.
  Native SOMA PCA identities only (no SMPL/MHR files needed).
- `g1/`: Unitree G1 from MuJoCo Menagerie `unitree_g1` (BSD-3-Clause, Unitree Robotics): `g1.xml`, `assets/*.STL`, LICENSE.

Rebuild the baked actors (twin/assets/): `.venv/bin/python twin/build_assets.py` (needs `py-soma-x`, `mujoco`, `trimesh`,
`fast-simplification`, `scipy`).
