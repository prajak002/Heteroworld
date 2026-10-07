"""Bake the animated actors for the BoundlessNYC twin replay (twin/replay.js).

SOMA pedestrians: NVIDIA SOMA-X body model (Apache-2.0, py-soma-x), native SOMA PCA identities, low LOD (4,505
vertices). A procedural walk cycle is posed with SOMA's own LBS, and every frame's vertices are stored (int16).
Unitree G1 humanoids: MuJoCo Menagerie unitree_g1 (BSD-3-Clause, Unitree Robotics). A procedural gait is set on the
leg/arm joints, MuJoCo forward kinematics gives every visual geom's pose per frame, and the decimated link meshes are
stored once.

Output (twin/assets/): soma.json + soma.bin, g1.json + g1.bin. Everything is in three.js axes with the actor facing +x,
y up, z to its right, feet on y = 0. Source assets live in twin/assets_src/ (see the README there for the download).

Usage: .venv/bin/python twin/build_assets.py [--check out.png]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SRC = HERE / "assets_src"
OUT = HERE / "assets"
FRAMES = 32            # frames per walk cycle (two steps)
N_IDENTITIES = 10      # distinct SOMA bodies
SEED = 7


# ---------------------------------------------------------------- small rotation helpers
def rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


# ---------------------------------------------------------------- SOMA pedestrians
def soma_world_rotations(phase: float | None, style: dict) -> dict[str, np.ndarray]:
    """Target world rotations (relative to the T-pose) for a walk at `phase` in [0, 1); None = standing.

    SOMA axes: x = the body's left, y = up, z = forward. A positive rotation about x swings a limb backward.
    """
    arm_down = style["arm_down"]
    if phase is None:
        sw = dict(thL=0.0, thR=0.0, kL=0.04, kR=0.04, fL=0.0, fR=0.0, yaw=0.0, armL=0.0, armR=0.0, bob=0.0)
    else:
        s = 2 * math.pi * phase
        a = style["stride"]
        knee = lambda u: 0.08 + 0.95 * max(0.0, math.cos(u + 0.35)) ** 2 + 0.12 * max(0.0, math.sin(u - 0.9)) ** 2
        sw = dict(thL=a * math.sin(s), thR=a * math.sin(s + math.pi), kL=knee(s), kR=knee(s + math.pi),
                  fL=-0.22 * math.sin(s), fR=-0.22 * math.sin(s + math.pi), yaw=0.07 * math.sin(s),
                  armL=style["arm_swing"] * math.sin(s), armR=style["arm_swing"] * math.sin(s + math.pi))
    W = {}
    W["Hips"] = ry(-sw["yaw"])
    W["Chest"] = ry(0.6 * sw["yaw"])
    W["Head"] = np.eye(3)
    W["Neck1"] = np.eye(3)
    for side, sign in (("Left", 1), ("Right", -1)):
        th, k, f = sw["th" + side[0]], sw["k" + side[0]], sw["f" + side[0]]
        W[side + "Leg"] = rx(-th)
        W[side + "Shin"] = rx(-th + k)
        W[side + "Foot"] = rx(f)
        W[side + "ToeBase"] = rx(min(0.0, f) * 0.5)
        down = rz(-sign * arm_down)
        swing = sw["arm" + side[0]]
        W[side + "Arm"] = rx(swing) @ down
        W[side + "ForeArm"] = rx(swing - style["elbow"] - 0.25 * max(0.0, -swing)) @ down
        W[side + "Hand"] = W[side + "ForeArm"]
    return W


def soma_pose_vector(names, parents, W_target):
    """Turn target world rotations into SOMA's joint-local axis-angle pose (77 x 3)."""
    from scipy.spatial.transform import Rotation

    world = {}
    pose = np.zeros((len(names) - 1, 3))
    for j, n in enumerate(names):
        if j == 0:
            world[j] = np.eye(3)
            continue
        Wp = world[parents[j]]
        Wj = W_target.get(n, Wp)
        world[j] = Wj
        pose[j - 1] = Rotation.from_matrix(Wp.T @ Wj).as_rotvec()
    return pose


def soma_colors(rest, joints, ix, rng):
    """Per-vertex clothing colours from the rest (T-pose) geometry: hair, skin, top, trousers, shoes."""
    skins = [(0.98, 0.84, 0.71), (0.92, 0.72, 0.56), (0.78, 0.57, 0.40), (0.55, 0.38, 0.26), (0.36, 0.24, 0.16)]
    tops = [(0.12, 0.13, 0.16), (0.55, 0.12, 0.14), (0.16, 0.30, 0.55), (0.82, 0.80, 0.74), (0.22, 0.40, 0.28),
            (0.85, 0.55, 0.15), (0.45, 0.45, 0.48), (0.10, 0.20, 0.35), (0.62, 0.30, 0.45), (0.95, 0.95, 0.95)]
    bottoms = [(0.10, 0.12, 0.18), (0.20, 0.25, 0.38), (0.30, 0.28, 0.24), (0.08, 0.08, 0.09), (0.55, 0.50, 0.42)]
    hairs = [(0.08, 0.06, 0.05), (0.25, 0.16, 0.09), (0.55, 0.38, 0.20), (0.70, 0.68, 0.64), (0.12, 0.10, 0.09)]
    shoes = [(0.06, 0.06, 0.07), (0.92, 0.92, 0.90), (0.35, 0.22, 0.14)]
    skin, top, bot, hair, shoe = (np.array(c[rng.integers(len(c))]) for c in (skins, tops, bottoms, hairs, shoes))
    long_sleeve, shorts = rng.random() < 0.55, rng.random() < 0.2
    x, y, z = rest[:, 0], rest[:, 1], rest[:, 2]
    J = lambda n: joints[ix[n]]
    col = np.tile(top, (len(rest), 1))
    hips_y, knee_y, ankle_y = J("Hips")[1], J("LeftShin")[1], J("LeftFoot")[1]
    neck_y, head = J("Neck2")[1], J("Head")
    elbow_x, wrist_x = abs(J("LeftForeArm")[0]), abs(J("LeftHand")[0])
    leg = y < hips_y - 0.02
    col[leg] = bot
    if shorts:
        col[leg & (y < knee_y + 0.06)] = skin
    col[y < ankle_y + 0.07] = shoe
    col[(y > neck_y - 0.01) & (np.abs(x) < 0.14)] = skin
    col[np.abs(x) > (wrist_x if long_sleeve else elbow_x - 0.05)] = skin
    hair_mask = (y > head[1] + 0.07) | ((y > head[1] - 0.04) & (z < head[2] - 0.03) & (np.abs(x) < 0.11))
    col[hair_mask & (y > neck_y + 0.02)] = hair
    return col


def build_soma(check: str | None):
    import torch

    torch.set_grad_enabled(False)
    from soma import SOMALayer

    L = SOMALayer(data_root=str(SRC / "soma"), device="cpu", identity_model_type="soma", lod="low", mode="torch",
                  correctives_model_path=None)
    names = list(L.public_joint_names)
    ix = {n: i - 1 for i, n in enumerate(names)}          # index into pose / joints arrays (Root excluded)
    rig = np.load(SRC / "soma" / "SOMA_neutral.npz", allow_pickle=False)
    assert list(map(str, rig["t_pose_world__v0.2.0__joint_names"])) == names
    parents = rig["t_pose_world__v0.2.0__parent_ids"].astype(int)
    faces = rig["triangles_low"].astype(np.int64)
    rng = np.random.default_rng(SEED)

    phases = [None] + [k / FRAMES for k in range(FRAMES)]
    bodies, frames_all = [], []
    for b in range(N_IDENTITIES):
        coeffs = np.zeros((1, 128), dtype=np.float32)
        coeffs[0, :12] = rng.normal(0, 0.85, 12)
        L.prepare_identity(torch.from_numpy(coeffs))
        style = dict(stride=rng.uniform(0.36, 0.46), arm_swing=rng.uniform(0.22, 0.38),
                     arm_down=rng.uniform(1.22, 1.32), elbow=rng.uniform(0.15, 0.35))
        rest = L.pose(torch.zeros(1, 77, 3), apply_correctives=False)
        rest_v, rest_j = rest.vertices[0].numpy(), rest.joints[0].numpy()
        poses = np.stack([soma_pose_vector(names, parents, soma_world_rotations(p, style)) for p in phases])
        out = L.pose(torch.from_numpy(poses).float(), apply_correctives=False)
        V = out.vertices.numpy().astype(np.float64)                   # (F, V, 3) SOMA axes
        V[:, :, 1] -= V[:, :, 1].min(axis=1, keepdims=True)           # lowest point on the ground each frame
        hz = out.joints.numpy()[:, ix["Hips"], 2:3]
        V[:, :, 2] -= hz                                              # hips above the origin
        # SOMA (x left, y up, z fwd) -> actor frame (x fwd, y up, z right)
        A = np.stack([V[..., 2], V[..., 1], -V[..., 0]], axis=-1)
        height = float(A[0, :, 1].max())
        bodies.append(dict(height=round(height, 3), stride=round(4 * 0.53 * height * math.sin(style["stride"]) * 0.95, 3),
                           colors=soma_colors(rest_v, rest_j, ix, rng)))
        frames_all.append(A.astype(np.float32))
        print(f"soma body {b}: height {height:.2f} m")

    if check:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(3, 9, figsize=(22, 9))
        for r in range(3):
            for c, k in enumerate([0, 1, 5, 9, 13, 17, 21, 25, 29]):
                ax = axs[r, c]
                A = frames_all[r][k]
                if r < 2:
                    ax.scatter(A[:, 0], A[:, 1], s=0.2, c=bodies[r]["colors"])
                else:
                    ax.scatter(A[:, 2], A[:, 1], s=0.2, c=bodies[r]["colors"])
                ax.set_aspect("equal"); ax.set_xlim(-0.8, 0.8); ax.set_ylim(0, 2); ax.set_title(f"b{r} f{k}")
        fig.savefig(check, dpi=70)
        print("check image", check)

    # write: int16 positions (scale 1/10000 m), uint8 colours, uint32 faces
    OUT.mkdir(exist_ok=True)
    blobs, meta, off = [], dict(frames=FRAMES, standing_frame=0, cycle_frames=[1, FRAMES], vertex_count=int(frames_all[0].shape[1]),
                               position_scale=1e-4, bodies=[]), 0

    def put(arr):
        nonlocal off
        buf = np.ascontiguousarray(arr).tobytes()
        pad = (-len(buf)) % 4
        blobs.append(buf + b"\0" * pad)
        start = off
        off += len(buf) + pad
        return [start, len(buf)]

    meta["faces"] = put(faces.astype(np.uint32))
    for b, A in zip(bodies, frames_all):
        q = np.round(A / 1e-4).astype(np.int16)
        meta["bodies"].append(dict(height=b["height"], stride=b["stride"], positions=put(q),
                                   colors=put(np.round(b["colors"] * 255).astype(np.uint8))))
    (OUT / "soma.bin").write_bytes(b"".join(blobs))
    (OUT / "soma.json").write_text(json.dumps(meta))
    print(f"soma: {len(bodies)} bodies x {FRAMES + 1} frames, {off / 1e6:.1f} MB")


# ---------------------------------------------------------------- Unitree G1
def g1_joint_targets(phase: float | None) -> dict[str, float]:
    """G1 joint angles (MuJoCo convention: + hip/ankle pitch swings the link backward, + knee flexes)."""
    if phase is None:
        th = {"left": 0.0, "right": 0.0}
        k = {"left": 0.0, "right": 0.0}
        arm = {"left": 0.0, "right": 0.0}
    else:
        s = 2 * math.pi * phase
        th = {"left": 0.32 * math.sin(s), "right": 0.32 * math.sin(s + math.pi)}
        knee = lambda u: 0.75 * max(0.0, math.cos(u + 0.35)) ** 2
        k = {"left": knee(s), "right": knee(s + math.pi)}
        arm = {"left": 0.3 * math.sin(s + math.pi), "right": 0.3 * math.sin(s)}
    q = {}
    for side in ("left", "right"):
        crouch = 0.28                                             # the G1 walks with bent knees
        hip = -th[side] - crouch * 0.5
        knee_a = crouch + k[side]
        q[f"{side}_hip_pitch_joint"] = hip
        q[f"{side}_knee_joint"] = knee_a
        q[f"{side}_ankle_pitch_joint"] = -(hip + knee_a) * 0.9
        q[f"{side}_shoulder_pitch_joint"] = arm[side] + 0.15
        q[f"{side}_shoulder_roll_joint"] = 0.18 if side == "left" else -0.18
        q[f"{side}_elbow_joint"] = 0.55
    return q


def build_g1():
    import mujoco
    import trimesh
    import fast_simplification

    xml = SRC / "g1" / "g1.xml"
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    # visual geoms only (group 2 in the menagerie class "visual")
    geoms = [g for g in range(model.ngeom) if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and model.geom_group[g] == 2]
    mesh_ids = sorted({int(model.geom_dataid[g]) for g in geoms})

    blobs, off = [], 0

    def put(arr):
        nonlocal off
        buf = np.ascontiguousarray(arr).tobytes()
        pad = (-len(buf)) % 4
        blobs.append(buf + b"\0" * pad)
        start = off
        off += len(buf) + pad
        return [start, len(buf)]

    meshes = {}
    tris_total = 0
    for m in mesh_ids:
        va, vn = model.mesh_vertadr[m], model.mesh_vertnum[m]
        fa, fn = model.mesh_faceadr[m], model.mesh_facenum[m]
        V = model.mesh_vert[va:va + vn].astype(np.float32)
        F = model.mesh_face[fa:fa + fn].astype(np.int32)
        tm = trimesh.Trimesh(V, F, process=True)
        if len(tm.faces) > 2500:
            V2, F2 = fast_simplification.simplify(np.asarray(tm.vertices, np.float32), np.asarray(tm.faces, np.int32),
                                                  target_reduction=1 - 2500 / len(tm.faces))
            tm = trimesh.Trimesh(V2, F2, process=True)
        tris_total += len(tm.faces)
        # MuJoCo (x fwd, y left, z up) -> actor frame (x fwd, y up, z right)
        Vt = np.stack([tm.vertices[:, 0], tm.vertices[:, 2], -tm.vertices[:, 1]], axis=1).astype(np.float32)
        meshes[m] = dict(name=mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, m),
                         positions=put(Vt), faces=put(np.asarray(tm.faces, np.uint32)))

    phases = [None] + [k / FRAMES for k in range(FRAMES)]
    frames = []
    for p in phases:
        mujoco.mj_resetData(model, data)
        for name, val in g1_joint_targets(p).items():
            data.qpos[model.jnt_qposadr[model.joint(name).id]] = val
        mujoco.mj_kinematics(model, data)
        xpos = data.geom_xpos[geoms].copy()
        xmat = data.geom_xmat[geoms].reshape(-1, 3, 3).copy()
        # ground the lowest point of the feet (vertex level) at z = 0
        zmin = math.inf
        for gi, g in enumerate(geoms):
            m = int(model.geom_dataid[g])
            va, vn = model.mesh_vertadr[m], model.mesh_vertnum[m]
            zmin = min(zmin, float((model.mesh_vert[va:va + vn] @ xmat[gi].T + xpos[gi])[:, 2].min()))
        xpos[:, 2] -= zmin
        pel = data.xpos[model.body("pelvis").id]
        xpos[:, 0] -= pel[0]
        xpos[:, 1] -= pel[1]
        rows = []
        for gi in range(len(geoms)):
            # convert pose into actor frame: C R C^T with C mapping (x,y,z)_mj -> (x, z, -y)
            C = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)
            R = C @ xmat[gi] @ C.T
            t = C @ xpos[gi]
            M = np.eye(4)
            M[:3, :3] = R
            M[:3, 3] = t
            rows.append(M.T.flatten())                               # column-major for THREE.Matrix4.fromArray
        frames.append(np.stack(rows))
    rgba = [(model.mat_rgba[model.geom_matid[g]] if model.geom_matid[g] >= 0 else model.geom_rgba[g]).tolist() for g in geoms]
    # stride (distance per gait cycle): two steps of a 0.62 m leg swinging +-0.32 rad
    meta = dict(frames=FRAMES, standing_frame=0, cycle_frames=[1, FRAMES], stride=round(4 * 0.62 * math.sin(0.32) * 0.95, 3),
                meshes={str(k): v for k, v in meshes.items()},
                geoms=[dict(mesh=int(model.geom_dataid[g]), rgba=[round(c, 3) for c in rgba[i]]) for i, g in enumerate(geoms)],
                matrices=put(np.stack(frames).astype(np.float32)))
    (OUT / "g1.bin").write_bytes(b"".join(blobs))
    (OUT / "g1.json").write_text(json.dumps(meta))
    print(f"g1: {len(geoms)} visual geoms, {tris_total} triangles, {off / 1e6:.1f} MB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", default=None, help="write a contact sheet of SOMA frames to this PNG")
    ap.add_argument("--only", choices=["soma", "g1"], default=None)
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    if a.only in (None, "soma"):
        build_soma(a.check)
    if a.only in (None, "g1"):
        build_g1()
