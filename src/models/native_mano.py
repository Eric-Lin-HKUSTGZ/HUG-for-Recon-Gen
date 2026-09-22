"""Decode source-side beta/pose in the canonical (mirrored-image) frame.

Image canonicalization does not make the native left shape basis identical
to MANO_RIGHT. Keep source handedness as geometry metadata, not a condition
token. Assets remain unchanged on disk. Legacy MANO is unaffected.
"""

import copy
import pickle
from pathlib import Path

import numpy as np
import torch

from .mano import MANO
from ..utils.transform_utils import six_d_to_rotation_matrix


class NativeCanonicalMANO(MANO):
    def __init__(self, left_asset):
        super().__init__()
        self.left_layer = copy.deepcopy(self.mano_layer)
        self.left_layer.side = "left"
        with Path(left_asset).open("rb") as f:
            data = pickle.load(f, encoding="latin1")
        for name, key in (
            ("th_shapedirs", "shapedirs"), ("th_posedirs", "posedirs"),
            ("th_v_template", "v_template"), ("th_J_regressor", "J_regressor"),
            ("th_weights", "weights"), ("th_faces", "f"),
        ):
            value = data[key]
            if hasattr(value, "toarray"):
                value = value.toarray()
            value = torch.as_tensor(np.asarray(value).copy())
            value = value.long() if key == "f" else value.float()
            if key == "v_template":
                value = value.unsqueeze(0)
            if value.shape != getattr(self.left_layer, name).shape:
                raise ValueError(f"incompatible left MANO buffer: {key}")
            setattr(self.left_layer, name, value)
        parents = list(np.asarray(data["kintree_table"])[0].tolist())
        if parents != self.mano_layer.kintree_parents:
            raise ValueError("left and right MANO kinematic trees differ")

    def faces_for_side(self, source_is_left):
        if source_is_left:
            return self.left_layer.th_faces[:, [0, 2, 1]]
        return self.mano_layer.th_faces

    def forward(self, mano_params, betas=None, source_is_left=None):
        if source_is_left is None:
            raise ValueError("native_side_v1 requires source_is_left from image canonicalization")
        mask = torch.as_tensor(source_is_left, device=mano_params.device)
        if mask.dtype != torch.bool or mask.shape != (mano_params.shape[0],):
            raise ValueError("source_is_left must be a bool tensor of shape (B,)")
        if betas is None:
            if mano_params.shape[-1] != 109:
                raise ValueError("native_side_v1 requires 109D source beta parameters")
            betas = mano_params[:, 99:109]

        # Decode rotation matrices directly: avoids a numerically unstable
        # matrix -> axis-angle -> matrix round trip near pi. MANO is FP32
        # even under training autocast, preserving millimeter-scale geometry.
        with torch.autocast(device_type=mano_params.device.type, enabled=False):
            x, beta = mano_params.float(), betas.float()
            rotations = six_d_to_rotation_matrix(x[:, 3:99].reshape(-1, 16, 6))
            reflection = x.new_tensor([-1, 1, 1])
            joints = x.new_zeros((len(x), 21, 3))
            vertices = x.new_zeros((len(x), 778, 3))
            for is_left, layer in ((False, self.mano_layer), (True, self.left_layer)):
                ids = torch.where(mask == is_left)[0]
                if ids.numel() == 0:
                    continue
                r = rotations[ids]
                if is_left:
                    r = r * reflection[None, None, :, None] * reflection[None, None, None, :]
                out = layer.skinning_layer(r, beta[ids])
                j, v = out["joints"], out["verts"]
                if is_left:
                    j, v = j * reflection, v * reflection
                joints = joints.index_copy(0, ids, j)
                vertices = vertices.index_copy(0, ids, v)
        return {"landmarks_3d": joints, "vertices": vertices, "t": x[:, :3],
                "R_3x3": rotations[:, 0], "pose_3x3": rotations[:, 1:]}


def geometry_kwargs_from_batch(batch, device):
    """New data has explicit geometry; old batches keep their old behavior."""
    keys = ("source_is_left", "gt_joints_3d", "gt_vertices")
    present = [k in batch for k in keys]
    if not any(present):
        return {}
    if not all(present):
        raise ValueError("incomplete native geometry batch")
    result = {k: batch[k].to(device, non_blocking=True) for k in keys}
    if "joint_convention_id" in batch:
        result["joint_convention_id"] = batch["joint_convention_id"].to(device, non_blocking=True)
    return result


def prediction_geometry_kwargs_from_batch(batch, device):
    """Geometry metadata for evaluation samples without MANO parameter GT."""
    return {k: batch[k].to(device, non_blocking=True)
            for k in ("source_is_left", "joint_convention_id") if k in batch}


def apply_joint_convention(joints, vertices, convention_id):
    """Match HO3D landmarks without changing meshes or DexYCB tip definitions."""
    if convention_id is None:
        return joints
    ids = torch.as_tensor(convention_id, device=joints.device)
    if ids.shape != (len(joints),) or ids.dtype != torch.long:
        raise ValueError("joint_convention_id must be an int64 tensor of shape (B,)")
    if torch.any((ids != 0) & (ids != 1)):
        raise ValueError("unsupported joint convention (0=manotorch, 1=HO3D official)")
    ho3d = joints.clone()
    ho3d[:, [4, 8, 12, 16, 20]] = vertices[:, [744, 333, 444, 555, 672]]
    return torch.where((ids == 1)[:, None, None], ho3d, joints)
