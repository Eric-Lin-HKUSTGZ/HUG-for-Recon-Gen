"""Decompose MPJPE into translation / rotation / shape-error components.

Runs sample() on a val subset and reports:
  - full MPJPE / PA-MPJPE (mm)
  - root-relative MPJPE (wrist subtracted -> translation removed)
  - translation error ||t_pred - t_gt|| (mm), per-axis signed bias
  - wrist global-rotation geodesic error (deg)
  - 2D reprojection error of predictions (px, 224 image)
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataloader.grasp_dataset import GraspDataset
from src.metrics import joint_mesh_errors, compute_similarity_transform
from src.models.grasp_model import GraspFlowModel
from src.eval_test import load_weights

CKPT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "/root/code/vepfs/HUG-for-Recon-Gen/hand_recon/20260902_v6_canonical/model_best.pt"
)
CFG = Path("/root/code/vepfs/HUG-for-Recon-Gen/hand_recon/20260902_v6_canonical/config.yaml")
N_SAMPLES = int(sys.argv[2]) if len(sys.argv) > 2 else 512

device = torch.device("cuda")
cfg = OmegaConf.load(CFG)

loaded = torch.load(CKPT, map_location=device, weights_only=False)
norm_stats = loaded.get("norm_stats")
if norm_stats is None:
    norm_stats = json.load(open(cfg.trainer.data.norm_stats_file))
model = GraspFlowModel(cfg, norm_stats=norm_stats).to(device)
load_weights(model, loaded, "ema")
model.eval()

entry = cfg.trainer.val.datasets[0]  # dexycb_s0_val
ds = GraspDataset(
    str(entry.path),
    samples_filename=str(entry.samples),
    split="val",
    n_points_input=cfg.trainer.data.n_points_input,
    pcl_crop_radius=cfg.trainer.model.get("pcl_crop_radius", 0.3),
    use_rgb=True,
    use_depth=True,
)
ds.grasp_files = ds.grasp_files[:N_SAMPLES]
loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=8)

torch.manual_seed(0)

def rot_geodesic_deg(R_pred, R_gt):
    R = R_pred @ R_gt.transpose(-1, -2)
    cos = ((R.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)
    return torch.acos(cos) * 180 / np.pi

agg = {k: [] for k in (
    "mpjpe", "pa_mpjpe", "rel_mpjpe", "rel_rot_mpjpe",
    "t_err", "t_bias", "rot_deg", "l2d_px", "z_depth",
)}

with torch.no_grad():
    for batch in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            samples = model.sample(
                point_uv=batch["point_uv"].to(device),
                camera_K=batch["camera_K"].to(device),
                steps=50,
                rgb=batch["rgb"].to(device),
                pcl_xyz=batch["pcl_xyz"].to(device),
                pcl_rgb=batch["pcl_rgb"].to(device),
            )
        samples = samples.float()
        gt = batch["mano_params"].to(device).float()

        pred_out = model.mano_forward(samples)
        gt_out = model.mano_forward(gt)
        pj, gj = pred_out["landmarks_3d"], gt_out["landmarks_3d"]
        pv, gv = pred_out["vertices"], gt_out["vertices"]

        errs = joint_mesh_errors(pj, gj, pv, gv)
        agg["mpjpe"].append(errs["mpjpe"].cpu())
        agg["pa_mpjpe"].append(errs["pa_mpjpe"].cpu())

        # translation removed (root-relative)
        pj_rel = pj - pj[:, :1]
        gj_rel = gj - gj[:, :1]
        agg["rel_mpjpe"].append(
            torch.sqrt(((pj_rel - gj_rel) ** 2).sum(-1)).mean(-1).cpu() * 1000
        )
        # translation + wrist rotation removed (finger geometry only)
        Rp = pred_out["R_3x3"].transpose(-1, -2)
        gj_cent = gj - gj[:, :1]
        pj_unrot = (pj - pj[:, :1]) @ Rp  # un-rotate pred wrist frame
        agg["rel_rot_mpjpe"].append(
            torch.sqrt(((pj_unrot - gj_cent) ** 2).sum(-1)).mean(-1).cpu() * 1000
        )

        t_pred, t_gt = samples[:, :3], gt[:, :3]
        agg["t_err"].append(torch.linalg.norm(t_pred - t_gt, dim=-1).cpu() * 1000)
        agg["t_bias"].append((t_pred - t_gt).cpu() * 1000)
        agg["rot_deg"].append(rot_geodesic_deg(pred_out["R_3x3"], gt_out["R_3x3"]).cpu())
        agg["z_depth"].append(t_gt[:, 2].cpu() * 1000)

        K = batch["camera_K"].to(device).float()
        uv = pj @ K.transpose(-1, -2)
        uv = uv[..., :2] / (uv[..., 2:3] + 1e-6)
        l2d = (uv - batch["landmarks_2d"].to(device).float()).abs().mean(dim=(1, 2))
        agg["l2d_px"].append(l2d.cpu())

def cat(k):
    return torch.cat(agg[k])

mpjpe = cat("mpjpe").mean().item()
pa = cat("pa_mpjpe").mean().item()
rel = cat("rel_mpjpe").mean().item()
relrot = cat("rel_rot_mpjpe").mean().item()
terr = cat("t_err")
rot = cat("rot_deg")
tbias = cat("t_bias")
l2d = cat("l2d_px").mean().item()

print(f"\n=== Global-error decomposition (n={len(cat('mpjpe'))}, ckpt step={loaded.get('step')}) ===")
print(f"MPJPE               {mpjpe:8.2f} mm")
print(f"PA-MPJPE            {pa:8.2f} mm   (similarity-aligned)")
print(f"root-relative MPJPE {rel:8.2f} mm   (translation removed)")
print(f"finger-only MPJPE   {relrot:8.2f} mm   (translation + wrist rotation removed)")
print(f"|t_pred - t_gt|     {terr.mean():8.2f} mm   (p50 {terr.median():.2f}, p90 {terr.quantile(0.9):.2f})")
print(f"t signed bias (x,y,z) {tbias.mean(0).numpy().round(2)} mm, std {tbias.std(0).numpy().round(2)}")
print(f"wrist rot error     {rot.mean():8.2f} deg  (p50 {rot.median():.2f}, p90 {rot.quantile(0.9):.2f})")
print(f"2D reprojection     {l2d:8.2f} px")
z = cat("z_depth")
print(f"t_err vs depth corr {np.corrcoef(z.numpy(), terr.numpy())[0,1]:.3f}  (depth mean {z.mean():.0f} mm)")
