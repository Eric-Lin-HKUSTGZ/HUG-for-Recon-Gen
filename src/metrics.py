"""Hand reconstruction metrics: MPJPE / PA-MPJPE / MPVPE / PA-MPVPE.

PA (Procrustes alignment) follows the batched similarity-transform solve used
in GPGFormer (gpgformer/metrics/pose_metrics.py, WiLoR lineage): float64 SVD
with improper-rotation fix. Inputs are meters (camera frame); outputs are mm.
"""

import torch


def compute_similarity_transform(S1: torch.Tensor, S2: torch.Tensor) -> torch.Tensor:
    """Batched orthogonal Procrustes: rigid+scale align S1 (B,N,3) onto S2."""
    B = S1.shape[0]
    S1 = S1.to(torch.float64).permute(0, 2, 1)  # (B,3,N)
    S2 = S2.to(torch.float64).permute(0, 2, 1)

    mu1, mu2 = S1.mean(dim=2, keepdim=True), S2.mean(dim=2, keepdim=True)
    X1, X2 = S1 - mu1, S2 - mu2

    var1 = torch.clamp((X1**2).sum(dim=(1, 2)), min=1e-8)
    K = torch.matmul(X1, X2.permute(0, 2, 1))  # (B,3,3)

    # cuSOLVER can fail to initialize under heavy CUDA allocator pressure
    # (observed as CUSOLVER_STATUS_INTERNAL_ERROR on cusolverDnCreate during
    # val after 1k training steps). The (B,3,3) SVD is tiny - run it on CPU.
    U, _, Vh = torch.linalg.svd(K.cpu())
    U, Vh = U.to(K.device), Vh.to(K.device)
    V = Vh.permute(0, 2, 1)

    Z = torch.eye(3, device=S1.device, dtype=torch.float64).unsqueeze(0).repeat(B, 1, 1)
    Z[:, -1, -1] *= torch.sign(torch.det(torch.matmul(U, Vh)))
    R = torch.matmul(torch.matmul(V, Z), U.permute(0, 2, 1))

    trace = torch.matmul(R, K).diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    scale = (trace / var1).view(B, 1, 1)
    t = mu2 - scale * torch.matmul(R, mu1)

    S1_hat = scale * torch.matmul(R, S1) + t
    return S1_hat.permute(0, 2, 1)


def _per_point_error(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Mean L2 over points per sample. (B,N,3) -> (B,) mm."""
    return (
        torch.sqrt(torch.clamp(((pred - gt) ** 2).sum(dim=-1), min=1e-12))
        .mean(dim=-1)
        .float()
        * 1000.0
    )


def joint_mesh_errors(
    pred_joints: torch.Tensor,
    gt_joints: torch.Tensor,
    pred_verts: torch.Tensor,
    gt_verts: torch.Tensor,
) -> dict:
    """Per-sample MPJPE / PA-MPJPE / MPVPE / PA-MPVPE, in mm.

    Returns {name: (B,) tensor}. PA variants align pred onto GT (similarity
    transform fitted on the same point set being measured: joints for MPJPE,
    verts for MPVPE).
    """
    out = {
        "mpjpe": _per_point_error(pred_joints, gt_joints),
        "mpvpe": _per_point_error(pred_verts, gt_verts),
        "pa_mpjpe": _per_point_error(
            compute_similarity_transform(pred_joints, gt_joints), gt_joints
        ),
        "pa_mpvpe": _per_point_error(
            compute_similarity_transform(pred_verts, gt_verts), gt_verts
        ),
    }
    return out
