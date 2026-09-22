"""CPU integration check for HO3D native train/eval PKLs and cached conditions.

Without --cache, creates a clearly marked all-detector-miss TEST FIXTURE.
With --cache, checks the real cached conditions without running any detector.
No image encoder, training optimizer, or GPU is loaded.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.prepare_ho3d_v2 import read_lines, load_record, project, write_json, sha
from src.cache_train_conditions import arrays
from src.dataloader.grasp_dataset import GraspDataset
from src.models.grasp_model import GraspFlowModel
from src.models.native_mano import (
    NativeCanonicalMANO, apply_joint_convention, geometry_kwargs_from_batch,
    prediction_geometry_kwargs_from_batch,
)
from src.train import _make_dataset


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--samples-file", type=Path, required=True)
    p.add_argument("--cache", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--samples", type=int, default=32)
    p.add_argument("--augmentation-config", type=Path)
    a = p.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(20260919)
    np.random.seed(20260919)
    a.output.mkdir(parents=True, exist_ok=True)
    stems = read_lines(a.samples_file)
    sample_ids = sorted(np.random.default_rng(20260919).choice(len(stems), min(a.samples, len(stems)), replace=False).tolist())
    train = load_record(a.dataset_root, stems[0])["grasp"] is not None
    if a.cache is None:
        cache = arrays(stems, 0)
        # Used ONLY to exercise the detector failure path, never for training.
        for i, stem in enumerate(stems):
            d = load_record(a.dataset_root, stem)
            w, h = d["camera"]["width"], d["camera"]["height"]
            cache["image_size_wh"][i] = [w, h]
            cache["crop_bbox_xyxy"][i] = [0, 0, w-1, h-1]
        cache_path = a.output / "TEST_FIXTURE_all_detector_misses.npz"
        np.savez_compressed(cache_path, **cache)
    else:
        cache_path = a.cache
        meta = json.loads((cache_path.parent / "metadata.json").read_text())
        assert Path(meta["dataset_root"]).resolve() == a.dataset_root.resolve()
        with np.load(cache_path, allow_pickle=False) as z:
            assert len(z["sample"]) == len(set(z["sample"].tolist()))
            assert set(stems).issubset(set(z["sample"].tolist()))
    crop = dict(enabled=True, keypoint_source="rtmpose_cache", train_keypoint_source="rtmpose_cache",
                eval_keypoint_source="rtmpose_cache", condition_cache=str(cache_path), skeleton_drop_prob=0.)
    data = OmegaConf.create(dict(n_points_input=256, geometry_overlay="invalid_global_dexycb_overlay",
                                hand_crop=dict(enabled=True, keypoint_source="gt"), augmentation=dict(enabled=False)))
    model_cfg = OmegaConf.create(dict(d_mano=109, image_size=224, use_rgb=True, use_depth=True))
    if train and a.augmentation_config:
        data.augmentation = OmegaConf.load(a.augmentation_config).trainer.data.augmentation
    # Verify per-dataset overrides: the global DexYCB overlay must not be opened.
    entry = OmegaConf.create(dict(geometry_overlay=None, hand_crop=crop))
    ds = _make_dataset(data, model_cfg, a.dataset_root, "train" if train else "val",
                       samples_filename=a.samples_file, indices=sample_ids, dataset_entry=entry)
    assert ds.geometry_overlay is None and ds.keypoint_source == "rtmpose_cache"
    assert data.geometry_overlay == "invalid_global_dexycb_overlay" and data.hand_crop.keypoint_source == "gt"
    model = GraspFlowModel.__new__(GraspFlowModel)
    torch.nn.Module.__init__(model)
    model.mano_geometry = "native_side_v1"
    model.mano = NativeCanonicalMANO("/root/code/vepfs/GPGFormer/weights/mano/MANO_LEFT.pkl")
    max_joint_mm = max_vertex_mm = 0.
    n = 0
    for batch in DataLoader(ds, batch_size=8, num_workers=0):
        assert (batch["source_is_left"] == False).all()
        assert (batch["joint_convention_id"] == 1).all()
        for k, v in batch.items():
            if torch.is_tensor(v) and v.is_floating_point():
                assert torch.isfinite(v).all(), k
        if train:
            x = batch["mano_params"].clone().requires_grad_()
            pred, target = model._build_dicts(x, x, x.detach(), x.detach(),
                                            **geometry_kwargs_from_batch(batch, "cpu"))
            jerr = (pred["landmarks_3d"] - target["landmarks_3d"]).norm(dim=-1).max().item()*1000
            verr = (pred["vertices"] - target["vertices"]).norm(dim=-1).max().item()*1000
            max_joint_mm, max_vertex_mm = max(max_joint_mm,jerr), max(max_vertex_mm,verr)
            assert jerr < .01 and verr < .01, (jerr,verr)
            for j in range(len(x)):
                xy = project(batch["gt_joints_3d"][j].numpy(), batch["camera_K"][j].numpy())
                np.testing.assert_allclose(xy, batch["landmarks_2d"][j].numpy(), atol=.002, rtol=0)
            pred["landmarks_3d"].square().sum().backward()
            assert torch.isfinite(x.grad).all() and (x.grad[:,99:].abs().sum(1)>0).all()
        else:
            # Arbitrary pose is enough to verify native eval decoding needs no beta GT.
            assert "mano_params" not in batch and "mano_shape" not in batch
            b = len(batch["source_is_left"])
            r6 = torch.eye(3)[:,:2].reshape(1,6).repeat(b,16)
            x = torch.cat([torch.tensor([[0.,0.,.6]]).repeat(b,1),r6,torch.zeros(b,10)],1)
            pred = model.mano_forward(x, **prediction_geometry_kwargs_from_batch(batch,"cpu"))
            torch.testing.assert_close(pred["landmarks_3d"][:,[4,8,12,16,20]],pred["vertices"][:,[744,333,444,555,672]])
            assert batch["joints_gt"].shape == (b,21,3) and batch["verts_gt"].shape == (b,778,3)
        n += len(batch["source_is_left"])
    # Mixed-dataset convention selection must preserve DexYCB and mesh gradients.
    j = torch.randn(2,21,3,requires_grad=True)
    v = torch.randn(2,778,3,requires_grad=True)
    mapped = apply_joint_convention(j,v,torch.tensor([0,1]))
    torch.testing.assert_close(mapped[0],j[0])
    torch.testing.assert_close(mapped[1,[4,8,12,16,20]],v[1,[744,333,444,555,672]])
    mapped.sum().backward()
    assert v.grad[1].abs().sum()>0 and v.grad[0].abs().sum()==0
    report = dict(status="passed",checked=n,mode="train" if train else "evaluation",
                  real_condition_cache=a.cache is not None,cache=str(cache_path),
                  augmentation=bool(train and a.augmentation_config),
                  max_joint_roundtrip_mm=max_joint_mm if train else None,
                  max_vertex_roundtrip_mm=max_vertex_mm if train else None,
                  samples_sha256=sha(a.samples_file))
    write_json(a.output/"loader_check_report.json",report)
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
