"""Read-only, memory-mapped geometry corrections; RGB/Depth remain in PKLs."""

import json
import os
from pathlib import Path

import numpy as np

VERSION = "dexycb_native_overlay_v1"
CONVENTION = "source_beta_canonical_pose_v1"
ARRAYS = {
    "params": ((109,), np.dtype("float32")),
    "pose_aa": ((48,), np.dtype("float32")),
    "joints": ((21, 3), np.dtype("float32")),
    "vertices": ((778, 3), np.dtype("float32")),
    "camera_K": ((3, 3), np.dtype("float32")),
    "source_is_left": ((), np.dtype("bool")),
}


class GeometryOverlay:
    def __init__(self, directory, dataset_root):
        self.directory = Path(directory).resolve()
        self.dataset_root = Path(dataset_root).resolve()
        self._pid = None
        self._arrays = None
        self._open()

    def __getstate__(self):
        state = dict(self.__dict__)
        state.update(_pid=None, _arrays=None)
        return state

    def _open(self):
        if self._pid == os.getpid() and self._arrays is not None:
            return
        manifest = json.loads((self.directory / "manifest.json").read_text())
        if (manifest.get("version") != VERSION or manifest.get("status") != "complete"
                or manifest.get("parameter_convention") != CONVENTION):
            raise ValueError("geometry overlay is incomplete or has an unsupported convention")
        if Path(manifest["dataset_root"]).resolve() != self.dataset_root:
            raise ValueError("geometry overlay belongs to a different PKL dataset")
        n = int(manifest["count"])
        arrays = {k: np.load(self.directory / f"{k}.npy", mmap_mode="r", allow_pickle=False)
                  for k in (*ARRAYS, "samples", "faces_left", "faces_right")}
        for k, (shape, dtype) in ARRAYS.items():
            if arrays[k].shape != (n, *shape) or arrays[k].dtype != dtype:
                raise ValueError(f"invalid overlay array: {k}")
        samples = arrays["samples"]
        if samples.shape != (n,) or samples.dtype.kind != "U" or n == 0:
            raise ValueError("invalid overlay sample index")
        if n > 1 and not np.all(samples[1:] > samples[:-1]):
            raise ValueError("overlay sample index must be sorted and unique")
        for k in ("faces_left", "faces_right"):
            if (arrays[k].ndim != 2 or arrays[k].shape[1] != 3
                    or arrays[k].dtype.kind not in "iu" or arrays[k].min() < 0
                    or arrays[k].max() >= 778):
                raise ValueError("invalid overlay mesh topology")
        self.manifest = manifest
        self._arrays, self._pid = arrays, os.getpid()

    def validate_samples(self, stems):
        self._open()
        samples = self._arrays["samples"]
        stems = np.asarray(stems, dtype=str)
        ids = np.searchsorted(samples, stems)
        if np.any(ids >= len(samples)) or not np.array_equal(samples[ids], stems):
            raise ValueError("geometry overlay does not cover the requested samples")

    def apply(self, stem, record):
        self._open()
        a = self._arrays
        idx = int(np.searchsorted(a["samples"], stem))
        if idx >= len(a["samples"]) or a["samples"][idx] != stem:
            raise KeyError(f"missing geometry correction: {stem}")
        left = bool(a["source_is_left"][idx])
        side = "left" if left else "right"
        if (record.get("source_mano_side") != side
                or record.get("canonical_mano_side") != "right"
                or record.get("canonicalization") != ("camera_x_reflection" if left else "none")):
            raise ValueError(f"source hand-side mismatch: {stem}")
        camera = record["camera"]
        if (camera.get("width") != 640 or camera.get("height") != 480
                or not np.allclose(camera["K"], a["camera_K"][idx], rtol=0, atol=1e-4)):
            raise ValueError(f"overlay requires matching full-resolution camera: {stem}")
        x, aa = a["params"][idx].copy(), a["pose_aa"][idx].copy()
        joints, vertices = a["joints"][idx].copy(), a["vertices"][idx].copy()
        if not all(np.isfinite(v).all() for v in (x, aa, joints, vertices)):
            raise ValueError(f"nonfinite geometry overlay: {stem}")
        if not np.allclose(x[:3], joints[0], atol=1e-7, rtol=0):
            raise ValueError(f"overlay wrist/parameter mismatch: {stem}")
        projection = joints @ np.asarray(camera["K"]).T
        if np.any(projection[:, 2] <= 0):
            raise ValueError(f"invalid overlay projection: {stem}")
        r = x[3:9].reshape(3, 2)
        transform = np.eye(4, dtype=np.float32)
        transform[:3, :3] = np.column_stack((r, np.cross(r[:, 0], r[:, 1])))
        transform[:3, 3] = x[:3]
        grasp = dict(record["grasp"])
        grasp.update(t=x[:3][None], R_6d=x[3:9][None], pose_6d=x[9:99].reshape(1, 15, 6),
                     pose=aa[3:].reshape(1, 15, 3), shape=x[99:][None].copy(),
                     shape_gt=x[99:][None].copy(), T_camera_wrist=transform,
                     landmarks_3d=joints, mesh_vertices=vertices,
                     landmarks_2d=(projection[:, :2] / projection[:, 2:]).astype(np.float32),
                     mesh_faces=a[f"faces_{side}"].copy())
        result = dict(record, grasp=grasp, schema_version="dexycb_native_geometry_v1",
                      mano_parameter_convention=CONVENTION,
                      geometry_overlay=str(self.directory))
        # The old mask was generated from incorrect geometry. New experiments
        # require detector crops and never sample a query from that old mask.
        result.pop("object_mask", None)
        result.pop("condition_point", None)
        return result
