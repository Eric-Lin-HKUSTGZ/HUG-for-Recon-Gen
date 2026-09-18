# DexYCB Native Geometry Overlay

## Contract

Keep the existing full-resolution PKLs, encoded RGB/Depth and detector/RTMPose cache. Store corrected parameters and geometry in a separate indexed directory. The schema is `dexycb_native_overlay_v1`; training accepts it only after the manifest reports a completed and validated build.

The builder reads official camera-frame labels and source-side MANO beta/PCA. It stores the official wrist as translation, converts rotations directly to the existing 6D layout, and reflects source-left labels once to match the already mirrored PKL image. No second image flip occurs. Official joints are preserved as independent targets. Meshes are decoded using the corresponding native hand model, then reflected into the image's canonical frame.

Parameters retain source-side beta. A fixed 10-by-10 beta conversion was checked previously and left a 0.543613 relative shape-basis residual; it is not equivalent. The opt-in `native_side_v1` decoder therefore uses known source handedness to select the correct MANO geometry and fingertip definition. This is metadata already needed for image canonicalization, not a new sensor or condition token. It is a deliberate extension beyond GPGFormer's right-MANO mesh fallback so that parameter and mesh supervision agree. A deployment pipeline must retain the side used to flip the input; the decoder fails rather than silently assuming right. This does not prove detection-side accuracy in unlabeled images.

The default `legacy_right` path and existing v29 configurations remain unchanged. New v30 RGB-only/RGB-D configurations use the overlay and its train-only normalization. They retain the existing split/augmentation/training budget. Test data still participates in model selection, as in v29; this change does not fix that evaluation-protocol choice. New experiments must not resume the optimizer or best score from an old geometry convention.

## Storage and Loader

Arrays: sorted sample names, 109D parameters, absolute axis-angles, official joints, decoded vertices, intrinsics, source side and source-label hashes. Arrays are `.npy` files opened read-only with memory mapping. Roughly 5 GB is expected for 492823 frames, dominated by meshes; no full PKL or image duplication is required.

The loader checks the source dataset, sample coverage, hand-side and camera conventions before overriding only geometric fields in a newly loaded record. It rebuilds 2D projections. The old geometry-derived mask is removed, and detector cropping is required; an old mask must not become a hidden GT condition. Existing detected boxes/keypoints retain their coordinates because the image, K and crop settings do not change. Affine augmentation with old caches remains outside this change and stays disabled in both new configurations.

## Builder and Verification

`scripts/run_dexycb_geometry_overlay.sh` first runs numerical/gradient and legacy regression tests, then a 512-per-split build, then the full build. It does not start training. The builder compares every decoded joint to official native GT (maximum allowed point error 0.05 mm) and checks original 2D projection consistency (0.01 px). Failure stops the build; incomplete overlays are rejected by the loader.

Only training frames contribute to normalization. Builds have an exclusive lock, immutable selection/code/asset manifest, per-chunk progress and input-label hashes. Resume revalidates already completed source labels. Finished outputs have array SHA256 hashes; rerunning the builder verifies them before returning. Original PKLs remain untouched.

## Current Execution Status

The implementation is deployed in `/root/code/HUG-for-Recon-Gen`. Existing dirty-worktree changes were preserved. The completed overlay is `/root/code/vepfs/dataset/hand_recon_hug/dexycb_native_overlay_v1` (4.9 GB): 492823 total frames, including 394193 training frames used for normalization. A second builder invocation verified all output hashes and returned without modifying the result.

All frames passed native MANO-to-official-joint validation. Maximum point error was 0.000266665 mm and maximum official 2D/3D projection error was 0.000080521 px. The 1536-frame smoke overlay also completed. Loader validation used real left/right PKLs and the existing detector cache; a two-worker batch returned 109D parameters, `(21,3)` official joints and `(778,3)` native meshes.

Remote verification passed: 8 overlay tests, 5 native MANO/gradient/animation tests, 5 geometry-repair regression tests and 5 original-loss regression tests. Both v30 config files parse, the RGB-only model constructs, and 116 compatible tensors load from the released pretrained checkpoint. Python compilation and `git diff --check` pass. No training job was launched.
