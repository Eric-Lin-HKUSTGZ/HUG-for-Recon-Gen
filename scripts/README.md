# scripts/ — 数据转换与预处理脚本

本目录包含将手部重建数据集（HO3D、DexYCB）转换为 **HUG pkl schema** 的脚本，以及配套的数据集划分与归一化统计工具。转换产物由 `src/dataloader/grasp_dataset.py` 直接消费。

详细背景与格式验证结论见 `analysis_docs/HUG_hand_recon_data_adaptation.md`。

## 统一约定（所有转换脚本遵循）

- **坐标系**：相机坐标系，OpenCV 约定（z 轴朝前为正），单位为米。
- **99D 抓取状态**：`t(3) + R_6d(6) + pose_6d(15×6)`，旋转使用连续 6D 表示。
- **GT landmarks / mesh**：用 HUG 标准 MANO shape β（`assets/mano_rhand_shape.npy`）前向生成，与 `GraspFlowModel._build_dicts` 中 pred 和 GT 均使用 fixed_betas 的约定一致（adaptation 文档 §4.3 option A）。
- **图像**：中心裁剪成正方形 → resize 到 224×224，相机内参 K 同步调整（与 `src/prepare_inputs.py` 相同）；深度以 uint16 PNG 存储，单位 1mm。

## 文件一览

| 文件 | 作用 |
|---|---|
| `conversion_common.py` | 共享工具库：旋转转换、MANO 前向、图像打包、pkl 写出 |
| `convert_ho3d.py` | HO3D_v3 → HUG pkl（训练集 + 评估集两种 schema） |
| `convert_dexycb.py` | DexYCB → HUG pkl |
| `compute_norm_stats.py` | 统计转换后数据集的 99D 状态均值/方差，生成 norm_stats JSON |
| `make_val_split.py` | 为 1M-HUGS 生成录制级（recording-level）验证集划分 |
| `nohup.out` | 一次 DexYCB 全量转换（508384 帧）的运行日志 |

## 各脚本说明

### `conversion_common.py`（库，不直接运行）

供两个转换脚本共用的核心模块：

- **旋转工具**：`aa_to_rotmat`（轴角→旋转矩阵）、`rotmat_to_6d`（矩阵→6D）、`pose48_to_99d`（由相机系 t、全局旋转 R、15 关节轴角组装 99D 状态）。
- **MANO 前向**：`get_mano()` 进程级单例（多进程 worker 安全），`mano_forward_canonical()` 用标准 β 生成 21 个 3D 关节点与 778 顶点网格。
- **图像处理**：`center_crop_square` / `adjust_K` / `project_uv` / `make_hand_mask`（投影网格顶点凸包 + 膨胀得到手部 mask）。
- **写出**：`FrameSample` 数据类 + `write_sample()`，完成裁剪、resize、MANO 前向、JPEG/PNG 编码并原子写入 pkl（先写 `.tmp.pkl` 再 rename）。**不写 `condition_point`**，让 `GraspDataset` 训练时从手部 mask 随机采样 query 点。

### `convert_ho3d.py`

将 HO3D_v3 转换为 HUG pkl。要点（均已实证验证）：

- 坐标转换：`(x,y,z) → (x,-y,-z)`，全局朝向 `Rconv @ R`（`Rconv = diag(1,-1,-1)`）。
- 深度图解码：3 通道 uint8 编码 16 位值，`depth_m = (G*256 + R) * 0.00012498664727900177`。
- `handJoints3D` 为空的帧会被跳过。
- **`--split train`（默认）**：读取 `train.txt`，MANO 标注完整，输出训练 schema。
- **`--split evaluation`**：评估集 meta 中**没有** MANO 参数，官方 GT 在 `evaluation_xyz.json` / `evaluation_verts.json` 中。因此输出评估 schema（`grasp=None`），附带 `joints_gt` / `verts_gt` 用于计算指标，并将 `condition_point` 设为手腕投影点以便直接推理。

```bash
python scripts/convert_ho3d.py --out-dir .../ho3d [--max-samples 200] [--workers 16]
python scripts/convert_ho3d.py --split evaluation --out-dir .../ho3d_eval
```

数据集路径硬编码为 `HO3D_ROOT = /root/code/vepfs/dataset/HO3D_v3`，使用前请按需修改。

### `visualize_dataloader.py`

对转换后的 pkl 走真实 `GraspDataset` 加载路径，生成 4 联图，检查 RGB、GT
骨架、mask、query 点、深度图和 PCL 是否对齐。每个样本同时打印关节出画比例、
query 深度与腕部深度差、以及 PCL 中靠近腕部的点比例。

```bash
# DexYCB train / val / test
python scripts/visualize_dataloader.py --dataset dexycb --split train --n 16 \
    --data-path /root/code/vepfs/dataset/hand_recon_hug/dexycb_v2_canonical_right \
    --split-dir /root/code/vepfs/dataset/hand_recon_hug/splits_v2
python scripts/visualize_dataloader.py --dataset dexycb --split test --n 16 \
    --out /root/code/vepfs/HUG-for-Recon-Gen/viz/dexycb_test --seed 7

# HO3D train / val / official evaluation
python scripts/visualize_dataloader.py --dataset ho3d --split train --n 16
python scripts/visualize_dataloader.py --dataset ho3d_eval --split eval --n 16

# 用清单索引精确复现某一帧
python scripts/visualize_dataloader.py --dataset dexycb --split train \
    --indices 0,10,100 --out /root/code/vepfs/HUG-for-Recon-Gen/viz/debug
```

输出包括单样本 PNG 和 `<dataset>_<split>_grid.png` 总览图。注意训练数据的
query 点本来是 mask 内随机采样，因此可视化日志中 query 深度与腕部深度存在
差异时，应结合 RGB+mask+query、depth+mask+query 和 RGB+PCL 三个面板判断，
不能只看单个数值。

### `convert_dexycb.py`

将 DexYCB 转换为 HUG pkl。要点：

- `pose.npz` 的 `pose_m (N,1,51)` = 全局轴角（3) + 45 维 PCA 系数 + 平移（3)，位于**世界系**（8 个相机共享）；全零帧无标注，跳过。
- PCA → 轴角：`aa45 = hands_mean + pca45 @ hands_components`，按 `meta.yml` 的
  `mano_sides[0]` 选择对应的 `MANO_LEFT/RIGHT.pkl`（不能从 `mano_calib` 名称后缀判断）。
- HUG 模型是右手模型，因此左手序列在转换时 canonicalize 到右手：RGB/深度水平翻转、
  `cx = W - 1 - cx`，相机系 3D 坐标 x 取反，全局和 15 个局部旋转做 `F @ R @ F`。
  新 pkl 写入 `source_mano_side`/`canonical_mano_side`/`canonicalization` provenance。
- 外参存的是 camera→world（3×4），使用时取逆；β 按受试者从 calibration 读取。
  左手文件路径：`/root/code/vepfs/GPGFormer/weights/mano/MANO_LEFT.pkl`。
- 深度为 uint16 毫米，与 HUG 原生格式一致，无需重编码。
- 每个序列 × 每帧 × 每个相机视角生成一个 pkl。

```bash
python scripts/convert_dexycb.py \
    --out-dir /root/code/vepfs/dataset/hand_recon_hug/dexycb_v2_canonical_right \
    [--max-seqs 2] [--max-frames-per-seq N] [--workers 16]
```

数据集路径硬编码为 `DEXYCB_ROOT = /root/code/vepfs/dataset/dex-ycb`。全量转换约 1000 个序列、50.8 万帧（见 `nohup.out`），建议用 `nohup` 后台运行。

### `validate_dexycb_conversion.py`

全量重转换前先用少量左右手样本做 parity audit：

```bash
python scripts/validate_dexycb_conversion.py \
    --converted-dir /tmp/dex_side_smoke --n 80
```

该脚本使用官方 per-camera `labels_*.npz` 检查左右手数量、pkl metadata、
转换后 2D 投影内部一致性以及 canonical beta 相对官方 2D GT 的残差。另用
`src.models.mano.MANO` 将 pkl 99D 恢复到 canonical 右手空间，与左手官方
`joint_3d` 经 `x -> -x` 后比较；小规模 fixture 已验证左手平均约 6.3mm、
右手约 2.4mm。

### `compute_norm_stats.py`

从转换后的 pkl 中读取 grasp 字典（`t + R_6d + pose_6d` 拼成 99D），用流式算法（Chan et al.）计算均值/标准差，输出与 `assets/norm_stats.json` 相同的 `{translation, wrist_rot, finger_rot: {mean, std}}` 布局，供训练配置中的归一化使用。支持多目录合并与随机子采样。

```bash
python scripts/compute_norm_stats.py \
    --data-dirs /root/code/vepfs/dataset/hand_recon_hug/ho3d \
                /root/code/vepfs/dataset/hand_recon_hug/dexycb_v2_canonical_right \
    --split-files /root/code/vepfs/dataset/hand_recon_hug/splits_v2/ho3d_train.clean.txt \
                  /root/code/vepfs/dataset/hand_recon_hug/splits_v2/dexycb_train.clean.txt \
    --out assets/norm_stats_handrecon_v2.json \
    [--max-samples-per-set 100000] [--workers 16] [--recompute]
```

### `make_val_split.py`

为 1M-HUGS 训练数据生成**录制级**验证集划分。之所以按录制划分：每次物理抓取会产生数百个 (frame, grasp) 样本（含同时间戳的 `_grayscale` 孪生帧），随机按帧划分会造成数据泄漏。按录制（stem 去掉 `<frame>_<hash>[_grayscale]` 后缀）分组，保证同一次抓取的所有帧落在同一侧。

```bash
python scripts/make_val_split.py \
    --dataset-path /root/code/vepfs/dataset/1m-hugs/grasp_data \
    --n-recordings 48 [--seed 42]
```

输出 `{dataset_path}/split_val.txt`（每行一个 stem），通过 `GraspDataset(samples_filename="split_val.txt")` 使用。需先构建 `samples.txt` 索引。

## 典型流程

```bash
# 1. 转换数据集
python scripts/convert_ho3d.py   --out-dir .../hand_recon_hug/ho3d --workers 16
python scripts/convert_dexycb.py --out-dir .../hand_recon_hug/dexycb --workers 16
python scripts/convert_ho3d.py   --split evaluation --out-dir .../hand_recon_hug/ho3d_eval

# 2. 计算归一化统计
python scripts/compute_norm_stats.py --data-dirs .../ho3d .../dexycb \
    --out assets/norm_stats_handrecon.json

# 3.（1M-HUGS）生成验证集划分
python scripts/make_val_split.py --dataset-path .../1m-hugs/grasp_data --n-recordings 48
```

## 依赖

`numpy`、`opencv-python`（cv2）、`tyro`（CLI）、`rich`（日志）、`pyyaml`（DexYCB 标定）、`torch` + `src.models.mano`（MANO 前向，仅训练样本转换需要）。所有脚本支持 `ProcessPoolExecutor` 多进程加速（`--workers`）。
