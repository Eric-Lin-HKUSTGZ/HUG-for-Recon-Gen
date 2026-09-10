# DexYCB / HO3D_v3 适配 HUG 手部重建训练方案

> 目标：在 HUG 架构（RGB-D + 点条件 → flow matching → 99D MANO）上训练**手部重建**任务，
> 数据集为 DexYCB（`/root/code/vepfs/dataset/dex-ycb`）和 HO3D_v3（`/root/code/vepfs/dataset/HO3D_v3`）。
> 参考实现：GPGFormer 的 `data/dex_ycb_dataset.py`、`data/ho3d_dataset.py`（RGB-only，已验证正确）。
> 本文所有数据格式结论均经过实际读取验证（含可视化验证）。

## 0. 任务差异分析：HUG 抓取生成 vs 手部重建

| 维度 | HUG 原任务（抓取生成） | 手部重建（本任务） |
|---|---|---|
| 图像中的手 | **无手**（采集时过滤掉有手帧，抓取标签从后续帧回传） | **有手**（恢复画面中这只手） |
| mask 含义 | 物体 mask（SAM3 传播） | 手部 mask（需自行生成，见 §3.4） |
| query 点 | 用户点击物体上一点 | 手上一点（建议：可见手部关键点投影，见 §4.2） |
| GT 来源 | MANO 拟合 Aria 跟踪 | 数据集官方 MANO 标注（更准） |
| β 形状 | 全数据固定 canonical | DexYCB 按 subject、HO3D 按帧变化（见 §4.3 决策） |

核心结论：**模型与训练框架（GraspDataset / train.py / flow matching / 损失）可以完全复用**，
只需把两个数据集离线转换成 HUG 的 pkl schema（`grasp` dict 装手部 MANO 标注即可）。

## 1. 目标 schema（复用 HUG 数据格式 → 训练代码零改动）

每个样本一个 pkl，字段与 1M-HUGS 完全同构：

```python
{
  "object_name": str,            # 可填序列名/物体名（重建任务语义化为"场景标识"）
  "frame_index": int,
  "grasp_index": int,
  "camera":  {"K": (3,3) float64, "width": 224, "height": 224},   # 224 分辨率下的 K
  "camera_original": {...},      # 方形裁剪分辨率的 K
  "image":  JPEG bytes,          # 224×224 RGB
  "depth":  PNG bytes,           # 224×224 uint16，1mm 单位
  "object_mask": PNG bytes,      # 224×224 手部 mask（语义变为"手"）
  "condition_point": (2,) float, # 手部关键点像素 [u, v]（224 坐标系）
  "grasp": {                     # 手部 MANO 标注（HUG 99D 格式）
    "t": (1,3), "R_6d": (1,6), "pose_6d": (1,15,6),
    "shape": (1,10),
    "landmarks_3d": (21,3),      # 相机系，米
    "landmarks_2d": (21,2),      # 224 图像坐标
    "T_camera_wrist": (4,4), "mesh_vertices": (778,3), ...
  },
}
```

复用点：`GraspDataset.__getitem__` / `_get_mano_params` / 点云裁剪 / query 采样 / train.py 全部不变。

## 2. 两数据集现状盘点（已实测验证）

### HO3D_v3（83,325 train / 20,137 eval 帧）

| 项 | 事实 | 出处/验证 |
|---|---|---|
| RGB | `train/<SEQ>/rgb/XXXX.jpg`，640×480 | ✓ |
| depth | `depth/XXXX.png`，**3 通道 uint8 编码 16-bit**：B 通道恒 0，**G=高字节、R=低字节**；`depth_m = (G*256+R) * 0.00012498664727900177` | ✓ 与手部关节深度对比一致（误差 <1cm） |
| 标注 | `meta/XXXX.pkl`：`camMat`(3,3)、`handPose`(48, 轴角)、`handBeta`(10)、`handTrans`(3)、`handJoints3D`(21,3) | ✓ |
| **坐标系坑** | **z 负在前、y 翻转**——必须做 `(x,y,z)→(x,-y,-z)`（绕 x 轴 180°）转换；global_orient 需左乘 `Rconv=diag(1,-1,-1)` | ✓ 可视化验证：转换前关节投影飘到显示器上，转换后精确落在手上 |
| split | 官方 `train.txt` / `evaluation.txt`；eval GT 在 `evaluation_xyz.json` / `evaluation_verts.json`（21 关节 + mesh 顶点，相机系米制） | ✓ GPGFormer 已用 |
| 注意 | 部分帧 `handJoints3D` 为空（0 维 object array）——转换时跳过 | ✓ 实测 |

### DexYCB（1,000 个序列 = 10 subjects × 多物体抓取，8 相机/序列）

| 项 | 事实 | 出处/验证 |
|---|---|---|
| RGB | `<SUBJ>/<SEQ>/<SERIAL>/color_XXXXXX.jpg`，640×480，8 个相机（序列号目录） | ✓ |
| depth | `aligned_depth_to_color_XXXXXX.png`，uint16（**毫米**，已与彩色对齐） | ✓ 中位数 ~2.2m 合理 |
| 手部标注 | `<SEQ>/pose.npz`：`pose_m` (N,1,51) = **3 维全局轴角 + 45 维 PCA 系数 + 3 维平移**，**世界坐标系**（整序列共享一份，需用外参转到各相机系）；全零帧 = 无标注，跳过 | ✓ toolkit 实证（见下方"已解决的待验证项"） |
| 内参 | `calibration/intrinsics/<SERIAL>_640x480.yml`（color 的 fx/fy/ppx/ppy） | ✓ |
| 外参 | `calibration/extrinsics_<DATE>/extrinsics.yml`，按序列号索引的 3×4（世界→相机）；序列用哪个外参文件由 `<SEQ>/meta.yml` 的 `extrinsics:` 字段指定 | ✓ |
| β | `calibration/mano_<DATE>_<SUBJ>_right/mano.yml` 的 `betas`（10 维，按 subject） | ✓ |
| 序列元信息 | `<SEQ>/meta.yml`：`serials`（8 相机）、`num_frames`、`mano_sides`、`ycb_ids` | ✓ |
| **GPGFormer 依赖** | 其 loader 读**预处理 JSON**（`DEX_YCB_{setup}_{split}_data.json`），**本机不存在** → 必须直接从原始 `pose.npz + calibration` 自建标注解析（可参考其 joints mm→m 的换算逻辑） | ✓ 已确认缺失 |

**✅ 已解决的待验证项（DexYCB pose_m 编码与外参方向，2026-08-27 全链路可视化验证通过）**：

1. **45 维关节 = PCA 系数**（非全轴角）。证据：`/root/code/dex-ycb-toolkit/dex_ycb_toolkit/layers/mano_layer.py:27-31` 官方 toolkit 用 `ManoLayer(use_pca=True, ncomps=45, flat_hand_mean=False)` 解码 pose_m。
2. **PCA → 轴角公式**：`aa45 = hands_mean + pca45 @ hands_components`（与 manopth `use_pca=True, flat_hand_mean=False` 的行为一致，GPGFormer `_pca_to_axis_angle` 同式）。`hands_mean`/`hands_components` 从 `MANO_RIGHT.pkl` 读取（HUG 的 `assets/mano/models/` 里有）。得到的绝对轴角可直接喂 HUG 的 MANO 层（`use_pca=False` 时 `flat_hand_mean` 参数不影响直接轴角输入）。
3. **外参方向**：`extrinsics.yml` 的 3×4 存储的是**相机→世界**（`X_world = R @ X_cam + t`，证据：`sequence_loader.py:324` 用 `[R|t]` 把相机系点云搬到世界系）。因此世界系 pose 转相机系要用**逆变换**：`R_cam = R_ext⁻¹ @ R_world`，`t_cam = R_ext⁻¹ @ t_w + t_ext_inv`（即 loader 的 `_R_inv/_t_inv`）。
4. **全链路验证**：frame 40（手抓蓝色罐子）按上述链路过 MANO 前向投影，21 个关节精确落在手上。

## 3. 转换管线设计

### 3.1 总体流程（离线一次性转换，两个数据集共用框架）

```
原始数据 ──→ 解析标注（各自 adapter）──→ 统一到 HUG 约定 ──→ 写 HUG-schema pkl
                                        │
        ┌───────────────────────────────┼───────────────────────────────┐
        ▼                               ▼                               ▼
  坐标系统一                      图像/深度统一                     MANO 统一
  OpenCV 约定（z 正在前）         224×224 + uint16 mm               轴角 → 6D 旋转
  相机系、米制                    K 同步调整                        → 99D + landmarks
```

### 3.2 共享转换工具（新增 `src/utils/` 或转换脚本内）

1. **axis-angle → 6D**：48 维轴角拆成 global(3) + 15×3；每个轴角 → Rodrigues → 旋转矩阵 → 6D（取矩阵前两列拉平）。仓库 `src/utils/transform_utils.py` 已有 `six_d_to_rotation_matrix`，反方向（`rotation_matrix_to_six_d`）需确认/补写。
2. **99D 组装**：`t = 手腕平移(相机系,米)`、`R_6d = 全局旋转 6D`、`pose_6d = 15×6`，拼接成 99D——与 `_get_mano_params` 的读取顺序严格一致。
3. **landmarks 生成**：用 HUG 自带的 MANO（`src/models/mano.py`，`center_idx=0` 手腕原点）前向得到 joints/verts，`landmarks_3d = joints + t`，`landmarks_2d = project(landmarks_3d, K_224)`。
4. **图像打包**：直接复用 `src/prepare_inputs.py` 的 `_center_crop_square` / `_adjust_K` / JPEG+PNG 编码逻辑（中心裁方 → resize 224 → K 调整），保证与 1M-HUGS 分布一致。
5. **深度统一为 uint16 mm**：DexYCB 原生即是；HO3D 按 §2 公式解码成米 → ×1000 → uint16。

### 3.3 HO3D 适配步骤

```
train.txt 每行 <SEQ>/<FRAME>:
  1. 读 meta/<FRAME>.pkl → camMat, handPose, handBeta, handTrans, handJoints3D
     （handJoints3D 为空则跳过该帧）
  2. 坐标转换：joints/trans (x,-y,-z)；global_orient: aa → R → Rconv@R → aa
  3. depth 解码：(G*256+R)*SCALE → 米 → uint16 mm
  4. β = handBeta（每帧自带）
  5. 走共享转换 → pkl
split: train.txt → 训练；evaluation.txt + evaluation_xyz/verts.json → 验证评估
```

### 3.4 DexYCB 适配步骤

```
遍历 <SUBJ>/<SEQ>/:
  1. 读 meta.yml → serials(8相机), extrinsics 文件名, num_frames
  2. 读 pose.npz → pose_m (N,1,51)，世界系
  3. 读 calibration：intrinsics/<serial>_640x480.yml（K）、
     extrinsics_<date>/extrinsics.yml（**相机→世界** 3×4，用逆变换）、
     mano_<...>/mano.yml（β，按 subject）
  4. 对每帧×每相机（pose_m 全零帧跳过）：
     a. PCA 解码：aa45 = hands_mean + pca45 @ hands_components
     b. 世界→相机：R_cam = R_ext⁻¹ @ R_world；t_cam = R_ext⁻¹ @ t_w + t_ext_inv
     c. 彩色图 + aligned depth（原生 uint16 mm）→ 共享转换 → pkl
split: 按官方 setup（s0/s1/s2 的序列划分）；建议先按 subject 留出（如 subject-09/10 做 val）
```

### 3.5 手部 mask 与 query 点生成

两数据集都**没有手部 mask**，自行生成（重建任务里 mask 的唯一作用是训练时随机采样 query 点，要求不高）：

- **mask**：MANO 前向 → 778 顶点投影到 224 图 → **凸包 + 形态学膨胀**（~5px）即可；
  若要更精确可用顶点投影做 Alpha Shape/简单光栅化。
- **condition_point**：从可见（落在图内）的 21 个手部关键点投影中**随机选一个**写入 pkl——
  与 HUG 训练时"mask 内随机采样"的增强效果一致（每次可重采样，见下）。

注意：`GraspDataset` 训练分支在**有 condition_point 时直接用、没有则从 mask 随机采样**。
建议 pkl 里**不写 condition_point**，让 dataloader 每次从手部 mask 随机采——白得 query 增强。

## 4. 关键设计决策

### 4.1 为什么"转换到 HUG pkl"而不是新写 Dataset 类

| 方案 | 优点 | 缺点 |
|---|---|---|
| **A. 离线转 pkl（推荐）** | train.py/GraspDataset 零改动；与 1M-HUGS 可混合训练；格式自包含 | 一次性转换成本（~66 万样本，约几十 GB） |
| B. 新写 Dataset 类 | 无冗余存储 | 需维护第三套 loader；与 HUG pipeline 的耦合点（点云构建/query 采样）要重写 |

推荐 A：转换脚本一次跑完，之后所有训练实验共用。

### 4.2 query 点与点云裁剪

- query 点 = 手上随机关键点（训练增强）；
- `pcl_crop_radius`：手（~0.2m）比 HUG 的物体小，建议从 0.3m 调到 **0.15–0.2m**（yaml 里改 `pcl_crop_radius` 即可，代码无需动）——让 4096 点更密集地覆盖手部。

### 4.3 β（手型）处理——三个选项

HUG 原设计把 β 固定为 canonical（`_build_dicts` 里 pred 和 GT 都用 `fixed_betas`），
但重建任务里手型随人变化（DexYCB 10 个 subject、HO3D 每帧 β 不同）：

| 选项 | 做法 | 评价 |
|---|---|---|
| **A. 固定 canonical（推荐 v1）** | GT landmarks 也用 canonical β 前向生成；β 信息丢弃 | 与 HUG 完全一致、零改动；损失绝对手型精度，PA 指标不受影响，绝对尺度指标略损 |
| B. GT 用真实 β | pred 用 canonical、GT 用样本 β | 引入 pred/GT 系统偏差，**不要** |
| C. 预测 β（v2 扩展） | 99D → 109D（+10 维 β），flow state 加一组 token | 最完整的重建方案；要改 DiT 输入/输出头和 norm_stats，工作量中等 |

**建议 v1 用 A 快速拉通，v2 视精度需求上 C。**

### 4.4 norm_stats 必须重算

`assets/norm_stats.json` 是 1M-HUGS 抓取分布的统计量（手腕 z 均值 0.54m 等），
手部重建的 t/pose 分布不同，**必须在转换后的新数据上重算**。
注意：仓库里 `grasp_model.py` 报错信息提到 `python -m src.utils.compute_norm_stats`，
但该脚本**并不存在**，需要新写（扫一遍转换后 pkl 的 grasp 字段，逐组算 mean/std）。

### 4.5 训练配置改动清单（configs/train_hug.yaml）

```yaml
trainer:
  model:
    pcl_crop_radius: 0.2        # 0.3 → 0.15~0.2（手更小）
  data:
    dataset_path: <转换后 pkl 目录>   # 可指向含 DexYCB+HO3D 混合 pkl 的目录
  train:
    total_steps: 50000          # 数据量 ~66 万，约为 1M-HUGS 的一半，可酌减
```

混采：直接把两数据集 pkl 放同一目录（GraspDataset 递归 glob），
或分目录+各写 samples 清单后合并；想加权采样可在 train.py 给 DataLoader 加 WeightedSampler（可选）。

## 5. 验证清单（转换正确性的判定标准）

1. ~~**DexYCB pose 编码判定**~~ **已解决**：45 维为 PCA，解码公式与外参方向均已全链路验证（见 §2 末尾"已解决的待验证项"）。
2. **每数据集 overlay 抽查**：随机 10 帧，`landmarks_2d` 画在 224 图上应落在手上；
   手背点云（由 depth 反投影）应与 MANO mesh 贴合。
3. **数值一致性**：HO3D 的 `handJoints3D`（转换后）≈ MANO 前向 joints + handTrans（误差应 <1cm）；
   DexYCB 检查 `t` 的 z 值分布在合理范围（~0.3–1.5m）。
4. **统计合理性**：重算 norm_stats 后，translation mean/std 应在合理范围（z 均值约 0.4–0.6m）。
5. **冒烟训练**：`--max-steps 30` 跑通，loss 从 ~2.5 起步正常下降（参考 1M-HUGS 冒烟基线）。

## 6. 实施步骤与工作量

| 步骤 | 内容 | 预估 |
|---|---|---|
| 1 | 写共享转换工具（aa→6D、99D 组装、mask 生成、打包） | 半天 |
| 2 | HO3D adapter（meta 解析 + 坐标翻转 + 深度解码） | 半天 |
| 3 | DexYCB adapter（pose.npz + 标定解析 + 世界→相机）+ §5.1 编码判定 | 一天 |
| 4 | 离线转换两个数据集（并行跑，~66 万 pkl） | 数小时机时 |
| 5 | 写 compute_norm_stats 并重算 | 1 小时 |
| 6 | 冒烟训练 + overlay 抽查 + 正式训练 | 半天 |
