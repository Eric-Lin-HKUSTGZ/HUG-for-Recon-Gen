# HUG 手部重建训练任务说明

基于 HUG 框架的手部重建（hand reconstruction）训练任务：支持 RGB-only 和 RGB-D 两种输入，
用 flow transformer（DiT）回归 109D MANO 重建状态（t(3) + R_6d(6) + pose_6d(90) + shape_gt(10)），
数据为 **HO3D_v3 + DexYCB 混合训练**。109D 状态由 t(3)、R_6d(6)、pose_6d(90) 和 subject-specific shape_gt(10) 组成；99D 仍保留兼容模式。
当前 DexYCB 主线为 **v30 native geometry overlay**：图像和姿态保持 canonical-right，beta 保留其来源手的 MANO shape basis。旧 v2 canonical 方案仅用于历史实验复现。

## 1. 训练指令

v30 DexYCB 新实验必须从 step 0 启动。RGB-only 与 RGB-D 分别使用：

```bash
cd /root/code/HUG-for-Recon-Gen
torchrun --nproc_per_node=4 -m src.train \
  --config configs/train_handrecon_v30_native_rgb_only.yaml

torchrun --nproc_per_node=4 -m src.train \
  --config configs/train_handrecon_v30_native_rgbd.yaml
```

- v30 配置必须同时设置 `mano_geometry: native_side_v1`、`geometry_overlay` 和 overlay 自带的 `norm_stats.json`；不要单独移植其中一个字段。
- 旧主线仅用于复现：`torchrun --nproc_per_node=4 -m src.train --config configs/train_handrecon.yaml`。第 3 节保留其摘要。
- `/root/code/vepfs/HUG-for-Recon-Gen/hand_recon/`（`output_dir`，vepfs 大文件系统）与
  其下的 `train_log.jsonl`（`train.log_file` 可配置路径+文件名；相对路径按
  `output_dir` 解析，默认 `<output_dir>/train_log.jsonl`）

### 冒烟测试（smoke test）

```bash
torchrun --nproc_per_node=4 -m src.train \
    --config configs/train_handrecon_v30_native_rgb_only.yaml \
    --max-steps 10 --max-train-samples 20000
```

⚠️ 冒烟测试会覆盖 `output_dir` 里的日志，正式训练前建议把 smoke 的
`output_dir` 改成别的目录（如 `outputs/hand_recon_smoke`），或在正式训练前清空。

## 2. 数据

### 数据来源（转换产物，由 `GraspDataset` 直接消费）

| 数据集 | 路径 | train 帧数 | val 帧数 | test 帧数（官方） |
|---|---|---|---|---|
| HO3D_v3 | `/root/code/vepfs/dataset/hand_recon_hug/ho3d` | 77,209 | 5,761 | 17,224（evaluation split） |
| DexYCB（v30 base PKL） | `/root/code/vepfs/dataset/hand_recon_hug/dexycb_v4_fullres_shape_gt` | 394,193 | 21,785 | 76,845（s0_test） |
| 合计 | — | **471,402** | 27,546 | **94,069** |

划分清单（stem 列表，位于数据目录外的 `splits/`）：
`/root/code/vepfs/dataset/hand_recon_hug/splits_v2/` 下
`{ho3d,dexycb}_{train,val,test}.clean.txt` 与 `ho3d_eval.clean.txt`。目录名沿用 `splits_v2`，官方划分和 stem 清单未因 v30 改变。

- HO3D train/val：按序列（recording-level）留出验证集；test = 官方
  evaluation split（无 MANO，只有 joints/verts GT）
- DexYCB：官方 s0 split（s0_train / s0_val / s0_test）。v30 继续复用 v4 full-resolution PKL 中的 RGB、Depth、相机和 canonical-right 图像，不重写几十万份 PKL；几何字段由 `/root/code/vepfs/dataset/hand_recon_hug/dexycb_native_overlay_v1` 按样本名覆盖。test 清单按 `s0_test.jsonl` 检索生成，不是重新划分。
- `.clean` 过滤规则（`filter_empty_masks.py`，`--lists` 可指定子集）：
  HO3D train / DexYCB 全部 = 手部 mask 非空（剔 ~3%）；HO3D eval =
  手腕投影（`condition_point`）在 224 画面内（官方 eval 集不发布分割
  mask，`object_mask` 为空字节，剔 14.5%）。想严格按官方全集评测可改用
  `.txt` 清单

v30 归一化统计使用 `/root/code/vepfs/dataset/hand_recon_hug/dexycb_native_overlay_v1/norm_stats.json`，由 builder 仅对 394,193 个 train 样本计算，无 val/test 泄漏。旧配置继续使用 `assets/norm_stats_dexycb_109d.json` 或混合数据统计，但这些统计不能用于 v30。加载
`train.pretrained` 时会跳过源 checkpoint 的 normalization buffers，保留当前
配置指定的统计；`train.resume` 则恢复完整 checkpoint。切换统计后必须新开训练，
不能 resume 旧 checkpoint。

当前实验配置按要求在训练期间从 DexYCB 官方 `s0_test` 等距抽取固定 4096 条
评测，并以其 PA-MPJPE/PA-MPVPE 均值选择 `model_best.pt`；训练结束后再在完整
`s0_test` 上汇报。由于 test 已参与 checkpoint 选择，该结果属于 test-selected
实验，不能作为独立盲测成绩。

### DexYCB v30 native geometry overlay（当前方案，2026-09-17）

#### 问题证据与根因

v29 RGB-only 消融的 MPJPE/MPVPE 变差符合移除深度后的预期，但它与同为单 RGB 的 HandFlow 在 DexYCB 上仍有约 2.5 mm PA-MPJPE 差距。进一步按手侧检查发现，旧 canonical v2 的左手几何误差显著高于右手：2000 样本审计中，左手 3D mean 为 7.2 mm、P90 为 12.3 mm，右手分别为 2.8 mm 和 5.0 mm；左手差异在 PA 对齐后的 joints 和 mesh 中仍存在，因此不是全局平移、旋转或尺度误差。

根因是旧链路把左手图像、相机坐标和 pose 镜像到 right canonical 空间后，又把 DexYCB 的左手 subject beta 送进 `MANO_RIGHT.pkl`。图像/姿态的 X 反射不会把 `MANO_LEFT` 与 `MANO_RIGHT` 的 identity shape basis 变成同一个基底。不能通过 beta 取反或交换符号修复；对两个 shapedirs 拟合固定 10x10 线性转换矩阵后，shape basis 相对残差仍约为 **0.543613**，无法等价表示。

GPGFormer 中可复用的是：按 `mano_sides` 选择原生 MANO、保留官方 joints，并对左手图像与相机系几何执行一致的 X 反射。其 right-MANO mesh fallback 不保证左手 beta 对应的 mesh 正确，因此 v30 没有照搬这部分。

#### v30 几何约定

参数约定名为 `source_beta_canonical_pose_v1`：

1. RGB、Depth、相机系 pose 仍使用已有 canonical-right 输入；左手输入只在上游做一次 X 镜像。
2. 109D 中的 10D beta 保留 DexYCB 来源手侧的原始数值，不转换到共享 right basis。
3. 右手样本用 `MANO_RIGHT.pkl` 解码；左手样本用真实 `MANO_LEFT.pkl` 解码，左手 rotation 按反射矩阵变换，解码后的 joints/vertices 再做一次 X 反射，进入与图像相同的 canonical-right 相机坐标系。
4. 左手使用其独立的 fingertip 索引；镜像后的左手三角面反转 winding，保证网格朝向。
5. joint 监督直接使用镜像后的 DexYCB 官方 `joint_3d`；mesh 监督使用来源手侧 MANO 生成并镜像的 778 顶点。训练和评测只有这一套目标，不存在“当前 GT / native GT”两套指标口径。

`source_is_left` 表示输入在 canonicalization 前的手侧，只用于选择 MANO shape basis、fingertips 和 mesh faces。它随 batch 传到 loss、评测和可视化解码器，但不编码成 condition token，也不提供给 RGB/Depth/fusion/flow transformer，模型不能借此直接预测 pose 或 shape。对没有可靠来源手侧的部署输入，调用方必须保存 detector 做镜像时采用的 side；`native_side_v1` 缺少该字段会直接报错，不会静默假设为右手。

#### overlay 产物

原始 492,823 个 v4 PKL 未修改。独立 overlay 位于：

```text
/root/code/vepfs/dataset/hand_recon_hug/dexycb_native_overlay_v1
```

它占用约 4.9 GB，通过 NumPy memory map 按需读取，不复制 RGB、Depth 或 PKL。`samples.npy` 提供排序且唯一的样本索引；逐帧数组包括 `params`(109)、`pose_aa`(48)、官方 `joints`(21x3)、native `vertices`(778x3)、`camera_K` 和 `source_is_left`，并分别保存左右手 faces。`manifest.json` 记录数据根目录、split、输入标签、builder/decoder/asset 的 SHA-256 以及所有输出数组哈希；`norm_stats.json` 只由 394,193 个 train 样本计算。

| 检查项 | 结果 |
|---|---|
| train / val / test | 394,193 / 21,785 / 76,845，共 492,823 帧 |
| manifest | `status=complete`，`count=492823`，`train_count=394193` |
| 最大 native MANO 解码 / 官方 joints 误差 | 0.000266664 mm |
| 最大官方 2D / 3D 投影误差 | 0.000080521 px |
| 完整性 | 第二次 builder 调用已逐文件复核数组及 norm stats 哈希 |

`GeometryOverlay` 打开时会检查 version、convention、complete 状态、dataset root、数组 shape/dtype、样本唯一性和 mesh topology；数据集启动时检查清单全覆盖；逐样本再检查 source side、canonicalization、640x480 相机 K 和 wrist 一致性。任一项不符立即失败。旧 geometry 生成的 `object_mask` 和 `condition_point` 会被移除，v30 强制使用 detector crop，避免旧错误 mesh 通过 mask/query 泄漏进新实验。

#### 代码职责

| 文件 | 职责 |
|---|---|
| `src/models/native_mano.py` | 加载左右 MANO basis，按来源手侧解码并镜像左手输出；磁盘上的 MANO asset 不做任何修改 |
| `scripts/build_dexycb_geometry_overlay.py` | 从官方 DexYCB label 构建 109D、官方 joints 和 native mesh，逐帧验证并生成 manifest/hash/stats |
| `src/dataloader/geometry_overlay.py` | 只读 mmap overlay，覆盖 PKL 几何字段并执行 dataset/sample/side/camera/wrist 防错检查 |
| `src/dataloader/grasp_dataset.py`、`augmented_grasp_dataset.py` | 将 overlay 与 `source_is_left`、官方 joints、native vertices 接入 batch |
| `src/models/grasp_model.py`、`src/models/mano.py` | 在 flow/joint/mesh loss 和单帧/动画解码中使用同一 native 几何约定 |
| `src/train.py`、`src/eval_test.py` | 训练、验证、测试透传几何元数据；checkpoint 记录 convention 并拒绝跨 convention resume |
| `src/inference.py`、`src/app.py` | dataset 推理/可视化读取 overlay side，并选择对应 faces |
| `configs/train_handrecon_v30_native_{rgb_only,rgbd}.yaml` | 完整 v30 配置；前者 256 RGB tokens，后者 256 RGB + 256 Depth tokens |

RGB-only 从 512 降为 256 是序列长度变化，`d_fusion=1024`、`d_model=512` 等 token 特征维度不变，attention 支持变长序列，因此不会造成架构维度错误。v30 RGB-D 仍为 512 tokens，并启用 fusion activation checkpointing。

#### 构建、复核与训练

overlay 已全量生成，正常训练无需再运行 builder。确需从官方 label 重建或复核时：

```bash
cd /root/code/HUG-for-Recon-Gen
HUG_PYTHON=/root/code/vepfs/miniconda3/envs/hug_mediapipe/bin/python \
  bash scripts/run_dexycb_geometry_overlay.sh
```

builder 使用独占 `.builder.lock`。中断产物只有显式 `--resume` 才能继续，并会先验证已完成部分的 source hash；已有完整产物会验证全部输出哈希后打印 `Already complete; no output modified`。manifest、split、代码或 asset 签名不一致时必须换新的输出目录，不能覆盖现有产物。

训练前确认配置同时指向 v4 base PKL、完整 overlay 和 overlay stats。v30 可加载 HUG 预训练结构参数（已验证 116 个 tensor 兼容），但必须 `resume: null` 并从 step 0 训练；`src/train.py` 也会拒绝从 `legacy_right` checkpoint resume 到 `native_side_v1`。输入 RGB/Depth、相机 K、RTMPose 和 detector condition cache 均未改变，可以继续复用。当前两个 v30 配置的 affine/depth/pointcloud augmentation 仍默认关闭；需要做增强实验时另建配置，避免覆盖 v30 基线口径。

#### 离线 condition cache 与 affine 增强同步

`AugmentedGraspDataset` 已支持在不重建 condition cache、也不在线运行 detector/RTMPose 的前提下开启 affine。cache 内坐标仍保持原始整图坐标，每次取样后使用本次真正生效的 2x3 全图仿射矩阵同步变换：

- `keypoints_xy` 按点变换；落出增强后图像的点将 confidence 置零，后续不会进入 skeleton condition、Depth ROI 或 palm anchor。
- `detector_bbox_xyxy` 必须变换四个角再取轴对齐包围框，不能只变换左上和右下角。
- `crop_bbox_xyxy` 根据变换后的 detector bbox 和 `hand_crop.expand` 重新生成，保持与在线 detector crop 相同的 square/expand 定义。
- RGB、Depth、mask、GT 2D landmarks、query 和 `K_full` 继续使用同一个全图变换；RGB crop 后另有 `K_rgb = A_crop @ K_full`。
- visibility 检查可能拒绝一次随机 affine。实现记录的是 `applied_affine_matrix`，只有图像实际完成变换才同步 cache；被拒绝时图像与 cache 都保留原坐标，禁止直接复用采样参数。

回归覆盖 cache 不可变、旋转 bbox 四角变换、出界关键点失效及 rejected-affine 回退。另用 v30 真实训练列表、native overlay 和 detector/RTMPose cache 等距抽查 100 帧：93 帧应用 affine、7 帧按可见性回退，`K_full`、`K_rgb`、crop-space keypoints 最大误差均为 0，100 帧输出全部有限。

#### v31 GT joints + augmentation

`configs/train_handrecon_v31_native_rgbd_gt_aug.yaml` 基于 v30 native RGB-D，只改变条件来源和训练增强，用于测量干净 GT 2D 条件下数据增强的收益：

- `train_keypoint_source`、`eval_keypoint_source` 和回退 `keypoint_source` 均为 `gt`；不配置 condition cache 或 RTMPose 路径，train/val/test 都不会初始化 RTMPose。
- train 由 GT joints 生成 RGB crop；val/test 仍用 detector bbox 生成 RGB crop，但 skeleton condition、Depth ROI 和 palm anchor 使用 GT joints。这与“训练 GT、测试 GT”的关键点口径一致，不代表使用 GT detector bbox。
- `skeleton_drop_prob: 0.0`，每个有标注样本均保留 GT skeleton。
- train 启用轻量 RGB 通道/亮度/对比度增强、60% affine，以及 50% 概率的中等强度 Depth 和 point-cloud 扰动；val/test 保持无增强。
- affine 范围为 scale `0.85-1.15`、rotation `+-15` 度、translation `+-8%`，并要求至少 95% GT joints 留在画面内。GT joints、RGB、Depth、K 和 query 使用同一变换。

启动命令：

```bash
cd /root/code/vepfs/repos/HUG-for-Recon-Gen
torchrun --nproc_per_node=4 -m src.train \
  --config configs/train_handrecon_v31_native_rgbd_gt_aug.yaml
```

该配置输出到 `hand_recon/20260917_v31_native_rgbd_gt_aug`，正式测试结果写入其中的 `test_results_gt.json`。已用真实 train/val 样本验证两边各有 21 个有效 GT 点、cache 路径为空且 RTMPose runtime 未创建。

#### v32 DINOv2 + LoRA

`configs/train_handrecon_v32_native_rgbd_gt_aug_lora.yaml` 是 v31 的严格模型消融：数据集、native MANO 几何、GT joints、RGB/Depth 输入、增强、loss、batch size 和训练步数全部沿用 v31，唯一变量是把冻结的 DINOv2 RGB encoder 改为 LoRA 微调。正式对比时 v31 和 v32 都必须使用当前训练代码从 step 0 启动。

- 在 DINOv2-base 的第 8--11 层 self-attention `query` 和 `value` 线性层注入 LoRA；`rank=8`、`alpha=16`、`dropout=0.05`，共 16 个 adapter tensor、98,304 个可训练参数。
- DINOv2 原始参数全部冻结。LoRA 的 `B` 矩阵零初始化，因此 step 0 的 encoder 输出与冻结 DINOv2 完全一致，初始差异不会污染消融。
- 第 8--11 层使用非重入 activation checkpoint。encoder 前向和 `encode_scene()` 仅在 encoder 完全冻结时使用 `no_grad`，LoRA 模式会保留梯度图。
- optimizer 分为主模型和 `image_encoder_lora` 两组，共用 warmup/cosine 调度。主模型峰值 LR 为 `1e-4`；LoRA 峰值 LR 为 `5e-5`、`lr_scale=0.5`、weight decay 为 0。
- checkpoint 继续删除冻结 DINOv2 基座以控制文件大小，但保留 raw model 和 EMA 中的 `lora_A/lora_B`。加载 checkpoint 时先从配置路径恢复同一个 DINOv2 基座，再覆盖 adapter。

启动命令：

```bash
cd /root/code/vepfs/repos/HUG-for-Recon-Gen
torchrun --nproc_per_node=4 -m src.train \
  --config configs/train_handrecon_v32_native_rgbd_gt_aug_lora.yaml
```

输出目录为 `hand_recon/20260917_v32_native_rgbd_gt_aug_lora`，结构化日志为 `logs/hand_recon/20260917_v32_native_rgbd_gt_aug_lora.jsonl`。真实本地 DINOv2 权重验证得到 98,304 个 LoRA 参数、16 个可训练 tensor、有效非零 `lora_B` 梯度和 `[8, 9, 10, 11]` checkpoint 层；adapter-only checkpoint 重载后的输出与保存前完全一致。完整 v32 模型构造确认 DINOv2 基座可训练参数为 0，LoRA optimizer 参数集合无遗漏或重复。相关回归测试为 `tests/test_dinov2_lora.py`。

#### 训练梯度清零修复与实验口径

共享训练循环此前缺少 `optimizer.zero_grad()`：每一步执行 `optimizer.step()` 后，旧梯度仍留在参数上并与下一步梯度相加。这不是配置中的梯度累积，因为每一步都会更新参数，也没有按累积步数缩放 loss。当前代码已在每个训练 step 的前向传播前调用 `optimizer.zero_grad(set_to_none=True)`，并有回归测试约束其位于 step 循环内且早于 `ddp_model()`。

该修复会影响所有以后启动的训练。v31 与 v32 的公平 LoRA 消融必须都使用修复后的代码重新从 step 0 训练；任何在修复前启动的 v31/v32 run 都不能与修复后的 run 直接比较。已经完成的 v30 指标仍按其当时训练代码作为历史结果保留，但与新训练结果的细微差异不能全部归因于 LoRA、GT joints 或增强中的单一变量。

已完成的回归验证包括：geometry overlay 8 项、native MANO/梯度/动画 5 项、geometry repair 5 项、原 HUG loss 5 项；1,536 帧 smoke overlay、492,823 帧全量逐样本检查、真实左右手 PKL + detector cache 加载、2-worker DataLoader、v30 RGB-only 模型构造、预训练权重加载、Python 编译、配置解析和 `git diff --check` 均通过。尚未启动 v30 训练。

#### 历史 canonical v2（仅复现旧实验）

`/root/code/vepfs/dataset/hand_recon_hug/dexycb_v2_canonical_right`、`assets/norm_stats_dexycb_109d.json` 和 `mano_geometry: legacy_right` 属于旧方案。其 2000 样本 parity audit 的 left 3D mean 7.2 mm / P90 12.3 mm、right mean 2.8 mm / P90 5.0 mm 正是促成本次修复的证据，不能当作 v30 的当前精度。保留旧路径是为了复现实验，不要与 v30 overlay、stats 或 checkpoint 混用。

## 3. 关键超参（`configs/train_handrecon.yaml`）

| 项 | 值 | 备注 |
|---|---|---|
| d_mano | 109 | 99D pose + 10D subject-specific shape_gt；99D 仍可兼容 |
| query_fusion_mode | legacy_broadcast | 恢复 HUG 发布代码/预训练 checkpoint 的融合方向：scene tokens 作 Q，单个 query token 作 K/V |
| total_steps | 25,000 | 预训练权重 finetune，25k 步足够 |
| batch_size | 256 / 卡（×4 卡 = 1024） | |
| lr / weight_decay | 1e-4 / 0 | AdamW，betas (0.9, 0.999)；warmup 后 cosine 衰减到 `lr × lr_min_ratio`（默认 0） |
| warmup_steps | 2,500 | total_steps 的 10% |
| grad_clip | 1.0 | |
| lambda_v / lambda_3d / lambda_mesh_3d | 1.0 / 20.0 / 10.0 | HUG velocity/joint loss，并增加 `(1-t)` 加权的相机系 778 顶点 L1；joint:mesh 为 2:1 |
| ema_start_step / decay | 12,500 / 0.999 | total_steps 的 50%（论文同比例） |
| bf16 | true | |
| seed | 42 | |
| pretrained | `/root/code/vepfs/HUG-for-Recon-Gen/hug_checkpoint/hug_full.safetensors` | HUG 官方预训练权重（EMA，85K 步），仅加载结构与形状匹配的模型参数做 finetune；冻结的 DINOv2 从 HF cache 加载，不含 optimizer/step |
| log / val / ckpt 间隔 | 20 / 1,000 / 1,000 | val 与 ckpt 对齐，续训点即评估点 |
| n_points_input | 4096 | query 点数 |
| pcl_crop_radius | 0.2 | 手比物体小，0.3→0.2 让点云密集覆盖手部 |

本节表格记录旧 `configs/train_handrecon.yaml` 的实验口径。v30 除新增 `mano_geometry: native_side_v1`、`native_left_asset`、`geometry_overlay` 和 overlay stats 外，还包含后续已经确认的 detector + RTMPose crop、skeleton condition、双流 DiT 等设置；应以两个 v30 YAML 为准，不要用本节手工拼装 v30 配置。

模型结构：109D flow state 将 shape_gt 作为独立 token；RGB 用冻结 DINOv2-base（带 registers），点云用可训练 PointNeXt
（`pcl_width=64`，SA radii `[0.025, 0.05, 0.10, 0.20]`），
fusion transformer（`d_fusion=1024, 4 层, 8 头, 256 patches`），
flow transformer（`d_model=512, 6 层, 8 头, 50 步采样`）。
模态开关与论文 full model 一致：`use_rgb + use_depth + pointpainting`。
当前使用 `legacy_broadcast`：它与 `/root/code/hug/src/models/fusion.py` 和官方
checkpoint 的参数结构一致。`query_to_scene` 分支仍保留在代码中，便于复现实验，
但本配置不启用。

训练目标包含 HUG 的 flow/joint 两项，并增加直接 mesh 监督：

```text
L = 1.0 * MSE(v_pred, v_target)
  + mean_batch[(1 - t) * (
        20.0 * mean_joint_xyz(|J(x0_hat) - J(x0)|)
      + 10.0 * mean_vertex_xyz(|V(x0_hat) - V(x0)|))]
```

flow MSE 对完整 109D 状态统一求均值，不再对 translation、wrist、finger、shape
分组重加权。joint 和 mesh 都在相机坐标系直接监督，分别对 21 个关节和 778 个
顶点的 xyz 坐标取 L1 mean，并共享精确的 `(1-t)` 权重。PA 对齐 joint/vertex、theta、
骨方向/长度、独立 shape、translation/rotation 和 2D 重投影项仍全部关闭；配置中
这些历史扩展项的 lambda 均为 0，训练调用也不读取它们。

## 4. 验证集与指标（真实采样口径，`trainer.val` 段）

**验证/选型集 = DexYCB 官方 s0_val + HO3D 官方 evaluation split**（后者跨主体，
用于确保选出的模型泛化到未见过的主体）。每次验证：

- 走真实推理路径 `sample()`（50 步 ODE 完整采样），**不算 loss**（不反向传播、
  不参与选型，只无谓开销）
- 两种 GT schema 自动路由：DexYCB（MANO GT）-> `build_loss_dicts`；
  HO3D eval（joints/verts GT，无 MANO）-> `mano_forward` 比对
- 指标（`src/metrics.py`，相机系、单位 mm）：**MPJPE / PA-MPJPE**（21 关节）、
  **MPVPE / PA-MPVPE**（778 顶点）；PA 即 Procrustes 对齐
- 各数据集按占比等距采样共 `max_samples=4096` 条，确定性、跨 checkpoint 可比
- **多卡分片并行**：每 rank 用 `DistributedSampler` 评估自己的 shard 后
  all_reduce 聚合（rank-0 单卡跑采样式 val 会超过 NCCL 默认 600s 看门狗
  导致全任务崩溃；NCCL 超时已放宽到 30min 兜底 vepfs 慢写）
- EMA 启动后（`ema_start_step`）验证的是 **EMA 权重**（部署用的就是它），
  之前验证原始模型

**best 模型保存判据**：各数据集 `0.5×(PA-MPJPE + PA-MPVPE)` 后**等权平均**为
score，创新低即保存 `model_best.pt`（等权防止被样本量大的 DexYCB 主导）。

历史教训（已修复）：

- 旧 val 走 `forward()` 的"随机 t 单步恢复 x0"近似指标，约 2 倍乐观且尺度
  噪声大，导致 best 选择失真（曾挑中 9k 步的 checkpoint）。checkpoint 里的
  `val_metric` 版本标记保证 resume 时 best_val 自动重置、口径不混
- 旧逻辑在 EMA 启动前保存 best 时，"ema" 字段是未更新的预训练初始权重；
  现在未启动时存 `null`（评测端自动回退到 model 权重）

## 4b. 官方测试集全量评测（`src/eval_test.py`）

训练中的 val 用于 best checkpoint 选择；最终指标用官方测试集**全量**评测
（`torchrun --nproc_per_node=4 -m src.eval_test --config configs/train_handrecon_v30_native_rgb_only.yaml`，
数据位置在 `trainer.test` 段配置）：

- **DexYCB s0_test**（76,845 条）：v30 overlay 带完整官方 joints 和 native MANO mesh GT，
  `sample()`（50 步 ODE 完整采样）-> `build_loss_dicts()` -> 四项指标
- **HO3D_v3 官方 evaluation split**（17,224 条）：只有 joints/verts GT
  （官方就不发布 MANO），`sample()` -> `mano_forward()` -> 与
  `joints_gt/verts_gt` 比对，同样四项指标。两条路径按 batch 内有无
  `mano_params` 字段自动切换
- 清单：`splits/dexycb_test.clean.txt` / `splits/ho3d_eval.clean.txt`
  （`make_handrecon_splits.py` + `filter_empty_masks.py --lists
  dexycb_test,ho3d_eval` 生成，剔除规则见第 2 节）
- 多卡分片：`torchrun --nproc_per_node=N`，指标经 all_reduce 聚合
- 默认评估 `<output_dir>/model_best.pt` 的 **EMA** 权重，结果表格打印并写
  `<output_dir>/test_results.json`；`--ckpt/--weights/--sets/--steps/
  --batch-size/--limit` 可覆盖（`--limit` 仅冒烟用，正式评测勿加）
- 冒烟：`python -m src.eval_test --config configs/train_handrecon_v30_native_rgb_only.yaml
  --ckpt <任意ckpt> --steps 2 --limit 32`（随机权重已验证两条路径跑通）

v30 评测必须继续使用对应配置，使 overlay side 与 native decoder 一起生效：

```bash
torchrun --nproc_per_node=4 -m src.eval_test \
  --config configs/train_handrecon_v30_native_rgb_only.yaml
```

评测链路直接以 batch 中的官方 `gt_joints_3d` 和 native `gt_vertices` 为唯一 GT，预测值按同一 `source_is_left` 解码；不要另算 legacy-right GT 后与该结果混合汇报。

## 5. 输出与恢复

### TensorBoard 实时曲线

训练进程只在 rank 0 写 TensorBoard，避免 DDP 产生重复曲线；JSONL 仍作为完整
事实日志保留。当前配置对应的 event 目录为：

```text
/root/code/vepfs/HUG-for-Recon-Gen/hand_recon/20260910_v22_hug_loss_mesh_no_augmentation/tensorboard
```

TensorBoard 由 `trainer.train.tensorboard` 控制：

```yaml
tensorboard:
  enabled: true
  log_dir: tensorboard   # 相对路径按 output_dir 解析，也可以填写绝对路径
  flush_secs: 10
  max_queue: 10
```

可视化步骤：

1. 按第 1 节命令启动训练。第一次写训练日志后，`<output_dir>/tensorboard/`
   下会出现 `events.out.tfevents.*` 文件。
2. 在服务器的另一个终端启动 TensorBoard。使用与训练相同的 Python 环境，
   并只监听服务器回环地址：

   ```bash
   /root/code/vepfs/miniconda3/envs/hug/bin/python -m tensorboard.main \
     --logdir /root/code/vepfs/HUG-for-Recon-Gen/hand_recon/20260910_v22_hug_loss_mesh_no_augmentation/tensorboard \
     --host 127.0.0.1 \
     --port 6006
   ```

3. 在本地电脑另开终端，通过训练服务器建立 SSH 端口转发：

   ```bash
   ssh -N -L 6006:127.0.0.1:6006 \
     -p 58225 root@115.190.90.101
   ```

4. 浏览器打开 `http://127.0.0.1:6006`。TensorBoard 的 **Custom Scalars**
   页面已预设 flow/3D 的加权 loss、原始 loss 和相机系几何误差面板；完整曲线可以在
   **Scalars** 页面查看，训练配置位于 **Text -> run/config**。

主要曲线及含义：

| 曲线前缀 | 内容 |
|---|---|
| `loss/total` | 反向传播使用的总损失 |
| `loss_weighted/flow` | `lambda_v * velocity MSE` 对总损失的真实贡献 |
| `loss_weighted/3d_landmarks` | `lambda_3d * mean[(1-t) * landmark L1]` 对总损失的真实贡献 |
| `loss_weighted/3d_mesh` | `lambda_mesh_3d * mean[(1-t) * vertex L1]` 对总损失的真实贡献 |
| `loss_raw/flow` | 完整 109D velocity MSE |
| `loss_raw/3d_landmarks` | 未乘 `(1-t)` 与 lambda 的相机系 joint L1 |
| `loss_raw/3d_landmarks_time_weighted` | 乘 `(1-t)`、未乘 `lambda_3d` 的 joint L1 |
| `loss_raw/3d_mesh` | 未乘 `(1-t)` 与 lambda 的相机系 vertex L1 |
| `loss_raw/3d_mesh_time_weighted` | 乘 `(1-t)`、未乘 `lambda_mesh_3d` 的 vertex L1 |
| `error_mm/*` | 仅作监控的当前训练 batch 相机系 MPJPE/MPVPE，不参与 loss |
| `optimization/*` | 学习率和裁剪前的总梯度范数；实际梯度仍按 `grad_clip` 裁剪 |
| `schedule/one_minus_t_mean` | 当前 batch 的 `(1-t)` 均值 |
| `validation/<dataset>/*_mm` | 完整 ODE 采样得到的 MPJPE、PA-MPJPE、MPVPE、PA-MPVPE |
| `validation/selection_score_mm` | `0.5 × (PA-MPJPE + PA-MPVPE)`，即 `model_best.pt` 的选择依据 |

`error_mm/train_*` 来自训练时随机时间步的 decoded-x0 结果，只适合观察优化
趋势；它与验证阶段完整 ODE 采样计算的 `validation/*/pa_*` 口径不同，不能直接当作
最终精度。观察 `loss_weighted/flow`、`loss_weighted/3d_landmarks`、
`loss_weighted/3d_mesh` 和验证 PA 指标：前三者用于判断三项训练目标的量级，后者
用于判断真实性能是否提升。

断点续训时继续使用同一个 `log_dir` 即可。训练代码根据 checkpoint 的
`start_step` 设置 TensorBoard `purge_step=start_step+1`，隐藏旧 event 中 checkpoint
之后的残留点，曲线会从已保存 step 后继续。新实验应使用新的 `output_dir`，避免与
其他实验的 event 混在一起。

如果 6006 端口已被占用，可同时把服务器命令的 `--port` 和本地转发命令中的两个
6006 改成同一个空闲端口。若环境中没有 TensorBoard，训练会继续写 JSONL 而不会
失败；依赖已在 `environment.yaml` 中声明。


`train.log_file` 指定结构化训练指标 JSONL；每条记录包含
`schema_version`、`event`、`run_id`、UTC 时间、rank/world_size 和 step。
此外，`output_dir/logs/` 下每个 rank 都有：

- `rank-<rank>.log`：该进程的完整 INFO/WARNING/ERROR 文本日志
- `rank-<rank>.error.log`：仅 ERROR 及以上，保留完整 traceback
- JSONL：rank 0 的 startup、配置、数据集、heartbeat、train、validation、checkpoint
  和 run_finished 事件，写入后立即 flush；`HUG_LOG_FSYNC=1` 可进一步启用 fsync

Python 未捕获异常会写入对应 rank 的 error 日志后重新抛出。OOM、SIGKILL、NCCL
原生 abort 等 Python 无法捕获的错误，还要保留 torchrun launcher 日志。
推荐在 tmux 中使用 hug 环境的 torchrun：

```bash
RUN=/root/code/vepfs/HUG-for-Recon-Gen/hand_recon/20260902_v6_canonical
mkdir -p "$RUN/logs/torchrun"
set -o pipefail
/root/code/vepfs/miniconda3/envs/hug/bin/torchrun \
  --log-dir "$RUN/logs/torchrun" --redirects 3 --tee 3 \
  --nproc_per_node=4 -m src.train --config configs/train_handrecon.yaml \
  2>&1 | tee "$RUN/logs/launcher.log"
```

`--tee 3` 同时保留终端和 worker 输出；`set -o pipefail` 确保 worker 失败时 shell
不会被 `tee` 错误地报告为成功。


## 5c. 109D shape 与中断续训

旧 PKL 中每个带训练标注的样本同时保存：

- grasp.shape：历史 HUG canonical beta，仅用于旧 99D/canonical 流程；
- grasp.shape_gt：DexYCB/HO3D 原始 subject-specific MANO beta，109D 训练目标使用该字段。

v30 不直接信任 PKL 中由旧 right-MANO 解码得到的几何，而是按样本名用 overlay 的 `params/joints/vertices/source_is_left` 覆盖。预测 beta 仍属于来源手的 basis：右手送入 MANO_RIGHT，左手送入 MANO_LEFT 后镜像输出。因此 109D flow、3D landmark 和 3D mesh loss 都会对正确的 shape 维度反向传播。HO3D 官方 evaluation split 没有 MANO 参数，因此评测时只使用其 joints/verts GT，不参与 109D 训练。

`source_is_left` 必须贯穿 train/val/test/inference 的几何解码，但它不是学习输入。checkpoint 会保存 `mano_geometry`；恢复时当前模型与 checkpoint convention 不同会报错。v29 及更早的 checkpoint 只能在 `legacy_right` 下恢复，v30 必须新开实验。

训练 checkpoint 的 model.pt 保存模型、optimizer、EMA（若已启动）、step、best score、
配置和归一化统计。model_best.pt 是验证集最优 checkpoint，可能早于最新 model.pt。

v11 当前最新安全 checkpoint：

    /root/code/vepfs/HUG-for-Recon-Gen/hand_recon/20260905_v11_canonical/model.pt

该文件保存于 step 6000。日志中 6001--6660 是尚未写入 checkpoint 的进程内 step，
因此恢复时从 6000 开始是预期行为。直接续训：

    cd /root/code/HUG-for-Recon-Gen
    /root/code/vepfs/miniconda3/envs/hug/bin/torchrun --nproc_per_node=4 -m src.train --config /root/code/vepfs/HUG-for-Recon-Gen/hand_recon/20260905_v11_canonical/config.yaml

v11 配置已设置 pretrained: null 和 resume: .../model.pt。代码在设置 resume 时会跳过
初始 pretrained 加载，避免重复读取。若从其他 checkpoint 恢复，只需在配置的
trainer.train.resume 中替换路径；trainer.train.total_steps 表示目标总 step，而不是
本次新增 step 数。

## 6. 已知注意事项

- 深度图单位 1mm uint16；HO3D 深度解码系数见 `scripts/README.md`。
- **DexYCB 左右手**：canonical-right 只统一图像/相机系 pose，不统一 MANO shape basis。v30 必须保留来源手侧，并使用 `NativeCanonicalMANO` 选择 MANO_LEFT/RIGHT；不要再把左手 beta 直接送进 MANO_RIGHT。
- 不要对左手 beta 取反、重排或套固定 10x10 变换；已经验证这不能等价转换 shape basis。也不要直接修改共享 `assets/mano/models/MANO_RIGHT.pkl` 的 shapedirs，否则会改变所有右手样本和旧实验的含义。
- 左手输入与输出各自只按 v30 约定反射一次。不要在 dataset、loss 或 eval 中额外镜像 joints/vertices；overlay 已处于 canonical-right 相机坐标系。
- **当前 v30 DexYCB-only 配置**使用 `dexycb_v4_fullres_shape_gt` + `dexycb_native_overlay_v1` + overlay 内的 `norm_stats.json`。三者必须成套使用，且 `mano_geometry` 必须为 `native_side_v1`。
- v30 不允许 resume v29 或其他 `legacy_right` checkpoint；只能把 HUG 权重作为 `pretrained` 后从 step 0 训练。不要用旧 normalization stats。
- overlay 会删除旧错误 mesh 产生的 mask/query，必须保持 `hand_crop.enabled: true` 并使用 detector crop。RGB/Depth、K、RTMPose 与 detector cache 可继续复用。
- 当前 v30 基线配置仍禁用 augmentation。另建增强配置后可以复用离线 detector/RTMPose cache；`AugmentedGraspDataset` 会把实际生效的 affine 同步到 bbox/keypoints，且 rejected-affine 不会误变换 cache。若以后新增 crop、perspective 或 horizontal flip 等其他几何增强，仍须为 cache 明确实现对应坐标变换，不能自动视为已支持。
- **HO3D eval 关节顺序**：官方 `evaluation_xyz.json` 是 MANO 原始运动学
  顺序 `[腕, 食×3, 中×3, 小×3, 环×3, 拇×3, 指尖×5(拇,食,中,环,小)]`，
  和本仓库的 manotorch 顺序 `[腕, 拇×4, 食×4, 中×4, 环×4, 小×4]` 不同。
  `GraspDataset` 加载时按 `HO3D_RAW_TO_STD` 重排（曾因顺序错位导致
  PA-MPJPE 虚高至 40mm ≈ 均值姿势基线）。转换产物存官方原始顺序，勿在
  转换脚本里排。
- HO3D 评估集（`--split evaluation` 产物）无 MANO 标注、无分割 mask，
  不在训练 loop 内，只用于第 4b 节的推理评测（`GraspDataset` 对空
  `object_mask` 已做兼容：有 `condition_point` 就不解码 mask）。
