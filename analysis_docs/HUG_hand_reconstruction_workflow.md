# HUG 手部重建分支：模型架构、训练、测试与 Mesh 可视化

本文档说明当前 HUG 手部重建（hand reconstruction）分支的完整工作流。
它与原始 HUG 抓取生成任务不同：输入仍然是 RGB-D 图像和一个查询点，
但输出被监督为相机坐标系中的 MANO 手部姿态、21 个关节和 778 个网格顶点。

## 0. 默认工作目录

如果没有特别指定，本文及后续操作中的默认工作目录均为：

~~~text
/root/code/HUG-for-Recon-Gen
~~~

文中的 `src/`、`scripts/`、`configs/` 和 `assets/` 都相对于该目录。
大体积模型权重、训练输出和可视化结果仍保存在
`/root/code/vepfs/HUG-for-Recon-Gen/`；需要调用 HUG-HMILab 中的评测或
可视化脚本时，会明确写出其绝对路径和工作目录。

## 1. 代码与权重位置

重建训练代码、数据转换脚本和本文档都位于默认的 HUG-for-Recon-Gen
工作目录；HUG-HMILab 另外提供评测和可视化入口：

| 内容 | 路径 |
|---|---|
| 重建模型、数据集、训练入口 | src/ |
| DexYCB/HO3D 转换与划分脚本 | scripts/ |
| 重建训练配置 | configs/train_handrecon.yaml |
| HUG-HMILab 测试入口 | /root/code/HUG-HMILab/scripts/evaluate_hand_recon.py |
| HUG-HMILab mesh 可视化入口 | /root/code/HUG-HMILab/scripts/visualize_hand_recon.py |
| 已训练 checkpoint | /root/code/vepfs/HUG-for-Recon-Gen/hand_recon/20260902_v6_canonical/model_best.pt |
| DINOv2 原始/HuggingFace 权重 | /root/code/vepfs/HUG-for-Recon-Gen/dinov2/ |
| MANO 资产 | assets/mano/ 及 assets/mano_rhand_*.npy |

model_best.pt 内含训练时的模型配置和归一化统计量；冻结的 DINOv2
参数在保存时被剥离，加载时从本地 HuggingFace cache 或配置的 DINOv2
目录恢复。

## 2. 输入、输出与坐标约定

### 2.1 单个样本 pkl

转换后的样本由 GraspDataset 递归读取，主要字段如下：

- image：JPEG 编码的 224×224 RGB 图像；
- depth：PNG 编码的 uint16 深度，单位为毫米，和 RGB 对齐；
- camera.K：与 224×224 图像对应的 3×3 相机内参；
- condition_point：查询像素 (u, v)；深度从该点附近 15×15 窗口取中值；
- DexYCB 训练/测试样本额外带 grasp（MANO 参数和 GT mesh）；
- HO3D evaluation 样本不带 MANO 参数，而是带 joints_gt 和 verts_gt。

深度被反投影为相机坐标系米制点云。以查询点为中心裁剪半径
0.2 m 的球，再随机采样 4096 个点，同时保留每个点的 RGB 颜色：

~~~text
pcl_xyz : (B, 4096, 3), 单位 m，几何位置
pcl_rgb : (B, 4096, 3), [0, 1]，对应点的颜色特征
point_uv: (B, 3) = (u, v, depth_m)
~~~

所有训练和推理图像都在 224×224 空间内处理，内参随图像变换同步调整。
模型使用右手 MANO。当前转换产物 dexycb_v2_canonical_right 已统一为右手
坐标和关节顺序；HO3D 官方关节顺序在读入时使用 HO3D_RAW_TO_STD 重排为：

~~~text
[wrist, thumb×4, index×4, middle×4, ring×4, pinky×4]
~~~

### 2.2 99D MANO 状态

模型预测的状态向量为：

~~~text
q = [t(3), wrist_R6D(6), finger_pose_R6D(15×6=90)] ∈ R^99
~~~

- t 是相机坐标系中的腕部平移，单位米；
- wrist 和 15 个手指关节使用连续 6D 旋转表示；
- 6D 旋转在 MANO 解码时转为旋转矩阵，再转为轴角；
- 手型 betas 使用固定的 canonical 右手形状，MANO 层冻结。

## 3. 模型架构

~~~text
RGB (B,3,224,224)
      │
      └─ 冻结 DINOv2-Base/14 + registers
             └─ 256 个 patch token × 768

Depth → 相机坐标点云 (B,4096,3)
RGB   → 点云颜色 (B,4096,3)
      │
      └─ 可训练 PointNeXt U-Net
             ├─ depth_patches: (B,256,512)
             └─ centroids:     (B,256,3)

centroids ── K 投影 ──→ RGB patch 特征双线性采样（PointPainting）
                         │
                         └─ [painted 768 || depth 512] → 1024
                                    + 3D Fourier 位置编码
                                    + query point token
                                    └─ PatchFusion → cond (B,256,1024)

q ~ N(0,I) (B,99) + cond
      │
      └─ TokenPerGroupDiT，6 层 AdaLN + Cross-Attention
             ├─ translation token: 3 → 512 → 3
             ├─ wrist token:       6 → 512 → 6
             └─ finger token:     90 → 512 → 90
                    └─ 50 步 Euler flow matching 采样

denormalize(q) → 冻结 MANO → landmarks (B,21,3) + vertices (B,778,3)
~~~

### 3.1 RGB 编码器

src/models/encoders.py 中的 DINOv2Encoder 加载
facebook/dinov2-with-registers-base。224×224 输入产生 16×16 patch 网格；
去掉 CLS 和 4 个 register token 后保留 256×768 patch 特征。整个 DINOv2
分支在训练和推理中都处于 eval() + no_grad()，不更新参数。

### 3.2 点云编码器

PointNeXtEncoder 使用四级 PointNeXt 下采样和 Feature Propagation：

| 阶段 | 点数 | 特征宽度 | 半径 |
|---|---:|---:|---:|
| 输入/stem | 4096 | 64 | — |
| SA1 | 1024 | 128 | 0.025 m |
| SA2 | 256 | 256 | 0.05 m |
| SA3 | 64 | 512 | 0.10 m |
| SA4 | 16 | 1024 | 0.20 m |
| FP 回到 256 | 256 | 512 | — |

SA2 的 256 个中心点作为 centroids，既是几何 token 的位置，也是
PointPainting 的投影锚点。pcl_use_rgb=true 时，stem 输入为
[xyz, rgb] 六维；关闭该选项需要从头训练 3D-only checkpoint。

### 3.3 PointPainting 与 PatchFusion

每个 3D centroid 使用针孔模型投影到 RGB 图像：

~~~text
u = fx·X/Z + cx
v = fy·Y/Z + cy
~~~

在 DINOv2 的 16×16 特征图上对这些位置做双线性采样，得到
painted (B,256,768)。它与 depth_patches (B,256,512) 拼接后投影到
d_fusion=1024，再加入 3D Fourier 位置编码。

查询点 (u,v,d) 通过同一相机内参反投影为米制 (x,y,z)，编码成 query
token，并通过 cross-attention 调制 256 个场景 token。这样条件序列同时
保留局部几何、图像语义和查询位置。

### 3.4 Flow matching DiT

GraspFlowMatching 在归一化的 99D 状态空间中使用 rectified flow：

~~~text
x_t = (1-t)·x_0 + t·ε
v*  = ε - x_0
~~~

TokenPerGroupDiT 将平移、腕部旋转、手指旋转拆成 3 个 token，每个 token
维度为 512。每个 AdaLN block 依次执行：

1. 三个 MANO token 之间的 self-attention；
2. MANO token 对 256 个场景 token 的 cross-attention；
3. 条件调制的 FFN。

推理时从高斯噪声开始，按 t=1→0 进行 50 步 Euler 积分，最后反归一化为
米制 MANO 参数。

### 3.5 MANO 解码

src/models/mano.py 中的 MANO 层固定为右手、center_idx=0（腕部为原点）。
网络输出的平移加到 MANO 的 wrist-relative joints/vertices 上，得到相机坐标
系结果：

~~~text
landmarks_3d: (B,21,3)
vertices:     (B,778,3)
~~~

## 4. 重建训练流程

### 4.1 数据准备

在默认工作目录中执行：

~~~bash
cd /root/code/HUG-for-Recon-Gen
python scripts/convert_dexycb.py
python scripts/convert_ho3d.py
python scripts/make_handrecon_splits.py
python scripts/compute_norm_stats.py
~~~

实际训练配置使用以下划分文件：

~~~text
/root/code/vepfs/dataset/hand_recon_hug/splits_v2/
  dexycb_train.clean.txt
  dexycb_val.clean.txt
  dexycb_test.clean.txt
  ho3d_train.clean.txt
  ho3d_val.clean.txt
  ho3d_eval.clean.txt
~~~

归一化统计量只从训练集计算，并写入
assets/norm_stats_handrecon_v2.json。推理优先读取 checkpoint 内嵌的
norm_stats，避免训练和测试使用不同统计量。

### 4.2 单步训练前向与损失

训练批次先编码场景，再从真实 MANO 状态 x_0 采样随机时间 t 和噪声
ε，构造 x_t，由 DiT 预测速度。损失在 fp32 中计算：

~~~text
Lv  = MSE(pred_velocity, target_velocity)
L3D = mean[(1-t) · |pred_landmarks - GT_landmarks|_1]
L2D = mean[(1-t) · |project(pred_landmarks,K) - GT_2D|_1 / 224]

L = λv·Lv + λ3d·L3D + λ2d·L2D
~~~

当前混合 DexYCB + HO3D 配置中的主要值为：

~~~text
λv=1, λ3d=20, λ2d=1
AdamW, lr=1e-4, warmup=2500 steps
25,000 total steps, 每卡 batch_size=200
bf16=true, EMA 从 step 12,500 开始
每 1,000 steps 验证并保存 checkpoint
~~~

其中 L2D 只在样本带有 landmarks_2d 时启用；HO3D evaluation 没有训练
MANO 参数，因此只在测试阶段使用其 joints_gt/verts_gt 做评测。

### 4.3 启动完整训练或冒烟测试

~~~bash
cd /root/code/HUG-for-Recon-Gen

# 完整训练/续训
torchrun --nproc_per_node=4 -m src.train \
  --config configs/train_handrecon.yaml

# 冒烟测试：验证数据读取、前向、反向、验证和 checkpoint 写出
torchrun --nproc_per_node=4 -m src.train \
  --config configs/train_handrecon.yaml \
  --max-steps 30 --max-train-samples 20000
~~~

训练输出目录由配置中的 trainer.train.output_dir 指定。目录中包含
model.pt、model_best.pt、rank 日志和结构化 JSONL 日志。checkpoint
保存 model、可用时的 ema、优化器、配置和 norm stats；冻结的
image_encoder.* 不重复保存。

显存不足时优先降低配置中的每卡 batch_size，再降低 num_workers；
这不会改变推理接口，但会改变训练吞吐。

## 5. DexYCB 与 HO3D 测试评测

测试脚本针对两个数据集的 GT schema 分流：

- DexYCB：sample() 后通过 build_loss_dicts() 用 GT MANO 参数解码；
- HO3D evaluation：sample() 后用 canonical betas 解码，再和
  joints_gt/verts_gt 比较；
- 四卡评测使用无补齐的 StridedSampler，不会因分布式整除而重复尾部样本。

在 HUG-HMILab 中运行：

~~~bash
cd /root/code/HUG-HMILab
torchrun --nproc_per_node=4 --master_port=29517 \
  scripts/evaluate_hand_recon.py \
  --checkpoint /root/code/vepfs/HUG-for-Recon-Gen/hand_recon/20260902_v6_canonical/model_best.pt \
  --weights ema \
  --steps 50 \
  --batch-size 64 \
  --num-workers 4
~~~

结果写入：

~~~text
/root/code/vepfs/HUG-for-Recon-Gen/hand_recon/20260902_v6_canonical/test_results.json
~~~

脚本报告四个单位为毫米的指标：

- MPJPE：21 个关节的平均欧氏误差；
- PA-MPJPE：对关节做相似变换（旋转、平移、尺度）后的误差；
- MPVPE：778 个 MANO 顶点的平均欧氏误差；
- PA-MPVPE：对顶点做相似变换后的误差。

## 6. GPGFormer 风格 Mesh 可视化

重建结果不使用原始 HUG 抓取生成的六联图渲染器。
HUG-HMILab/scripts/visualize_hand_recon.py 为每个样本生成一张两栏 PNG：

左栏：

- 原始 RGB；
- GT mesh 和 Pred mesh 的 2D 投影；
- GT/Pred 21 个关节和骨架连线；
- 绿色表示 GT，红色表示 Pred。

右栏：

- 以各自 wrist 为原点的 root-relative 3D GT/Pred mesh；
- 3D 关节和骨架；
- 图标题显示 MPJPE、MPVPE、PA-MPJPE、PA-MPVPE。

生成每个数据集 16 张示例图：

~~~bash
cd /root/code/HUG-HMILab
/root/code/vepfs/miniconda3/envs/hug/bin/python \
  scripts/visualize_hand_recon.py \
  --checkpoint /root/code/vepfs/HUG-for-Recon-Gen/hand_recon/20260902_v6_canonical/model_best.pt \
  --output-dir /root/code/vepfs/HUG-for-Recon-Gen/hand_recon/20260902_v6_canonical/vis_mesh \
  --num-samples 16 \
  --batch-size 4 \
  --weights ema \
  --steps 50
~~~

输出结构：

~~~text
vis_mesh/
  dexycb/
    <sample>.png
  ho3d/
    <sample>.png
  summary.json
~~~

summary.json 保存可视化样本名和每张图对应的四项指标，便于后续归档或
筛选困难样本。脚本使用无显示器的 Agg backend；如果 hug 环境没有
matplotlib，会自动使用服务器已有的 pose 环境 site-packages。

## 7. 复现检查清单

1. 确认 checkpoint、DINOv2 和 MANO 文件可读；
2. 确认数据 pkl 与 split 文件的 stem 能一一对应；
3. 确认 pcl_crop_radius、n_points_input、图像尺寸与 checkpoint 配置一致；
4. 使用 weights=ema 和 steps=50 进行正式测试；
5. 检查 test_results.json 与 vis_mesh/summary.json 是否成功写出；
6. 通过可视化确认 HO3D 关节重排、相机投影方向和右手朝向正确。
