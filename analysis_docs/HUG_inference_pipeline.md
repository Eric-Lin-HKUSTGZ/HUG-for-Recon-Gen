# HUG 推理 Pipeline 解析

> 分析对象：`src/inference.py`（批量推理）、`src/app.py`（交互式点击推理）、`src/models/grasp_model.py` / `grasp_flow.py`（采样）、`src/models/mano.py`（输出解码）。
> HUG 有两条推理入口，共享同一条核心链路：**checkpoint 加载 → 数据准备 → encode_scene（一次）→ 50 步 Euler ODE（轻量迭代）→ MANO 解码输出**。

## 0. 总体结构

```
checkpoint ──→ load_model（恢复 cfg + norm_stats + EMA 权重）
                    │
输入 pkl ──→ GraspDataset（与训练同一条数据处理 pipeline）
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
  批量 inference.py        交互 app.py（Viser 点击）
        │                       │ 点击像素 → 查深度 → 重建局部点云
        └───────────┬───────────┘
                    ▼
        model.sample：encode_scene（cond 只算一次）
                    → 50 步 Euler ODE（仅 3-token DiT 迭代）
                    → denormalize → (B,99)
                    ▼
        mano_params_to_grasp_dict（MANO 正算 → 完整 grasp dict）
                    ▼
        grasp_pred/{stem}.pkl（与 GT 同构的 schema）
```

## 1. Checkpoint 加载（`inference.py:50-143`）

三步，每步都有讲究：

**① 路径解析与格式统一**（`resolve_checkpoint_path` / `load_raw_checkpoint`）
- 目录输入按优先级找：`hug_full.safetensors` > `model_inference_bf16.pt` > `model.pt`；
- safetensors 格式下，cfg / norm_stats 存在 metadata JSON 中一并取出；`.pt` 直接 `torch.load`。

**② 配置自恢复**（`load_model`, L81-143）
- cfg 从 checkpoint 内嵌恢复（老格式回落到 `.hydra/config.yaml`）；
- norm_stats 从 ckpt 读取（详见 `HUG_norm_stats_analysis.md`）；
- 这是"结构开关与权重永远同源"的实现——`pcl_use_rgb`、`use_depth` 等全部跟随训练时配置，不会错配（见 `HUG_model_architecture.md` §3.3）。

**③ EMA 权重优先**（`use_ema=True` 默认）
- 加载 `ckpt["ema"]` 而非原始权重：论文 §4.2 "We keep an EMA from step 50K"——训练后期权重的滑动平均，推理更平滑稳定；
- 剥 DDP 的 `module.` 前缀、跳过 `n_averaged`；
- `model.eval()`，并把 `pcl_crop_radius` 挂到 model 上（交互 app 点击时按训练同款裁剪半径重建点云，见 §3）。

## 2. 批量推理主循环（`inference.py:215-286`）

**样本选择**：`sample_name`（单个 stem 或 .txt 清单）优先；否则 `rng(42)` 随机抽 `num_samples`（默认 256）——固定种子保证可复现。DataLoader：batch 32、4 workers。

**前向**：batch 数据按模型开关取用（`pcl_rgb` 仅在 `use_depth and pcl_use_rgb` 时传），bf16 autocast 下：

```python
samples = model.sample(point_uv, camera_K, steps=50,
                       rgb=rgb, pcl_xyz=pcl_xyz, pcl_rgb=pcl_rgb)
```

`sample` 内部（`grasp_model.py:243-257` → `grasp_flow.py:211-226`）：

```
encode_scene(point_uv, K, rgb, pcl)          ← 昂贵部分，只算 1 次
    → cond (B,256,1024)
x = randn(B,99)                              ← 归一化空间纯噪声
for i in 50..1:  x = x - DiT(x, t, cond)·dt  ← 50 步只在 (B,3,512) 的小 DiT 上迭代
x = denormalize(x)                           ← 回米制 (B,99)
```

**性能设计要点**：场景编码（DINOv2 + PointNeXt + 融合）每样本只算一次，50 步 ODE 迭代全在 3-token 轻量 DiT 上——推理快的关键。计时上首 batch 记为 warmup 排除统计（cuDNN 预热、内核编译）。

**逐样本保存**（L246-284）：

1. 重读原始 pkl 取元数据（object_name、frame_index、camera 等）；
2. `mano_params_to_grasp_dict`（`mano.py:110-164`）：99D → MANO 正算 → 组装**与 GT 完全同构**的 grasp dict：
   - `landmarks_3d` (21,3)（+t 平移到相机系）、`landmarks_2d` (21,2)（3D 经 K 投影）；
   - `mesh_vertices` (778,3)、`mesh_faces`；
   - `T_camera_wrist` (4×4 齐次矩阵)；
   - `pose`/`pose_6d`/`shape`/`R_6d`/`t`。
   - 同构意味着下游评估/可视化代码对 GT 和预测零区分；
3. **字段复用技巧**：点击点 (u,v)/224 归一化为 2 个 float32（8 字节）塞进 `object_mask` 字段——schema 无点击点字段，借此携带，`visualize_predictions` 解码还原点击标记；
4. 写 `grasp_pred/{stem}.pkl`，嵌套目录结构镜像输入（stem 为 root-relative 路径）。

## 3. 交互式推理（`app.py:304-360`）

Viser 网页应用：左 2D 图像、右 3D 场景，**点击图像任意像素 → 实时预测以该点为 query 的抓取**。`handle_click` 流程：

1. 点击归一化坐标 → 224 像素坐标 → 查 depth 图取 d（uint16 ÷ 1000 得米）；
2. **关键：点击时重建点云**——以点击点反投影的 3D 位置为球心，按 `model.pcl_crop_radius`（训练同款 0.3m）**重新**裁剪采样 4096 点，而非复用 pkl 自带点云。原因：训练时点云即以 query 点为中心裁剪，推理必须匹配该分布——点击位置变了，局部点云必须跟着变；
3. `model.sample` → `mano_params_to_grasp_dict` → 3D 场景画手 mesh + 点击处 1cm 标记小球；
4. `save_pred` 时每次点击存 `grasp_pred/<name>_<时间戳>.pkl`。

**与批量推理的关系**：批量版直接用 pkl 预存的 `condition_point` 和以其为中心构建的点云（dataloader 已按同逻辑处理）；交互版点击是任意位置，需在点击时刻现算深度与局部点云——两者逻辑等价，只是 query 点来源不同。

## 4. 两条推理路径对照

| 阶段 | 批量（inference.py） | 交互（app.py） |
|---|---|---|
| 权重 | EMA 优先，cfg/norm_stats 从 ckpt 恢复 | 同左 |
| query 点 | pkl 预存 `condition_point` | 用户实时点击 |
| 点云 | dataloader 预建（query 中心 0.3m 裁剪） | **点击时重建**（同参数） |
| 场景编码 | 每样本 1 次（DINOv2+PointNeXt+融合） | 同左 |
| ODE 采样 | 50 步 Euler，3-token DiT，bf16 autocast | 同左 |
| 输出 | `grasp_pred/{stem}.pkl`（MANO 全量字段） | 3D 可视化 + 可选存 pkl |
| 计时 | warmup 排除，ms/sample 统计 | — |

## 5. 一句话总结

> 推理 = **"重活一次，轻活五十次"**：场景编码（DINOv2/PointNeXt/融合）每样本只算一次得到条件序列，flow matching 的 50 步 Euler 迭代只在 3-token 微型 DiT 上进行，最后 denormalize 并经冻结 MANO 正算出与 GT 同构的完整抓取（骨架、mesh、手腕变换），写回 `grasp_pred/`；交互 app 额外在每次点击时按训练同款 0.3m 裁剪重建局部点云，保证输入分布一致。

## 附：关键代码位置

| 文件 | 位置 | 作用 |
|---|---|---|
| `src/inference.py:50-78` | `resolve_checkpoint_path` / `load_raw_checkpoint` | checkpoint 解析与格式统一 |
| `src/inference.py:81-143` | `load_model` | cfg/norm_stats 恢复、EMA 加载 |
| `src/inference.py:171-202` | 样本选择 + DataLoader | 随机子集（rng 42）/ 清单 |
| `src/inference.py:215-244` | 批量前向 | bf16 autocast、计时（warmup 排除） |
| `src/inference.py:246-284` | 逐样本保存 | 重读元数据、MANO 解码、写 pkl |
| `src/models/grasp_model.py:243-257` | `GraspFlowModel.sample` | encode_scene → flow.sample |
| `src/models/grasp_flow.py:211-226` | `GraspFlowMatching.sample` | 50 步 Euler ODE |
| `src/models/mano.py:110-164` | `mano_params_to_grasp_dict` | 99D → 完整 grasp dict |
| `src/app.py:304-360` | `handle_click` | 点击 → 深度查询 → 重建点云 → 推理 |
