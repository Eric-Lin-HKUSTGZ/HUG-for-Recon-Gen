# HUG-for-Recon-Gen 核心代码地图

## —— 完整代码位置与功能总结

### 一、项目结构总览（4,718 行 Python）

```
HUG-for-Recon-Gen/
├── src/
│   ├── __init__.py
│   ├── app.py                    # 565行 - 交互式抓取预测 Viser 应用
│   ├── inference.py              # 301行 - 批量推理管线
│   ├── prepare_inputs.py         # 262行 - 原始 RGB-D → .pkl 输入制作
│   ├── visualize_predictions.py  # 538行 - 3D 预测结果可视化
│   ├── dataloader/
│   │   ├── data_classes.py       #  59行 - GraspData / Grasp 数据类
│   │   └── grasp_dataset.py      # 328行 - PyTorch Dataset 加载器
│   ├── models/
│   │   ├── encoders.py           #  82行 - DINOv2 + PointNeXt 编码器
│   │   ├── pointnext.py          # 290行 - PointNeXt U-Net 点云编码
│   │   ├── fusion.py             # 215行 - PatchFusion (PointPainting)
│   │   ├── transformer.py        # 195行 - 自定义 Transformer Blocks
│   │   ├── grasp_flow.py         # 248行 - Flow Matching DiT 去噪器
│   │   ├── grasp_model.py        # 257行 - 顶层 GraspFlowModel
│   │   └── mano.py               # 230行 - MANO 手部模型层
│   └── utils/
│       ├── data_keys.py          #  16行 - 资源路径常量
│       ├── pcl_utils.py          # 132行 - 点云反投影/采样
│       ├── transform_utils.py    #  10行 - 6D 旋转→矩阵
│       ├── viser_utils.py        # 363行 - Viser 3D 场景工具
│       └── visualization_utils.py # 627行 - 2D/3D 可视化函数
├── assets/
│   ├── norm_stats.json           # 99D MANO 参数归一化统计量
│   ├── mano_rhand_shape.npy      # 固定手部 shape (10,)
│   └── mano_rhand_mesh_faces.npy # MANO mesh 面片 (1552, 3)
├── pyproject.toml                # setuptools 配置
├── environment.yaml              # conda 环境
└── README.md
```

---

### 二、文件逐一详解

#### 2.1 数据层 (`dataloader/`)

**[data_classes.py](HUG-for-Recon-Gen/src/dataloader/data_classes.py)** (59行)

| 类/结构 | 功能 |
|---------|------|
| `CameraIntrinsics` | 数据类：`K: (3,3)`, `width: int`, `height: int` |
| `GraspData` | 数据类：`object_name`, `frame_index`, `grasp_index`, `camera`, `camera_original`, `grasp`, `image`(JPEG bytes), `depth`(PNG bytes), `object_mask`(PNG bytes), `condition_point: (2,)`, `T_world_camera: (4,4)` |
| `Grasp` | 数据类：`pose: (1,15,3)`, `pose_6d: (1,15,6)`, `shape: (1,10)`, `landmarks_3d: (21,3)`, `landmarks_2d: (21,2)`, `T_camera_wrist: (4,4)`, `R_6d: (1,6)`, `t: (1,3)`, `mesh_vertices: (778,3)`, `mesh_faces: (1552,3)` |

**[grasp_dataset.py](HUG-for-Recon-Gen/src/dataloader/grasp_dataset.py)** (328行)

| 类/方法 | 功能 |
|---------|------|
| `GraspDataset.__init__` | 递归扫描 dataset_path 下所有 .pkl，排除 `grasp_pred/`。支持 `samples.txt` 过滤 |
| `__getitem__(idx)` | **核心数据流水线**：解码 JPEG→RGB, PNG→depth, PNG→mask；读取 condition_point；反投影深度为 XYZ；FPS 采样 4096 点；crop 半径 0.3m 内点云；ImageNet 归一化 RGB |
| `encode_scene()` | 将 depth + K 反投影为 metric XYZ |
| `get_original_for_viz()` | 返回原始分辨率 RGB + 真值 grasp 用于可视化 |

---

#### 2.2 模型层 (`models/`)

**[encoders.py](HUG-for-Recon-Gen/src/models/encoders.py)** (82行)

| 类 | 功能 |
|----|------|
| `DINOv2Encoder` | **冻结**的 `facebook/dinov2-with-registers-base`。`forward(rgb: B×3×224×224) → B×256×1024`（去掉 CLS + register tokens） |
| `PointNeXtEncoder` | **可训练**的 PointNeXt U-Net。`forward(xyz: B×N×3, rgb_pcl: B×N×3) → (B×256×D, B×256×3)` 返回特征 + 中心点坐标 |

**[pointnext.py](HUG-for-Recon-Gen/src/models/pointnext.py)** (290行)

| 组件 | 功能 |
|------|------|
| `PointNeXt` | 4 层 Set Abstraction + 4 层 Feature Propagation U-Net。半径：`[0.025, 0.05, 0.10, 0.20]` m。默认 `width=64` |
| SA 层 | `torch_cluster.fps` 最远点采样 + `torch_cluster.knn` 球查询 |
| FP 层 | 反向距离加权插值上采样 |

**[fusion.py](HUG-for-Recon-Gen/src/models/fusion.py)** (215行)

| 类/方法 | 功能 |
|---------|------|
| `FourierPosEmbed` | 随机傅里叶特征 + 线性旁路：XYZ(m) → sin/cos embedding |
| `PatchFusion` | **两条融合路径**：(1) PointPainting：PCL 中心投影到 RGB 平面→双线性采样 DINOv2→concat→MLP (2) 双流：RGB+PCL 独立投影→concat+模态嵌入 |
| `_project_centroids()` | 3D XYZ → 2D 归一化像素坐标 |
| `_paint()` | 用 `F.grid_sample` 实现 PointPainting |
| `forward()` | 输入 point_token + rgb_patches + depth_patches + depth_centroids + K → **512 fused tokens** |

**[transformer.py](HUG-for-Recon-Gen/src/models/transformer.py)** (195行)

| 类 | 功能 | 关键设计 |
|----|------|---------|
| `GeLUMLP` | 4× 扩展 FFN | GeLU 激活 |
| `Attention` | 多头自注意力 | 融合 QKV 投影 + **QK-norm** (RMSNorm) + **Flash Attention** (F.scaled_dot_product_attention) |
| `TransformerBlock` | Pre-norm 自注意块 | RMSNorm → Attn → residual → RMSNorm → MLP → residual |
| `CrossAttention` | 交叉注意力 | Q 来自 x, KV 来自 context。QK-norm + Flash Attention |
| `CrossAttentionBlock` | Pre-norm 交叉注意块 | |
| `AdaLNCrossAttnBlock` | **核心 DiT 模块** | 自适应 LayerNorm（时间步调制 7 参数：scale1/shift1/gate1/gate_cross/scale2/shift2/gate2）→ Self-Attn → Cross-Attn → FFN。**零初始化门控** |

**[grasp_flow.py](HUG-for-Recon-Gen/src/models/grasp_flow.py)** (248行)

| 类/方法 | 功能 |
|---------|------|
| `GraspFlow.__init__` | DiT 主干：`n_layers` 个 AdaLNCrossAttnBlock，`d_model=512`, `n_heads=8`。时间嵌入：Sin/Cos → MLP |
| `forward(x, t, condition_tokens)` | 预测速度场 `v(x, t, c)`。输入 x: (B, 99)，t: (B,)，condition_tokens: (B, 512, 512) |
| `sample(x_0, condition_tokens, steps=50)` | **Euler ODE 采样**：dt = 1/steps，逐步积分 `x_{t+dt} = x_t + v * dt` |

**[grasp_model.py](HUG-for-Recon-Gen/src/models/grasp_model.py)** (257行)

| 类/方法 | 功能 |
|---------|------|
| `GraspFlowModel.__init__` | 组装完整模型：DINOv2 + PointNeXt + PatchFusion + GraspFlow + MANO 层。从 cfg 读取所有超参 |
| `forward(batch)` | **训练前向**：编码→融合→FM 去噪→解码 |
| `predict(batch, steps)` | **推理前向**：随机噪声→FM 采样→MANO 解码 |
| `_extract_mano_params()` | 99D → `{t, R_6d, pose_6d}` |
| `_denormalize()` | 利用 `norm_stats.json` 反归一化 99D |
| `_decode_mano()` | 6D 旋转→轴角式→MANO forward→mesh+landmarks |

**[mano.py](HUG-for-Recon-Gen/src/models/mano.py)** (230行)

| 类/方法 | 功能 |
|---------|------|
| `MANOHandLayer` | 封装 `manotorch.ManoLayer`。右手，`ncomps=45`（PCA 姿态空间）。`center_idx=0`（手腕为中心） |
| `forward(pose_aa, shape)` | 轴角式 pose (B, 48) + shape (B, 10) → `{verts, joints, landmarks}` |
| `decompose()` | 99D → `{t, R_6d, pose_6d, shape}` |
| `gram_schmidt_6d()` | 用 `roma.special_gramschmidt` 转换 6D→旋转矩阵 |

---

#### 2.3 工具层 (`utils/`)

**[data_keys.py](HUG-for-Recon-Gen/src/utils/data_keys.py)** (16行)

```python
MANO_RIGHT_SHAPE_FILE    = PROJECT_ROOT / "assets" / "mano_rhand_shape.npy"
MANO_RIGHT_MESH_FACES_FILE = PROJECT_ROOT / "assets" / "mano_rhand_mesh_faces.npy"
NORM_STATS_FILE          = PROJECT_ROOT / "assets" / "norm_stats.json"
```

**[pcl_utils.py](HUG-for-Recon-Gen/src/utils/pcl_utils.py)** (132行)

| 函数 | 功能 |
|------|------|
| `backproject(depth, K)` | 深度图→相机坐标系 XYZ。过滤 Z>3m |
| `sample_fps(xyz, n)` | FPS 采样到 n 点 |
| `crop_pcl(xyz, rgb, center_xyz, radius)` | 以 center_xyz 为中心，radius 半径裁切点云 |

**[transform_utils.py](HUG-for-Recon-Gen/src/utils/transform_utils.py)** (10行)

```python
rotation_6d_to_matrix(r): roma.special_gramschmidt(r)  # (B,6) → (B,3,3)
```

**[viser_utils.py](HUG-for-Recon-Gen/src/utils/viser_utils.py)** (363行)

| 函数 | 功能 |
|------|------|
| `add_skeleton()`, `add_keypoints()` | Viser 中添加 MANO 骨架/关键点 |
| `add_mesh()` | 添加 MANO mesh（支持半透明） |
| `add_wrist_frame()` | 手腕坐标系 |
| `compose_scene()` | 组装完整 Viser 场景：点云 + 手 mesh + 骨架 |

**[visualization_utils.py](HUG-for-Recon-Gen/src/utils/visualization_utils.py)** (627行)

| 函数 | 功能 |
|------|------|
| `draw_2d_overlay()` | 在 RGB 图上叠加 2D landmarks + mask 边框 |
| `create_plotly_fig()` | 3D Plotly 交互可视化 |
| `render_side_by_side()` | GT vs 预测对比图 |

---

#### 2.4 应用层

**[prepare_inputs.py](HUG-for-Recon-Gen/src/prepare_inputs.py)** (262行)

**功能**：将原始相机输出转为推理用的 .pkl

```
输入: rgb.png + depth.png + intrinsics.txt
处理: 中心裁剪正方形 → resize 224×224 → 调整 K → JPEG/PNG 编码
输出: {image, depth, camera, camera_original, condition_point, T_world_camera}.pkl
```

**[inference.py](HUG-for-Recon-Gen/src/inference.py)** (301行)

**功能**：批量推理管线

| 函数 | 功能 |
|------|------|
| `resolve_checkpoint_path()` | 按优先级找 checkpoint：`hug_full.safetensors` → `model_inference_bf16.pt` → `model.pt` |
| `load_raw_checkpoint()` | 统一加载 .safetensors / .pt |
| `build_model_from_checkpoint()` | 从 checkpoint 中恢复 cfg + norm_stats，构建 GraspFlowModel |
| `run_inference()` | 遍历 dataset → 批次推理 → 保存 `grasp_pred/{name}.pkl` |

**数据流**：
```
.pkl 文件 → GraspDataset → DataLoader → GraspFlowModel.predict()
→ MANO 参数 (99D) → MANOHandLayer → mesh/landmarks
→ 保存回 GraspData.pkl
```

**[app.py](HUG-for-Recon-Gen/src/app.py)** (565行)

**功能**：基于 Viser 的交互式抓取预测 Web 应用

流程：
1. 加载数据集 + checkpoint
2. 启动 Viser 服务器 → 浏览器 3D 可视化
3. **用户点击物体** → 触发推理 → 显示 MANO mesh + 骨架
4. 支持多抓取采样（随机种子）

**[visualize_predictions.py](HUG-for-Recon-Gen/src/visualize_predictions.py)** (538行)

**功能**：离线 3D 可视化预测结果

---

### 三、完整数据流

```
┌──────────────────────────────────────────────────────────────┐
│                       推理管线                                │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  [输入]                   [编码]              [融合]          │
│  RGB 224×224 ──────→ DINOv2 (frozen) ──→ 256 patch tokens ─┐│
│                                                             ││
│  Depth 224×224 ──→ 反投影 XYZ ──→ FPS(4096) ──→ PointNeXt  ││
│                                         └──→ 256 PCL tokens─┤│
│                                                             ││
│  Condition Point ──→ 反投影 3D ──→ FourierPosEmbed ────────┤│
│                                      └──→ 1 point token ───┤│
│                                                             ││
│                              ┌──────────────────────────────┘│
│                              ▼                               │
│                    PatchFusion (PointPainting)                │
│                    512 fused tokens × d_model=1024           │
│                              │                               │
│                              ▼                               │
│  [Flow Matching Transformer]                                 │
│  x_noise (B,99) ──→ AdaLNCrossAttn × N_layers               │
│       ↑                    ↑                                │
│   time(t)          condition_tokens                          │
│       │                    │                                 │
│       └──── Sin/Cos MLP ───┘                                 │
│                              │                               │
│                              ▼                               │
│  [Euler ODE 采样, 50 steps]                                  │
│  v_pred ← GraspFlow(x_t, t, condition)                       │
│  x_{t+dt} = x_t + v_pred * dt                                │
│                              │                               │
│                              ▼                               │
│  [MANO 解码]                                                 │
│  99D → {t(3), R_6d(6), pose_6d(90)}                         │
│  roma.gram_schmidt: 6D → matrix                              │
│  manotorch.ManoLayer: pose_aa + shape → mesh + landmarks    │
│                                                              │
│  [输出]                                                      │
│  mesh_vertices (778,3) + landmarks_3d (21,3) + 所有参数      │
└──────────────────────────────────────────────────────────────┘
```

### 四、关键超参数与配置

| 参数 | 默认值 | 定义位置 |
|------|--------|---------|
| `image_size` | 224 | `grasp_model.py` |
| `encoder_name` | `facebook/dinov2-with-registers-base` | `encoders.py` |
| `n_points_input` | 4096 | `grasp_dataset.py` |
| `pcl_crop_radius` | 0.3m | `grasp_dataset.py` |
| `d_rgb_patch` | 1024 (DINOv2 hidden_size) | `encoders.py` |
| `d_depth_patch` | 256 (PointNeXt output_dim) | `pointnext.py` |
| `d_model` | 1024 (fusion) / 512 (FM transformer) | `fusion.py` / `grasp_flow.py` |
| `n_patches` | 256 | `fusion.py` |
| `n_layers` | 4 (fusion) / N (grasp_flow) | 各自模块 |
| `n_heads` | 8 | `transformer.py` |
| `use_pointpainting` | True | `fusion.py` |
| `use_rgb` / `use_depth` | True / True | `grasp_model.py` |
| `sampling_steps` | 50 | `grasp_flow.py` |
| MANO `ncomps` | 45 (PCA) | `mano.py` |
| 固定 shape beta | `[-2.37, -1.25, ...]` (10,) | `mano_rhand_shape.npy` |

### 五、99D MANO 参数结构

```
99D = [t(3) | R_6d(6) | pose_6d(90)]

t:        手腕平移 (相机坐标系, 米)
R_6d:     手腕旋转 (6D 连续表示 → roma.gram_schmidt → SO(3))
pose_6d:  15 个关节 × 6D = 90D
          (1 腕关节 + 14 指关节, 每关节 6D)

归一化: 按组 (translation/wrist_rot/finger_rot) 用 norm_stats.json 的 mean/std
目标空间: [0, 1] (经 mean/std 归一化后近似)
```

### 六、核心调用链

```
# 推理入口
inference.py:run_inference()
  ├── build_model_from_checkpoint()
  │     └── GraspFlowModel(cfg)  # 组装完整模型
  ├── GraspDataset(dataset_path, split='val')
  └── model.predict(batch, steps=50)
        ├── DINOv2Encoder(rgb) → rgb_patches (B,256,1024)
        ├── PointNeXtEncoder(xyz, pcl_rgb) → depth_patches (B,256,256)
        ├── PatchFusion(point, rgb_patches, depth_patches, centroids, K)
        │     └── _paint() → PointPainting 融合
        │     └── CrossAttentionBlock + TransformerBlock × 4
        │     → condition_tokens (B, 512, 1024)
        ├── GraspFlow.sample(noise, condition_tokens, 50)
        │     └── AdaLNCrossAttnBlock × N_layers  # 预测速度场
        │     └── Euler 积分 × 50 steps
        └── MANOHandLayer(pose_aa, shape)
              └── manotorch.ManoLayer → mesh, landmarks

# 数据处理入口
prepare_inputs.py → .pkl 文件
grasp_dataset.py:GraspDataset.__getitem__() → 解码 → 反投影 → 采样 → 归一化
```
