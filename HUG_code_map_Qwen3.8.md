# HUG-for-Recon-Gen 核心代码地图 (Qwen3.8)

## —— 完整代码位置与功能总结

> **项目**: Human Universal Grasping (HUG) — 人类通用抓取
> **论文**: [arXiv:2606.17054](https://arxiv.org/abs/2606.17054)
> **核心任务**: 输入单张 RGB-D 图像 + 用户点击物体 → 生成多样化的人手抓取姿态（MANO 参数）
> **技术路线**: 冻结 DINOv2（视觉语义）+ 可训练 PointNeXt（几何结构）→ PointPainting 多模态融合 → Flow Matching DiT 生成 → MANO 解码
> **总代码量**: 约 4,718 行 Python
> **当前状态**: 推理 + 可视化代码已开源；训练代码和 1M-HUGs 数据集尚未发布

---

### 一、项目结构总览

```
HUG/
├── src/
│   ├── __init__.py                    # 包标记（空文件）
│   ├── app.py                         # 565行 - 交互式抓取预测 Viser Web 应用
│   ├── inference.py                   # 301行 - 批量推理管线
│   ├── prepare_inputs.py              # 262行 - 原始 RGB-D 数据 → .pkl 输入文件制作
│   ├── visualize_predictions.py       # 538行 - 3D 预测结果离线可视化
│   ├── dataloader/
│   │   ├── __init__.py
│   │   ├── data_classes.py            #  59行 - GraspData / Grasp / CameraIntrinsics 数据类
│   │   └── grasp_dataset.py           # 328行 - PyTorch Dataset 数据加载器
│   ├── models/
│   │   ├── __init__.py
│   │   ├── encoders.py                #  82行 - DINOv2（冻结）+ PointNeXt（可训练）编码器
│   │   ├── pointnext.py               # 290行 - PointNeXt U-Net 点云编码器
│   │   ├── fusion.py                  # 215行 - PatchFusion 多模态融合（PointPainting）
│   │   ├── transformer.py             # 195行 - 自定义 Transformer 组件库
│   │   ├── grasp_flow.py              # 248行 - Flow Matching DiT 去噪器
│   │   ├── grasp_model.py             # 257行 - 顶层 GraspFlowModel 组装
│   │   └── mano.py                    # 230行 - MANO 手部模型封装层
│   └── utils/
│       ├── __init__.py
│       ├── data_keys.py               #  16行 - 资源文件路径常量
│       ├── pcl_utils.py               # 132行 - 点云反投影 / 采样 / 裁剪
│       ├── transform_utils.py          #  10行 - 6D 旋转表示 → SO(3) 矩阵
│       ├── viser_utils.py             # 363行 - Viser 3D 场景构建工具
│       └── visualization_utils.py     # 627行 - 2D / 3D 可视化函数
├── assets/
│   ├── norm_stats.json                # 99D MANO 参数的分组归一化统计量
│   ├── mano_rhand_shape.npy           # 固定的手部 shape 参数 (10,)
│   └── mano_rhand_mesh_faces.npy      # MANO 右手 mesh 三角面片 (1552, 3)
├── data/
│   ├── custom/                        # 自定义相机输入示例（ZED 2i）
│   │   ├── rgb.png
│   │   ├── depth.png
│   │   └── intrinsics.txt
│   └── hug_bench/                     # HUG-Bench 的 18 个样本 .pkl 文件
├── pyproject.toml                     # setuptools 打包配置（包名: hug）
├── environment.yaml                   # conda 环境配置
└── README.md                          # 项目说明
```

---

### 二、各文件逐一详解

#### 2.1 数据层（`src/dataloader/`）

##### [data_classes.py](src/dataloader/data_classes.py)（59行）— 数据结构定义

| 类 | 字段 | 说明 |
|----|------|------|
| `CameraIntrinsics` | `K (3,3)`, `width`, `height` | 相机内参数据类 |
| `GraspData` | `object_name`, `frame_index`, `grasp_index`, `camera`, `camera_original`, `grasp`, `image`(JPEG字节), `depth`(PNG字节), `object_mask`(PNG字节), `condition_point (2,)` | 单帧抓取数据，图像均以编码字节存储以减小文件体积 |
| `Grasp` | `pose (1,15,3)`, `pose_6d (1,15,6)`, `shape (1,10)`, `landmarks_3d (21,3)`, `landmarks_2d (21,2)`, `T_camera_wrist (4,4)`, `R_6d (1,6)`, `t (1,3)`, `mesh_vertices (778,3)`, `mesh_faces (1552,3)` | 单次右手抓取的全部标注数据 |

##### [grasp_dataset.py](src/dataloader/grasp_dataset.py)（328行）— PyTorch 数据集

| 方法 | 功能 |
|------|------|
| `__init__` | 递归扫描 `dataset_path` 下所有 `.pkl` 文件（排除 `grasp_pred/` 输出目录）；支持 `samples.txt` 子集过滤和 `indices` 指定 |
| `find_pkls()` | 静态方法：递归 glob 所有 `.pkl`，排除预测输出目录 |
| `_load_file_list()` | 解析样本列表：优先读 `samples.txt`，否则全局扫描并缓存为 `samples.txt` |
| `__getitem__(idx)` | **核心数据流水线**：① pickle 加载 .pkl → ② cv2 解码 JPEG→RGB、PNG→depth(uint16→米)、PNG→mask → ③ 从 mask 中随机采样条件点（先腐蚀 mask 确保深度有效）→ ④ 深度反投影为 XYZ 点云 → ⑤ 以条件点为中心 0.3m 半径裁切 → ⑥ 随机采样到 4096 点 → ⑦ ImageNet 归一化 RGB → ⑧ 返回 `{point_uv, camera_K, stem, mano_params, rgb, pcl_xyz, pcl_rgb}` |
| `_sample_point_from_mask()` | 从物体 mask 中随机采样一个像素：先腐蚀 mask 取有效深度区域，若无则回退到原始 mask，再不行用深度均值 |
| `_build_pcl()` | 组装点云：depth + rgb + K → 反投影 → 可选裁剪 → 固定点数采样 → tensor |
| `get_inference_data(stem)` | 加载推理所需最小数据：RGB、depth、camera、mesh_faces、预构建的 PCL |
| `get_original_for_viz(idx)` | 加载原始分辨率数据用于 3D 可视化 |
| `_decode_image / _decode_mask / _decode_depth_uint16` | 字节解码辅助函数 |

---

#### 2.2 模型层（`src/models/`）

##### [encoders.py](src/models/encoders.py)（82行）— 双编码器

| 类 | 功能 | 关键细节 |
|----|------|---------|
| `DINOv2Encoder` | **冻结**的 ViT 编码器，加载 `facebook/dinov2-with-registers-base` | 去掉 CLS token 和 4 个 register tokens，输出 `(B, 256, 1024)` 的 patch 特征；`output_dim=1024`；`requires_grad=False`，`model.eval()` |
| `PointNeXtEncoder` | **可训练**的点云编码器，封装 PointNeXt U-Net | 输入 `xyz (B,N,3)` + `rgb_pcl (B,N,3)`，输出 `features (B,256,512)` + `centroids (B,256,3)`；`output_dim = 8*c = 512`（c=64 时） |

##### [pointnext.py](src/models/pointnext.py)（290行）— PointNeXt 点云 U-Net

| 组件 | 功能 |
|------|------|
| `_shared_mlp(dims)` | 通用 MLP 构建器：`Linear → LayerNorm → GeLU` 堆叠 |
| `_fps_indices(xyz, k)` | 批量最远点采样（FPS），使用 `torch_cluster.fps`，`random_start=True` |
| `_knn_indices(query, ref, k)` | 批量 kNN 邻域查询，使用 `torch_cluster.knn` |
| `SetAbstraction` | **SA 层**：FPS 选取质心 → kNN 球查询 → 相对位置按 PointNeXt Eq.2 除以球半径归一化 → edge MLP → max-pool。当某质心球内无邻居时，回退到 kNN 最近邻 |
| `LocalAggregation` | 同分辨率 kNN + 1 层 edge MLP + max-pool（InvResMLP 内部使用） |
| `InvResMLP` | **倒残差 MLP 块**（PointNeXt Sec 3.2.2）：LocalAggregation → 2 层 pointwise MLP（4× 扩展，倒瓶颈）→ 残差连接 + GeLU |
| `FeaturePropagation` | **FP 层**：3-NN 反距离加权插值 → 与 skip connection concat → MLP |
| `PointNeXt` | **4 层 SA + 4 层 FP 的 U-Net 主结构**。SA 半径 `[0.025, 0.05, 0.10, 0.20]` m，质心数 `[1024, 256, 64, 16]`，每层 k=`[32,32,32,16]`。默认 `width=64`，`blocks=(1,2,1,1)` = PointNeXt-B。InvResMLP 半径放大 2×（因下采样后需要更大感受野）。输出：FP3 阶段的 `(B, 256, 8c)` 特征 + 对应质心坐标 |


##### [fusion.py](src/models/fusion.py)（215行）— 多模态融合（核心创新）

| 组件 | 功能 |
|------|------|
| `FourierPosEmbed` | **随机傅里叶位置编码**：`sin(2π·coord·B) ∥ cos(2π·coord·B)` + 可学习的线性旁路。支持 3D（米制坐标）和 2D（归一化像素坐标）。`scale` 参数控制 B 的标准差（最短可分辨波长 ≈ 1/scale） |
| `PatchFusion` | **核心融合模块**，两条路径可切换 |
| `_project_centroids()` | 3D 质心 `(B,N,3)` → 2D 归一化 `[-1,1]` 像素坐标，使用相机内参 K 做纯几何投影 |
| `_paint()` | **PointPainting 核心操作**：将 PCL 质心投影到 RGB 平面 → 用 `F.grid_sample` 双线性采样 DINOv2 patch 特征图 → 返回各质心处"涂抹"的 RGB 特征 |
| `forward()` | **路径 1（use_pointpainting=True，默认）**：paint DINOv2 特征 → concat PCL 特征 → MLP 融合为 256 token × d_model → 加 3D 位置嵌入 → 条件点交叉注意力 → 4 层 Transformer 自注意力精炼。**路径 2（双流）**：RGB patches + PCL tokens 各自线性投影 + 位置嵌入 + 模态嵌入 → concat 为 512 token → 条件点交叉注意力 → Transformer |


##### [transformer.py](src/models/transformer.py)（195行）— Transformer 组件库

| 类 | 功能 | 关键设计 |
|----|------|---------|
| `GeLUMLP` | 4× 扩展的前馈网络 | `Linear→GeLU→Dropout→Linear→Dropout`，无 bias |
| `Attention` | 多头自注意力 | **融合 QKV 投影**（单次 Linear 出 3×d_model）、**QK-norm**（对每个头的 Q/K 做 RMSNorm）、**Flash Attention**（`F.scaled_dot_product_attention`） |
| `TransformerBlock` | Pre-norm 自注意力块 | `RMSNorm → Attn → 残差 → RMSNorm → MLP → 残差` |
| `CrossAttention` | 交叉注意力 | Q 来自 x，KV 来自 context。同样 QK-norm + Flash Attention。支持 `attn_mask` |
| `CrossAttentionBlock` | Pre-norm 交叉注意力块 | `x = x + CrossAttn(Norm(x), Norm(ctx)) + MLP(Norm(x))` |
| `AdaLNCrossAttnBlock` | **DiT 核心模块** | 时间步嵌入通过 `adaLN_modulation`（SiLU→Linear）输出 **7 个调制参数**：`scale1, shift1, gate1`（自注意力）、`gate_cross`（交叉注意力门控）、`scale2, shift2, gate2`（FFN）。所有门控**零初始化**（训练初期等价于恒等映射）。结构：AdaLN Self-Attn → Cross-Attn → AdaLN FFN |


##### [grasp_flow.py](src/models/grasp_flow.py)（248行）— Flow Matching 生成器

| 类/方法 | 功能 |
|---------|------|
| `SinusoidalPosEmb` | Sin/Cos 时间步位置编码：`t → [sin(w_i·t), cos(w_i·t)]`，`w_i = 1/10000^(2i/d)` |
| `TokenPerGroupDiT` | **分组 token DiT 主干**。输入 99D 按语义拆为三组：`translation(3) / wrist(6) / fingers(90)`，各组独立投影为 d_model=512 的 token → 堆叠 3 个 token + 可学习的 `token_type_emb` → N 层 `AdaLNCrossAttnBlock`（以时间嵌入 c 为调制信号，以 condition tokens 为交叉注意力上下文）→ 最终调制（SiLU→Linear 出 scale/shift）→ RMSNorm → 三个独立输出头投影回 3/6/90 维 → concat。所有输出头**零初始化**。`normalize()/denormalize()` 按三组分别使用 norm_stats 做归一化 |
| `GraspFlowMatching` | **Rectified Flow Matching 封装**。`d_mano=99`, `d_model=512`, `sampling_steps=50` |
| `forward(x_start, cond)` | **训练前向**：`t ~ U(0,1)`，`x_norm = normalize(x_start)`，`x_t = (1-t)·x_norm + t·ε`（线性插值），目标速度 `v = ε - x_norm`，预测 `v_pred = DiT(x_t, t, cond)`。返回 `{pred, target, t, x_noisy}` |
| `sample(cond, steps=50)` | **Euler ODE 推理采样**：`x_1 ~ N(0,I)`，从 `t=1` 到 `t=0` 以 `dt=1/steps` 逐步积分 `x_{t-dt} = x_t - v_pred·dt`，最后 denormalize |
| `recover_x0(output)` | 从训练输出恢复 x0：`x0 = x_t - t·v_pred` |


##### [grasp_model.py](src/models/grasp_model.py)（257行）— 顶层模型组装

| 方法 | 功能 |
|------|------|
| `__init__(cfg, norm_stats)` | 从配置读取所有超参，组装 DINOv2 + PointNeXt + PatchFusion + GraspFlowMatching + MANO。处理 `use_rgb/use_depth/use_pointpainting/use_2d_point` 之间的依赖约束和强制修正。加载固定 shape beta 和 mesh faces |
| `encode_scene(point_uv, K, rgb, pcl_xyz, pcl_rgb)` | **场景编码**：条件点反投影为 metric XYZ → DINOv2(RGB) → PointNeXt(PCL) → PatchFusion 融合 → 返回条件 tokens |
| `forward(batch)` | **训练前向**：encode_scene → flow matching 训练 → recover_x0 → MANO 解码 → 构建 `{preds, targets, time_weight}` |
| `sample(point_uv, K, rgb, pcl_xyz)` | **推理前向**（no_grad）：encode_scene → flow matching 采样 → 返回 99D MANO 参数 |
| `mano_forward(mano_params, betas)` | MANO 解码 + 将 landmarks 从手腕坐标系变换到相机坐标系 |
| `_backproject(point_uv, K)` | 纯几何操作：`(u,v,d) + K → (x,y,z)`，K 不参与学习 |
| `_build_dicts()` | 从预测和目标 MANO 参数构建包含归一化参数和 3D landmarks 的字典 |
| `build_loss_dicts()` | 验证时从采样结果构建损失字典 |


##### [mano.py](src/models/mano.py)（230行）— MANO 手部模型

| 类/函数 | 功能 |
|---------|------|
| `MANO.__init__` | 封装 `manotorch.ManoLayer`：右手、`center_idx=0`（手腕为原点）、`ncomps=45`（PCA 姿态空间）、`flat_hand_mean=True`。MANO 参数冻结 |
| `decode_mano_params(mano_params)` | 99D → `{t (B,3), R_6d (B,6), pose_6d (B,15,6)}` |
| `forward(mano_params, betas)` | ① 6D 旋转 → `roma.special_gramschmidt` → SO(3) 矩阵 → ② `manotorch.rotation_to_axis_angle` → 轴角式 → ③ 拼接手腕旋转(3) + 手指姿态(45) = 48D pose_coeffs → ④ `ManoLayer(pose_coeffs, betas)` → `{landmarks_3d, vertices, t, R_3x3, pose_3x3}` |
| `mano_params_to_grasp_dict()` | 模型输出 → 完整 Grasp 格式字典（numpy），用于保存预测结果 |
| `mano_params_to_animation()` | 生成预抓取→抓取的线性插值动画序列：手腕从偏后方接近，手指从张开（thumb 预弯曲）lerp 到预测姿态 |
| `project_3d_to_2d()` | `(N,3) @ K^T → (N,2)` |


---

#### 2.3 工具层（`src/utils/`）

##### [data_keys.py](src/utils/data_keys.py)（16行）— 路径常量

```python
MANO_MODELS_FOLDER        = PROJECT_ROOT / "assets" / "mano"
MANO_RIGHT_MESH_FACES_FILE = PROJECT_ROOT / "assets" / "mano_rhand_mesh_faces.npy"
MANO_RIGHT_SHAPE_FILE     = PROJECT_ROOT / "assets" / "mano_rhand_shape.npy"
NORM_STATS_FILE           = PROJECT_ROOT / "assets" / "norm_stats.json"
```

##### [pcl_utils.py](src/utils/pcl_utils.py)（132行）— 点云处理

| 函数 | 功能 |
|------|------|
| `backproject_to_pcl(depth_m, rgb, K)` | 深度图→相机坐标系 XYZ + RGB。过滤 `Z≤0` 或 `Z>3m`。支持以 `center` 为中心、`crop_radius` 为半径的球体裁剪 |
| `sample_fixed_n(xyz, rgb, n)` | 随机采样到恰好 n 个点（多于则不放回抽取，少于则放回重复） |
| `pixel_to_xyz(u, v, depth, K)` | 单像素反投影：`x=(u-cx)*d/fx, y=(v-cy)*d/fy, z=d` |
| `depth_to_pcl_tensors()` | 端到端：depth + rgb + K（numpy 或 torch）→ `(xyz_tensor, rgb_tensor)`，RGB 归一化到 [0,1] |

##### [transform_utils.py](src/utils/transform_utils.py)（10行）— 旋转转换

```python
def six_d_to_rotation_matrix(r):  # (B,6) → (B,3,3)
    return roma.special_gramschmidt(r.reshape(*r.shape[:-1], 3, 2))
```

##### [viser_utils.py](src/utils/viser_utils.py)（363行）— Viser 3D 场景

| 函数/类 | 功能 |
|---------|------|
| `add_hand_skeleton()` | 在 Viser 场景中添加 MANO 骨架线段（21 个关键点之间的连接） |
| `add_hand_keypoints()` | 添加 21 个关键点小球 |
| `add_mano_mesh()` | 添加 MANO mesh（支持半透明 + 颜色） |
| `add_wrist_frame()` | 添加手腕坐标系（RGB 三轴） |
| `backproject_depth_to_point_cloud()` | 深度图反投影 + 降采样为 Viser 点云 |
| `make_clickable_image_html()` | 生成含 JS 点击桥接的 HTML（用于 Web 端点击选点） |
| `PredictionStore` | 管理多个预测结果：颜色轮换、批量显隐/透明度控制、clear-last/clear-all |

##### [visualization_utils.py](src/utils/visualization_utils.py)（627行）— 可视化

| 函数 | 功能 |
|------|------|
| `draw_point_marker()` | 在 RGB 图上绘制同心圆 + 光晕的点击标记 |
| `draw_mask_overlay()` | 半透明 mask 叠加 + 边框 |
| `draw_mano_mesh()` | 带深度排序的 3D mesh 渲染（骨架+关键点+指甲） |
| `create_3d_plotly()` | 交互式 Plotly 3D 可视化 |
| `create_prediction_visualization()` | GT vs 预测并排对比图 + 损失/手腕指标叠加 |

---

#### 2.4 应用层（`src/`）

##### [prepare_inputs.py](src/prepare_inputs.py)（262行）— 原始数据转 .pkl

**入口**: `python -m hug.prepare_inputs --dataset-path data/custom`

**功能**: 将用户自采的 RGB + depth + intrinsics 转为模型可读的 .pkl 文件

**处理流程**:
1. 自动检测文件名（`*rgb*`, `*depth*`, `*intrinsics*`）
2. 读取 intrinsics（支持 `fx fy cx cy` 四元组 / 3×3 矩阵 / .npy / .json）
3. 中心裁剪为正方形（短边）→ resize 为 224×224
4. 同步调整相机内参 K
5. RGB 编码为 JPEG bytes、depth 编码为 PNG bytes（uint16 毫米单位）
6. 写入 `{stem}.pkl`（不含抓取标注，仅推理用）

##### [inference.py](src/inference.py)（301行）— 批量推理

**入口**: `python -m hug.inference --checkpoint-path checkpoints/ --dataset-path data/hug_bench/`

| 函数 | 功能 |
|------|------|
| `resolve_checkpoint_path()` | 按优先级查找：`hug_full.safetensors` → `model_inference_bf16.pt` → `model.pt` |
| `load_raw_checkpoint()` | 统一加载 .safetensors / .pt，.safetensors 额外解析 JSON metadata |
| `load_model()` | 从 checkpoint 恢复 cfg + norm_stats → 构建 `GraspFlowModel` → 加载 EMA 或原始权重 → eval 模式 |
| `main()` | 遍历 dataset → DataLoader(batch=32) → `model.sample()` → `mano_params_to_grasp_dict()` → 保存 `grasp_pred/{stem}.pkl`。支持 `--sample-name` 子集过滤 / `--num-samples` 随机采样。使用 Rich Live Table 实时显示逐 batch 推理耗时 |

**推理数据流**:
```
.pkl → GraspDataset → DataLoader → model.sample()
→ 99D MANO 参数 → mano_params_to_grasp_dict()
→ mesh(778,3) + landmarks(21,3) → 保存 GraspData.pkl
```

##### [app.py](src/app.py)（565行）— 交互式 Web 应用

**入口**: `python -m hug.app --checkpoint-path checkpoints/ --dataset-path data/hug_bench/ --save-pred`

**流程**:
1. 加载模型 + 数据集
2. 启动 Viser 服务器（端口 8080）→ 浏览器 3D 可视化
3. 显示深度点云 + RGB 图像（侧边栏）
4. **用户点击物体像素** → 触发推理 → 显示 MANO mesh（半透明）+ 骨架 + 关键点
5. 支持预抓取动画（手腕从偏后方接近 + 手指从张开 lerp 到抓取）、多抓取叠加显示（不同颜色）
6. `--save-pred` 模式下每次点击保存一个时间戳命名的 .pkl

**GUI 控制面板**:
- **Hand**: 显隐 / 透明度 / 动画开关+时长 / 预抓取偏移 / 条件点显示 / 清除
- **Image**: 样本下拉切换 / 前后导航
- **Visibility**: 相机视锥显隐 / 深度点云大小 / 最大深度裁剪

##### [visualize_predictions.py](src/visualize_predictions.py)（538行）— 离线可视化

**入口**: `python -m hug.visualize_predictions --dataset-path data/hug_bench/`

**功能**: 加载 `grasp_pred/` 中的预测 .pkl，在 Viser 3D 场景中显示绿色预测手部 mesh + 骨架 + 关键点 + 条件点标记 + 相机视锥

---

### 三、完整数据流（推理管线）

```
┌──────────────────────────────────────────────────────────────────┐
│                         HUG 推理管线                               │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─── 输入 ──────────────────────────────────────────────────┐   │
│  │  RGB 224×224        Depth 224×224        条件点 (u,v,d)   │   │
│  │   (3通道)            (uint16 mm)          (用户点击像素)    │   │
│  └──────────┬────────────────┬───────────────────┬───────────┘   │
│             │                │                   │               │
│             ▼                ▼                   ▼               │
│  ┌──────────────┐  ┌──────────────────┐  ┌──────────────────┐   │
│  │ DINOv2       │  │ 反投影 → XYZ     │  │ 反投影 → (x,y,z) │   │
│  │ (冻结 ViT)   │  │ → 随机采样4096点 │  │ (纯几何, K不用   │   │
│  │              │  │ → 0.3m 球体裁剪 │  │  于学习权重)     │   │
│  └──────┬───────┘  └────────┬─────────┘  └────────┬─────────┘   │
│         │ 256 patch tokens  │ 256 PCL tokens      │ 1 point     │
│         │ (B,256,1024)      │ (B,256,512)         │ token       │
│         │                   │ + centroids (B,256,3)│             │
│         └───────────────────┼─────────────────────┘             │
│                             ▼                                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              PatchFusion (PointPainting 融合)             │   │
│  │                                                          │   │
│  │  1. PCL 质心 ──投影──→ 2D 像素坐标 (用 K)                │   │
│  │  2. 在 DINOv2 patch 特征图上双线性采样"涂抹"RGB 特征      │   │
│  │  3. [painted_RGB(1024D) ∥ PCL_feat(512D)] → MLP → 256   │   │
│  │     fused tokens × 1024D                                 │   │
│  │  4. 加 3D Fourier 位置嵌入                                │   │
│  │  5. 条件点 Cross-Attention → 条件信息注入                 │   │
│  │  6. Transformer × 4 层自注意力精炼                        │   │
│  └────────────────────────┬─────────────────────────────────┘   │
│                           │ condition_tokens (B,256,1024)       │
│                           ▼                                      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │          Flow Matching DiT (TokenPerGroupDiT)             │   │
│  │                                                          │   │
│  │  99D 噪声 → 分组投影为 3 个 token:                        │   │
│  │    [translation(3)] [wrist_rot(6)] [fingers(90)]         │   │
│  │    → 各自 Linear → d_model=512 → + token_type_emb        │   │
│  │                                                          │   │
│  │  AdaLNCrossAttnBlock × N_layers:                         │   │
│  │    ├─ 时间嵌入 c (Sin/Cos → MLP) → 7 个调制参数          │   │
│  │    │   {scale1,shift1,gate1, gate_cross,                 │   │
│  │    │    scale2,shift2,gate2}                              │   │
│  │    ├─ AdaLN Self-Attention (3 token 之间)                │   │
│  │    ├─ Cross-Attention → condition_tokens                 │   │
│  │    └─ AdaLN FFN                                          │   │
│  │                                                          │   │
│  │  最终调制 → 三组输出头 → concat → 99D 速度场 v_pred      │   │
│  └────────────────────────┬─────────────────────────────────┘   │
│                           │                                      │
│                           ▼                                      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              Euler ODE 采样 (50 步)                       │   │
│  │                                                          │   │
│  │  x_1 = randn(B, 99)         # t=1: 纯噪声                │   │
│  │  for i = 49, 48, ..., 0:    # 从 t=1 积分到 t=0         │   │
│  │      t = (i+1) / 50                                      │   │
│  │      v = DiT(x_t, t, condition)  # 预测速度场            │   │
│  │      x_{t-dt} = x_t - v * dt    # Euler 步进             │   │
│  │  denormalize(x_0) → 99D MANO 参数                        │   │
│  └────────────────────────┬─────────────────────────────────┘   │
│                           │                                      │
│                           ▼                                      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    MANO 解码                              │   │
│  │                                                          │   │
│  │  99D → t(3) + R_6d(6) + pose_6d(15×6)                   │   │
│  │  roma.special_gramschmidt: 6D → SO(3) 旋转矩阵           │   │
│  │  manotorch.rotation_to_axis_angle: matrix → axis-angle   │   │
│  │  ManoLayer(pose_aa(48), shape(10)):                      │   │
│  │    → mesh vertices (778, 3)   # 手部 mesh                │   │
│  │    → joints (21, 3)           # 关键点                   │   │
│  └────────────────────────┬─────────────────────────────────┘   │
│                           │                                      │
│                           ▼                                      │
│  ┌─── 输出 ──────────────────────────────────────────────────┐   │
│  │  • mesh_vertices (778,3)    手部三角 mesh                 │   │
│  │  • landmarks_3d (21,3)      MANO 关键点                  │   │
│  │  • landmarks_2d (21,2)      2D 投影                      │   │
│  │  • T_camera_wrist (4,4)     手腕位姿                      │   │
│  │  • pose (1,15,3)           轴角式姿态                     │   │
│  │  • pose_6d (1,15,6)        6D 姿态                       │   │
│  └──────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

---

### 四、99D MANO 参数结构

```
99D = [t(3) | R_6d(6) | pose_6d(90)]

┌────────────┬──────┬──────────────────────────────────────────┐
│  分量       │ 维度  │  说明                                    │
├────────────┼──────┼──────────────────────────────────────────┤
│ t          │  3   │  手腕平移向量（相机坐标系，单位：米）       │
│ R_6d       │  6   │  手腕旋转的 6D 连续表示                    │
│ pose_6d    │ 90   │  15 个关节 × 6D（1 腕关节 + 14 指关节）   │
└────────────┴──────┴──────────────────────────────────────────┘

归一化方式:
  - 分三组独立归一化（translation / wrist_rot / finger_rot）
  - 每组使用 assets/norm_stats.json 中的 mean 和 std
  - 归一化后近似在 [0, 1] 范围内
  - TokenPerGroupDiT 内部按组投影为 3 个独立 token

固定参数:
  - shape beta: [-2.37, -1.25, -2.05, -0.85, 1.66, -1.35, -1.85, -0.67, -1.69, -1.21]
    （所有抓取共享同一手型，来自 mano_rhand_shape.npy）
```

---

### 五、关键超参数速查表

| 参数 | 默认值 | 所在文件:行号 |
|------|--------|-------------|
| 图像分辨率 `image_size` | 224 | `grasp_dataset.py:49`, `fusion.py:78` |
| DINOv2 模型 | `facebook/dinov2-with-registers-base` | `encoders.py:16` |
| 输入点云点数 `n_points_input` | 4096 | `grasp_dataset.py:53` |
| 点云裁剪半径 `pcl_crop_radius` | 0.3 m | `grasp_dataset.py:54` |
| PointNeXt 宽度 `c` | 64 | `pointnext.py:213` |
| SA 半径 `sa_radii` | (0.025, 0.05, 0.10, 0.20) m | `pointnext.py:214` |
| SA 质心数 | (1024, 256, 64, 16) | `pointnext.py:228` |
| InvResMLP 块数 `blocks` | (1, 2, 1, 1) = PointNeXt-B | `pointnext.py:215` |
| PointNeXt 输出维度 | 256 token × 512D (8c) | `pointnext.py:258` |
| DINOv2 patch 数 / 维度 | 256 × 1024 | `encoders.py:22-23` |
| 融合后维度 `d_fusion` | 1024 | `fusion.py:58` |
| 融合 token 数 `n_patches` | 256 | `fusion.py:59` |
| 融合 Transformer 层数 | 4 | `fusion.py:61` |
| FM DiT 维度 `d_model` | 512 | `grasp_flow.py:174` |
| FM DiT 层数 `n_layers` | 6 | `grasp_flow.py:178` |
| 注意力头数 `n_heads` | 8 | 各 transformer 模块 |
| Dropout | 0.1 | 各模块 |
| Flow Matching 采样步数 | 50 | `grasp_flow.py:182` |
| MANO PCA 分量 `ncomps` | 45 | `mano.py:32` |
| MANO 手性 | right（右手） | `mano.py:29` |
| MANO 中心关节 `center_idx` | 0（手腕中心） | `mano.py:38` |

---

### 六、核心调用链

```
# ======================== 推理入口 ========================
inference.py:main()
  ├── load_model(checkpoint_path)
  │     ├── resolve_checkpoint_path()        # 查找 .safetensors / .pt
  │     ├── load_raw_checkpoint()            # 统一加载 + 解析 cfg/norm_stats
  │     └── GraspFlowModel(cfg, norm_stats)  # 组装完整模型
  ├── GraspDataset(dataset_path, split='val')
  └── for batch in DataLoader:
        └── model.sample(point_uv, camera_K, rgb, pcl_xyz, pcl_rgb)

# ===================== model.sample() =====================
grasp_model.py:GraspFlowModel.sample()
  │
  ├── encode_scene(point_uv, camera_K, rgb, pcl_xyz, pcl_rgb)
  │     │
  │     ├── _backproject(point_uv, camera_K)     # (u,v,d)→(x,y,z) 纯几何
  │     │
  │     ├── [if use_rgb] DINOv2Encoder.forward(rgb)
  │     │     └── AutoModel(rgb) → 去掉 CLS+register → (B,256,1024)
  │     │
  │     ├── [if use_depth] PointNeXtEncoder.forward(xyz, pcl_rgb)
  │     │     └── PointNeXt(xyz, rgb_pcl)
  │     │           ├── stem: Linear(6→c)       # xyz+RGB concat
  │     │           ├── SA1(1024, r=0.025) → InvResMLP×1
  │     │           ├── SA2(256, r=0.05)  → InvResMLP×2
  │     │           ├── SA3(64, r=0.10)   → InvResMLP×1
  │     │           ├── SA4(16, r=0.20)   → InvResMLP×1
  │     │           ├── FP4: 16→64  (插值+skip)
  │     │           └── FP3: 64→256 (插值+skip) → (B,256,512)+(B,256,3)
  │     │
  │     └── PatchFusion.forward(point, rgb_patches, depth_patches, centroids, K)
  │           ├── FourierPosEmbed(point)            # 条件点→point token
  │           ├── _paint(rgb_patches, centroids, K) # PCL→2D投影→双线性采样
  │           ├── painting_proj([painted ∥ depth])  # concat→MLP→256 tokens
  │           ├── + pos_embed_3d(centroids)         # 3D 位置编码
  │           ├── point_cross_attn(x, point_token)  # 条件注入
  │           └── Transformer × 4                   # 自注意力精炼
  │           → condition_tokens (B, 256, 1024)
  │
  └── GraspFlowMatching.sample(cond, steps=50)
        ├── x = randn(B, 99)                       # t=1 纯噪声
        └── for i = 49, ..., 0:                    # Euler ODE 50步
              ├── t = (i+1) / 50
              ├── v = TokenPerGroupDiT(x, t, cond)
              │     ├── 分组投影: trans(3→512) / wrist(6→512) / fingers(90→512)
              │     ├── + token_type_emb(3,512)
              │     ├── time_mlp(SinusoidalPosEmb(t×1000))
              │     ├── cond_proj(condition_tokens)  # 1024→512
              │     ├── AdaLNCrossAttnBlock × 6
              │     │     ├── AdaLN 调制(time→7 params)
              │     │     ├── QK-norm Self-Attn(3 tokens)
              │     │     ├── QK-norm Cross-Attn(→cond tokens)
              │     │     └── AdaLN GeLU FFN
              │     ├── final_modulation(time→scale,shift)
              │     └── 三头输出: trans(3) / wrist(6) / fingers(90) → concat 99D
              └── x = x - v * dt                    # Euler 步进
        └── denormalize(x) → 99D MANO 参数

# ==================== MANO 解码 ====================
mano.py:mano_params_to_grasp_dict()
  ├── MANO.decode_mano_params(99D) → t, R_6d, pose_6d
  ├── roma.special_gramschmidt(R_6d) → R_3×3
  ├── manotorch.rotation_to_axis_angle → axis-angle
  ├── ManoLayer(pose_aa(48), shape(10))
  │     → vertices (778,3)  # MANO mesh 顶点
  │     → joints (21,3)     # 关键点
  └── 组装为 Grasp 字典（全 numpy）

# ==================== 数据处理 ====================
prepare_inputs.py:main()
  ├── 自动检测 rgb.png + depth.png + intrinsics.txt
  ├── 中心裁剪正方形 → resize 224×224 → 调整 K
  └── JPEG/PNG 编码 → pickle.dump(.pkl)
```

---

### 七、训练管线（代码已有，训练入口尚未发布）

```
grasp_model.py:GraspFlowModel.forward(batch)
  │
  ├── encode_scene(...) → condition_tokens
  │
  └── GraspFlowMatching.forward(gt_mano_params, condition_tokens)
        ├── normalize(x_start) → x_norm           # 分三组归一化
        ├── t ~ U(0, 1)                            # 随机采样时间步
        ├── x_t = (1-t)·x_norm + t·ε              # 线性插值（Rectified Flow）
        ├── target = ε - x_norm                    # 速度场目标
        ├── pred = TokenPerGroupDiT(x_t, t, cond)  # 预测速度场
        └── loss = MSE(pred, target)               # 速度场预测损失
             (time_weight = 1 - t, 偏重后期 t→0 的步)
```

---

### 八、关键设计亮点

1. **冻结 DINOv2 + 可训练 PointNeXt 双编码器**
   - DINOv2 提供大规模预训练的视觉语义理解（物体类别、形状、材质），完全冻结不参与训练
   - PointNeXt 从深度点云中学习场景几何结构（曲面、边缘、距离），端到端可训练
   - 两者互补：语义告诉你"这是什么物体，哪面适合抓"，几何告诉你"手离物体多远，什么角度不会穿透"

2. **PointPainting 跨模态融合**
   - 将 3D 点云质心投影到 2D 图像平面，在 DINOv2 的 patch 特征图上做双线性采样
   - 本质是让每个 3D 点"看到"它在 2D 图像中对应位置的语义特征
   - 相比简单的 concat 双流方案，PointPainting 实现了像素级的 2D-3D 精确对齐
   - K（相机内参）仅用于纯几何投影，不作为可学习参数输入，保证跨相机泛化

3. **Rectified Flow Matching 生成框架**
   - 使用线性插值路径 `x_t = (1-t)·x_0 + t·ε`，比 DDPM 的扩散路径更简单
   - 速度场预测目标 `v = ε - x_0`，训练目标为直线上恒定速度
   - Euler ODE 50 步即可完成采样，比 DDPM 的 1000 步大幅加速
   - 相比 DDIM 等加速采样方法，Rectified Flow 天然直线路径理论上允许更少步数

4. **TokenPerGroup DiT 架构**
   - 将 99D MANO 参数按语义分为三组（translation/wrist/fingers），每组独立投影为一个 token
   - 三个 token 通过自注意力交互，同时通过交叉注意力读取场景条件
   - 可学习的 token_type_emb 让模型区分不同语义组的 token
   - AdaLN（自适应 LayerNorm）用时间步嵌入生成 7 个调制参数，控制各子层的缩放和平移
   - 所有门控零初始化 → 训练初期模型等价于恒等映射，大幅提升训练稳定性

5. **纯几何条件化**
   - 用户点击的像素 `(u,v)` 通过相机内参 K 反投影为 3D 坐标 `(x,y,z)`
   - K 仅用于 `(u-cx)*z/fx` 这种几何计算，**不作为神经网络的可学习输入**
   - 这意味着模型不依赖特定相机的内参数值，可泛化到任何双目相机

6. **固定手型 + 生成姿态**
   - 使用预设的 `mano_rhand_shape.npy`（10 维 PCA shape），所有抓取共享同一手型
   - 模型只需生成 99D（手腕位姿 + 手指姿态），将生成空间从 109D 缩减为 99D
   - 显著降低了生成任务的难度，聚焦于"怎么抓"而非"用什么手抓"

7. **条件点交叉注意力注入**
   - 融合模块中，用户点击的 3D 点作为一个单独的 query token 与所有场景 patch 做交叉注意力
   - 这相当于告诉模型"重点关注这个位置"，引导生成的手部向该物体区域靠近
   - 同时使用 Fourier 位置编码让模型感知该点的精确 3D 位置

---

### 九、外部依赖

| 类别 | 包名 | 用途 |
|------|------|------|
| **深度学习框架** | `torch` (2.9.1), `torchvision`, `torchaudio` | 模型训练/推理基础框架 |
| **点云算子** | `torch-cluster` (from pyg) | FPS（最远点采样）+ kNN（球查询） |
| **Transformer** | `transformers` (HuggingFace) | 加载 DINOv2 预训练权重 |
| | `xformers` | 高效的 attention 实现 |
| **手部模型** | `manotorch` (git) | MANO 手部参数化模型前向计算 |
| | `roma` | 6D 旋转表示 → SO(3) 矩阵（special_gramschmidt） |
| | `chumpy` (git) | manotorch 的底层依赖 |
| **图像处理** | `opencv-python` | 图像编解码（JPEG/PNG）、resize、腐蚀 |
| | `pillow` (via torchvision) | 图像读取 |
| **3D 可视化** | `viser` | 交互式 3D Web 可视化（浏览器渲染） |
| | `plotly` | 交互式 2D/3D 图表 |
| **配置/CLI** | `omegaconf` | 训练配置管理（YAML 结构化配置） |
| | `tyro` | CLI 参数解析（比 argparse 更简洁） |
| **序列化** | `safetensors` | 模型权重安全存储格式（比 .pt 更快更安全） |
| **终端美化** | `rich` | 彩色终端输出、表格、进度条 |
| **数值计算** | `numpy` | 矩阵运算、数组操作 |
| **代码质量** | `ruff`, `pre-commit` | 代码格式化和 lint |

---

### 十、99D 归一化统计量（`assets/norm_stats.json`）

```json
{
  "translation": {"mean": [x̄, ȳ, z̄], "std": [σx, σy, σz]},
  "wrist_rot":   {"mean": [6个值的均值], "std": [6个值的标准差]},
  "finger_rot":  {"mean": [90个值的均值], "std": [90个值的标准差]}
}
```

- 训练时：原始 99D → 分三组减去 mean 除以 std → 输入 Flow Matching
- 推理时：Flow Matching 输出 → 分三组乘以 std 加上 mean → 原始 99D → MANO 解码

---
