# HUG 数据处理 Pipeline 解析

> 分析对象：`src/dataloader/grasp_dataset.py`、`src/utils/pcl_utils.py`、`src/prepare_inputs.py`、`src/models/grasp_model.py`、`src/inference.py`。
> Pipeline 分为五段：**离线制备 → 样本索引 → 单样本加载 → 模型侧编码 → 推理输出**。

## 0. 总览

```
【离线】原始采集 (RGB+depth+K)                    【数据集】train/val/test pkl
   prepare_inputs.py ──→ eval-schema pkl ────────┐
                                                 ▼
                              GraspDataset.__getitem__（train/eval 统一入口）
                                                 │
        ┌──────────────┬─────────────────────────┼──────────────────┐
        ▼              ▼                         ▼                  ▼
   RGB 分支       query 点                  depth→点云分支        GT 分支(仅train)
   JPEG解码       condition_point           uint16→米→反投影      99D MANO
   ImageNet归一化  或mask内随机采样          0.3m球裁剪→4096点     +landmarks
        └──────────────┴─────────────────────────┘
                       ▼
            encode_scene（模型内：DINOv2 / PointNeXt / 融合）
                       ▼
            flow transformer（归一化空间）→ 50步Euler → denormalize
                       ▼
            grasp_pred/{stem}.pkl（推理输出）
```

## 1. 离线制备：任意相机采集 → 统一 224² pkl

文件：`src/prepare_inputs.py`

把任意分辨率的 RGB（uint8）+ depth（uint16，1mm 单位）+ 内参（`fx fy cx cy` 或 3×3 K，支持 .txt/.csv/.npy/.json）打包成与数据集同构的 eval-schema pkl：

- **中心方形裁剪**（沿短边，`_center_crop_square`）→ resize 到 224×224：
  - RGB 用 `cv2.INTER_AREA`（缩图抗锯齿）；
  - depth 用 `cv2.INTER_NEAREST`（保值域，不产生插值伪深度）。
- **内参同步调整**（`_adjust_K`, L48-54）：主点先减裁剪偏移 `(x_off, y_off)`，再乘缩放因子 `224/sq_size`——几何一致性最关键的一步。裁剪后方形分辨率的 K 存为 `camera_original`。
- **编码内嵌**：RGB→JPEG bytes、depth→PNG bytes 存入 pkl（单文件自包含）。
- 写出 `grasp=None, object_mask=b""` 的 `GraspData` schema；先写 `.tmp.pkl` 再 rename，避免半写文件。

用法：`python -m src.prepare_inputs --dataset-path data/custom`（自动探测 `*rgb*` / `*depth*` / `*intrinsics*` 文件）。

## 2. 样本索引：samples.txt 缓存机制

文件：`src/dataloader/grasp_dataset.py:84-117`（`_load_file_list`）

- 递归 glob 数据集根目录下所有 `.pkl`，**排除 `grasp_pred/` 输出目录**（`find_pkls`, L75-82），防止推理输出回流成输入。
- 首次 glob 后把 root-relative stem 列表写入 `samples.txt` 缓存——128 万样本在 NFS 上 glob 需数分钟，之后启动直接读清单。
- `samples_filename` 参数支持显式子集文件（文件不存在则报错）——对应论文 data scaling 实验的嵌套训练子集（25K/50K/.../1M）。

## 3. 单样本加载：`__getitem__`（train/eval 统一入口）

文件：`src/dataloader/grasp_dataset.py:278-328`

### 3.1 解码（L136-161）

| 字段 | 编码 | 解码后处理 |
|---|---|---|
| `image` | JPEG bytes | BGR→RGB，(H,W,3) uint8 |
| `depth` | PNG uint16（1mm） | ≥65535 置 0；除 1000 转米；`nan_to_num`；clip [0,100] |
| `object_mask` | PNG 灰度 | (H,W) uint8 |

### 3.2 query 点：train / eval 的分叉点（L289-301）

- **eval**：读 pkl 中的 `condition_point [u,v]`，查深度图取 `d`；该像素深度无效时用全图有效深度均值兜底（L296-298）。
- **train**（`_sample_point_from_mask`, L186-211）：
  1. mask 3×3 腐蚀一次（避开边缘噪声）；
  2. 取 `腐蚀mask ∩ 有效深度` 区域**随机采样**一个像素；
  3. 两级兜底：交集为空→退回原 mask；再无有效深度→用均值或 0.5m。

  **注意：每个 epoch 重新随机**——同一样本每次 query 点不同，是免费的条件增强，防止模型记住"点永远在物体中心"。

### 3.3 RGB 分支（L319-320）

PIL → `ToTensor` → **ImageNet mean/std 归一化**（`IMAGENET_MEAN/STD`, L33-34）——冻结 DINOv2 的标准输入格式。与 99D grasp 的 `norm_stats` 是两套完全独立的归一化（后者见 `HUG_norm_stats_analysis.md`）。

### 3.4 点云分支（`src/utils/pcl_utils.py`）

1. `pixel_to_xyz` (L81-92)：query 像素 (u,v,d) 经 K 反投影为米制 3D 点——纯几何运算。
2. `backproject_to_pcl` (L13-52)：全图深度反投影，滤 `z≤0` 与 `z>3m`；以 query 点为球心做 **0.3m 球裁剪**（`pcl_crop_radius`），把点云密度聚焦目标物体（消融：去掉 crop 掉 ~10 SR 点）。
3. `sample_fixed_n` (L55-78)：随机采样到**固定 4096 点**——多于则无放回抽样，少于则重复采样，空点云返回全零兜底。
4. 输出：`xyz` (4096,3) float32 米制；`pcl_rgb` (4096,3) ∈ [0,1]。

### 3.5 GT 分支（仅 train，L312-318）

`grasp` 非 None 时才提取：

- `mano_params`：99D = `t`(3, 米) + `R_6d`(6) + `pose_6d`(90)（`_get_mano_params`, L123-134）——后续 flow matching 归一化的输入；
- `mano_shape` (10)、`landmarks_3d` (21,3)、`landmarks_2d` (21,2)。

eval pkl 的 `grasp=None`，此分支整体跳过（注释：*"Eval pkls carry no grasp label; GT fields are train-only"*）。

## 4. 模型侧编码：`encode_scene`

文件：`src/models/grasp_model.py:129`

数据进模型后的处理（属于 forward，但 pipeline 意义上的最后一段）：

- **query 点**：(u,v,d) 在模型内部再次经 K 反投影成米制 3D 点。**K 只用于几何运算，从不作为可学习输入**——论文强调的跨相机泛化设计（另有 `use_2d_point` 消融模式：(u,v) 直接除 image_size 归一化）。
- **RGB 流**：冻结 DINOv2-Base（`torch.no_grad`）→ 256 patch tokens。
- **点云流**：PointNeXt U-Net（可训练）→ 256 region tokens + 米制质心。
- **融合**（point painting）：质心投影回图像、双线性采样 DINOv2 patch 特征 → 与 PC token 拼接 → MLP → query 交叉注意 + 4 层 transformer → 条件序列 s。
- 之后进入 flow transformer（归一化空间 ODE，详见归一化分析文档）。

## 5. 推理输出

文件：`src/inference.py`

50 步 Euler 采样（`grasp_flow.py:211-226`）→ `denormalize` 回米制 → `mano_params_to_grasp_dict` 组装为与 GT 同构的 grasp dict → 写 `grasp_pred/{stem}.pkl`。可视化走 `get_inference_data`（L231-276），额外加载 `image_original/`（若存在）与 canonical MANO shape/faces 兜底。

## 6. 设计要点回顾

| 设计 | 目的 |
|---|---|
| 一切米制、相机系；K 不进可学习参数 | 跨相机/跨场景泛化（论文 App. C.2） |
| 训练时 query 点在 mask 内随机采样 | 条件增强，防过拟合"点在中心" |
| 0.3m 球裁剪 + 固定 4096 点 | 密度聚焦目标物体，张量尺寸恒定 |
| train/eval 统一 schema，`grasp=None` 区分 | 同一代码路径服务训练与评测 |
| 图像/深度内嵌编码进 pkl | 单文件自包含，NFS 友好 |
| `samples.txt` 索引缓存 | 128 万样本快速启动 |
| depth uint16 有效性清洗（≥65535 置 0、去 NaN/Inf） | 真实传感器噪声鲁棒 |

## 附：关键代码位置

| 文件 | 位置 | 作用 |
|---|---|---|
| `src/prepare_inputs.py:39-66` | 裁剪/K 调整/深度编码 | 离线制备核心 |
| `src/prepare_inputs.py:69-126` | `prepare_pkl` | 写 eval-schema pkl |
| `src/dataloader/grasp_dataset.py:84-117` | `_load_file_list` | 样本索引与缓存 |
| `src/dataloader/grasp_dataset.py:123-134` | `_get_mano_params` | 99D GT 提取 |
| `src/dataloader/grasp_dataset.py:150-161` | `_decode_depth_uint16` / `_depth_meters` | 深度清洗 |
| `src/dataloader/grasp_dataset.py:186-211` | `_sample_point_from_mask` | 训练 query 点随机采样 |
| `src/dataloader/grasp_dataset.py:278-328` | `__getitem__` | 单样本加载主流程 |
| `src/utils/pcl_utils.py:13-52` | `backproject_to_pcl` | 深度反投影 + 球裁剪 |
| `src/utils/pcl_utils.py:55-78` | `sample_fixed_n` | 固定点数重采样 |
| `src/models/grasp_model.py:129` | `encode_scene` | 模型侧场景编码 |
