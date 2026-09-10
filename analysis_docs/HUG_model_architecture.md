# HUG 模型架构解析

> 分析对象：`src/models/` 下全部模块（`grasp_model.py`、`encoders.py`、`pointnext.py`、`fusion.py`、`grasp_flow.py`、`transformer.py`、`mano.py`）。
> 配置取代码默认值，与论文一致：DINOv2-Base、N=256 tokens、D_f=1024、D_m=512、L=6 层 DiT。
> 结构：**双模态编码器 → 点绘制融合 → token-per-group DiT flow matching → MANO 解码**。

## 0. 总体维度流（推理方向）

```
rgb (B,3,224,224) ──→ DINOv2(冻结) ──────────────→ rgb_patches (B,256,768)
                                                          │ grid_sample
pcl_xyz (B,4096,3) ─→ PointNeXt(可训练) ─→ depth_patches (B,256,512) ─┤
pcl_rgb (B,4096,3) ─┘                    └→ centroids (B,256,3) ──────┘
point_uv (B,3) ──K反投影──→ point (B,3) ──RFF──→ point_token (B,1,1024)
                                                          ▼
                                  PatchFusion ──→ cond (B,256,1024)
                                                          ▼
x~N(0,I) (B,99) ──→ TokenPerGroupDiT ×6层 ──→ v (B,99)  [×50步 Euler]
                                                          ▼
                                          denormalize → (B,99) 米制
                                                          ▼
                                          MANO(冻结, β固定) → landmarks (B,21,3)
                                                            vertices (B,778,3)
```

## 1. 关于输入形式的两个澄清

### 1.1 Depth 图的"大小"：存在但不进模型

Depth 图与 RGB 同分辨率 **224×224**（uint16 PNG，1mm 单位），但它只活在数据处理阶段——dataloader 中即被反投影为点云，模型从未见过 depth"图"：

| 阶段 | RGB 流 | Depth 流 |
|---|---|---|
| pkl 存储 | JPEG bytes | PNG uint16 bytes |
| 解码后 | (224,224,3) uint8 | (224,224) uint16（毫米）→ 米制 float32 |
| **进模型的张量** | rgb (B,3,224,224) | pcl_xyz + pcl_rgb (B,4096,3) |

设计原因：RGB 喂冻结 DINOv2 必须保持图像格式；深度要参与米制几何运算（0.3m 球裁剪、FPS、质心投影），点云是天然载体，且对深度空洞/噪声鲁棒（无效深度直接滤除）。

### 1.2 pcl_xyz 与 pcl_rgb：同一个点云的两份属性

**不是两个点云**——两者按索引一一对应（`pcl_utils.py:121-131` 中同一掩码、同一采样索引生成）：

- `pcl_xyz[i]` = (x,y,z)：相机系米制坐标，来自 depth 反投影——干**几何**的活：FPS、ball-query、质心、RFF 位置编码、point painting 投影；
- `pcl_rgb[i]` = (r,g,b)：同一像素在 RGB 图上的颜色，归一到 [0,1]——只当**特征**用。

两者在 PointNeXt stem 才汇合：`cat([xyz, rgb_pcl], -1)` → (B,4096,6) → MLP（`pointnext.py:268`）。拆成两个张量是因为下游用途完全不同，且 `pcl_use_rgb` 开关支持训练纯几何模型（消融/无颜色传感器，详见 §3.3、§3.4）。

## 2. RGB 编码器：冻结 DINOv2

文件：`src/models/encoders.py:12-43`

- `dinov2-with-registers-base`：patch=14，hidden=768，4 个 register token；**全程 no_grad 冻结**；
- 输入 (B,3,224,224)（ImageNet 归一化）→ `last_hidden_state` (B,261,768)；
- 切掉前 5 个 token（1 CLS + 4 register）→ **rgb_patches (B,256,768)**，即 16×16 patch 网格。

## 3. 点云编码器：可训练 PointNeXt U-Net

文件：`src/models/pointnext.py:195-285`（宽度 c=64，输入为 0.3m 球裁剪后重采样的 4096 点）

| 阶段 | 操作 | 点数 | 特征维 | 半径/邻域 |
|---|---|---|---|---|
| stem | cat[xyz,rgb]→MLP | 4096 | 6→64 | — |
| SA1 + 1×InvResMLP | FPS+分组 | 4096→1024 | 64→128 | r=0.025m, k=32 |
| SA2 + 2×InvResMLP | | 1024→256 | 128→256 | r=0.05m, k=32 |
| SA3 + 1×InvResMLP | | 256→64 | 256→512 | r=0.10m, k=32 |
| SA4 + 1×InvResMLP | | 64→16 | 512→1024 | r=0.20m, k=16 |
| FP4 | 上采样+插值 | 16→64 | 1024+512→512 | — |
| FP3 | | 64→256 | 512+256→512 | — |

输出：**depth_patches (B,256,512)**（`out_dim=8c`）+ **centroids (B,256,3)**（SA2 层米制质心，供融合阶段位置编码与投影）。半径为米制（0.025–0.20m），与 0.3m 裁剪球匹配。

### 3.1 centroids (B,256,3)：256 是 token 数而非维度

**(B,256,3) 的正确读法**：256 个区域 token × 每个 token 的 3D 质心坐标（米制 XYZ，相机系）。真正的特征维在 `depth_patches (B,256,512)` 的 512 里——centroids 与 depth_patches 是**同一批 256 个 token 的"位置"和"特征"两份属性**（如同 pcl_xyz/pcl_rgb 是同一点云的两份属性）。

**centroids 的来源：PointNet++ 自带概念，HUG 的用法设计**：
- SA 层每级 = FPS 选中心点 + 聚合邻域特征，输出天然是"中心点坐标 + 特征"对（PointNet++/PointNeXt 结构固有，代码中 xyz1~xyz4 逐级维护）；
- HUG 的设计：**在 FP3 阶段（256 点）截停**，把 SA2 层的 `xyz2` 显式暴露为输出（`return feat2_up, xyz2`），供下游 RFF 位置编码与 painting 投影——标准 PointNeXt 分类用法中质心只是中间产物（下采样到最粗后全局池化）。

**为什么恰好是 256**：与 DINOv2 的 16×16 = 256 个 patch token 对齐（论文 §4.1："outputs N = 256 per-region tokens"）。两流 token 数一致，融合序列长度整齐（painting 路径 256 token；concat 消融路径 256+256=512）。

**FPS（Farthest Point Sampling，最远点采样）**：从 N 个点中迭代选 k 个空间覆盖最匀的代表点——每轮选取"到已选集合的最近距离最大"的点加入集合（`pointnext.py:29` `_fps_indices`，O(N·k)）。为何不用随机采样：深度反投影的点云密度极不均匀（图像中心密、边缘稀；近处密、远处稀），随机采样会被高密度区域绑架、稀疏区域无代表；FPS 强制代表点在空间铺开，保证 0.3m 裁剪球内各处（含点稀的物体边缘）都有 token 覆盖——这对指尖级 3D 监督至关重要。

**质心的两个下游用途**：(1) RFF 位置编码——告诉 transformer 每个 token 的空间位置；(2) painting 投影锚点——决定往图像哪个位置采语义特征（见 §4）。

### 3.2 为什么 PointNeXt 需要 pcl_rgb 颜色信息

论文里 RGB 已通过 point painting 注入（§4），stem 处再喂原始颜色看似重复，实则解决**纯几何的固有歧义**：

1. **物体边界**：0.3m 球裁剪会把桌面、背景、相邻物体一起圈入；几何上目标可能与桌面平滑连续（尤其深度噪声下），颜色/纹理边缘清晰标出"哪片点属于目标物体"，直接影响 FPS 分组与局部特征质量；
2. **弱几何物体**：~1cm 高的扁平物体在点云里只是桌面上几乎不可分辨的"小鼓包"，颜色对比是其存在的最强信号；
3. **材质与可操作部件**：同形状的毛巾/书、手柄/瓶身，几何区分不了，颜色可以；
4. **深度噪声的独立冗余**：Aria 立体深度（S2M2）在黑色、反光、透明、无纹理表面出洞/出错，这些位置几何坐标本身不可信；颜色是与深度物理独立的测量，可兜底（论文 Table 2 PC-only 消融：深度不可靠时模型会"吸附到旁边更大的物体"）。

**与 point painting 的分工**（不是重复，是两个层级）：

| | stem 的 pcl_rgb | 融合阶段的 point painting |
|---|---|---|
| 内容 | 原始颜色（低层外观：边缘、纹理） | DINOv2 特征（高层语义："这是手柄"） |
| 粒度 | 逐点，4096 个点全有 | 256 个质心处采样，patch 级（14×14 像素/块） |
| 注入时机 | PointNeXt 第一层，影响几何处理本身 | PointNeXt 编码完之后，影响特征融合 |
| 作用 | 帮网络解读几何（边界/归属） | 给已编码 token 贴语义标签 |

关键差别在时机与粒度：stem 颜色从第一层参与局部特征提取（"带着眼镜看形状"）；DINOv2 patch 特征空间分辨率粗，细物体边界已被抹平，逐点原始颜色补上分辨率缺口。两者是早期/晚期互补融合。

**作者的定位：有用但不可依赖的辅助信号**——颜色只多 3 维、几乎零成本，用于提升感知上限；同时用灰度帧训练（§3.2）防止模型把它当拐杖。

### 3.3 颜色缺席时如何保持可用：灰度帧训练 + pcl_use_rgb 开关

两个独立机制，一个作用于数据，一个作用于结构：

**机制一：灰度帧训练（数据层面，防"颜色依赖"）**

- **问题**：部署相机不一定是彩色的——论文 in-the-wild 实验直接用 Aria 的立体 SLAM 相机（灰度/单色）；若模型只见过彩色输入，可能学会"靠颜色找边界"的捷径，灰度部署即崩；
- **做法**（论文 §4 数据集统计）：训练集实为 2M 条 = **1M 彩色帧 + 1M 灰度帧**（灰度帧来自 Aria 左目立体相机），且灰度帧与同时刻彩色帧**配对**（同场景、同 grasp 标注，传感器不同）；
- **为何有效**：灰度帧中 pcl_rgb 只剩亮度无色彩，模型仍必须预测对 grasp，梯度逼着它把几何/亮度/语义等不依赖色彩的通路练扎实；"同时刻配对"等于对同一抓取直接监督出颜色不变性。本质是域随机化——**网络不会依赖一个不可依赖的特征**。部署到灰度相机零适配（图像照常 3 通道解码，只是 R=G=B）。

**机制二：`pcl_use_rgb` 开关（结构层面，彻底拔掉颜色通道）**

配置项（`grasp_model.py:65`），为 False 时点云颜色属性**整个不存在**（不是"颜色是灰的"，而是输入维度都没有）。用途：(1) 消融实验，量化颜色贡献；(2) 纯几何传感器（无逐点亮度对齐的深度相机/激光雷达）。

粒度对比：

| | 灰度帧训练 | `pcl_use_rgb` 开关 |
|---|---|---|
| 作用层面 | 训练数据 | 模型结构/配置 |
| 颜色状态 | 有通道，内容无色彩（灰） | 点云输入里通道直接消失 |
| 生效时机 | 训练中一直生效（一半样本） | 需手动配置，从头训练 |
| 解决的问题 | 部署灰度相机不掉点（零适配） | 量化颜色贡献 / 无颜色传感器 |

类比：灰度帧训练是"打疫苗"（训练时一半时间无色彩也必须完成任务，部署天生免疫）；开关是"截肢选项"（需要时彻底去掉颜色输入）。

### 3.4 pcl_use_rgb 的实现：构造时静态分支，不是运行时开关

**PointNeXt 不能接受变维度输入，也没有尝试接受**。`pointnext.py:218`：

```python
self.stem = _shared_mlp([6 if use_rgb else 3, c])
```

`__init__` 执行时即定死第一层 Linear 权重形状：`True` → 64×**6**，`False` → 64×**3**。forward 中的分支（`pointnext.py:266-270`）只是按同一标志走对应路径，并有 `assert rgb_pcl is not None` 兜底。

**关键推论**：开关改变的是"训练哪种模型"，不是"让模型变形"——

```
pcl_use_rgb=True   →  训练出 stem 为 6→64 的模型 A
pcl_use_rgb=False  →  训练出 stem 为 3→64 的模型 B
A、B 是两个不同 checkpoint，权重形状互不兼容
```

不能拿 A 的权重在推理时拨成 False（`load_state_dict` 形状不匹配直接报错）。推理一致性靠 **checkpoint 自带配置**：`inference.py:95-120` 从 checkpoint 读训练时 cfg（含 `pcl_use_rgb`）构建模型再加载权重，标志与权重永远同源。

**为何不做成"真·可变输入"**（如颜色缺失填零）：零填充会让第一层权重被迫兼容 6 维/3 维两种输入分布，表达能力打折。作者的划分更干净——灰度相机靠灰度帧训练解决（stem 仍是 6 维输入，内容灰）；彻底不要颜色则用开关**从头训练**纯几何模型。

### 3.5 相关配置级联逻辑

`grasp_model.py:36-46` 处理标志间依赖：

- `use_pointpainting` 需要 RGB+depth 双全，否则强制 False；
- `use_depth=False` → 强制 `use_2d_point=True`（3D 点条件依赖深度反投影）；
- `use_depth=False` → `pcl_use_rgb=False`（无点云则无点云颜色）。

注意 `pcl_use_rgb`（点云颜色属性）与 `use_rgb`（DINOv2 图像分支）是两个独立开关：前者只拔点云颜色，后者动整个图像分支（并级联关 point painting）。

## 4. 融合：PointPainting + query 条件

文件：`src/models/fusion.py`（默认 `use_pointpainting=True`，d_model=1024，4 层 8 头）

1. **点绘制**（`_paint`, L139-161），完整机制：
   - rgb_patches (B,256,768) reshape 为 (B,768,16,16) 特征图（恢复 2D 空间结构）；
   - centroids 经针孔模型投影：`u = fx·X/Z + cx`，`v = fy·Y/Z + cy`（`_project_centroids`，Z clamp 防除零），归一化到 [-1,1]；
   - `F.grid_sample` 在 256 个亚像素位置**双线性插值**采样 → **painted (B,256,768)**；`padding_mode="zeros"`——投影落到图像视野外的质心拿零向量语义，优雅降级；
   - **三份张量的分工**（易混淆点：投影的是 centroid，但 depth_patches 并非没用到）：

     | 角色 | 张量 | 用途 |
     |---|---|---|
     | 投影锚点 | centroids (B,256,3) | 只提供"往图像哪里看"的坐标 |
     | 几何特征 | depth_patches (B,256,512) | 局部区域形状特征，**完整保留进拼接** |
     | 语义特征 | painted (B,256,768) | 锚点指定的图像位置采到的 DINOv2 特征 |

     比喻：centroid 是邮寄地址，depth_patches 是包裹（几何），painting 是按地址取信件（语义），`cat` 把包裹和信件捆在一起；
2. **拼接投影**：cat[painted, depth_patches] = (B,256,**1280**) → `painting_proj`（Linear 1280→1024 → SiLU → Linear 1024→1024）→ (B,256,1024)；
3. **位置编码**：`+ pos_embed_3d(centroids)`——随机傅里叶特征（RFF）+ 线性 bypass（保留原始坐标的线性通路）；
4. **query 条件**：point (B,3) → 同一 RFF → `point_proj` → **point_token (B,1,1024)**；`point_cross_attn` 让 256 个场景 token 交叉注意该 query token——共享同一 RFF 空间使注意力天然可度量"点击位置到各质心的 3D 距离"；
5. **4 层 TransformerBlock**（1024 维、8 头、RMSNorm + QK-norm + Flash SDPA）→ **cond (B,256,1024)**。

备选路径（`use_pointpainting=False`，消融用）：RGB/depth 两流分别投影后拼接成 512 token 序列，加模态嵌入。与 painting 的本质差别：concat 路径中 RGB 特征与几何特征只是"相邻"，**没有空间对齐**；painting 用 centroid 做桥梁，让每个 3D token 拿到的恰好是它自己投影位置的图像语义——空间上严格对齐的逐点融合。消融去掉 painting 掉 ~10 val / ~15 test SR 点，证明对齐优于简单拼接。

## 5. Flow transformer：TokenPerGroupDiT

文件：`src/models/grasp_flow.py:28-162`（d_model=512，6 层，8 头）

- **输入 token 化**：归一化 x (B,99) 切三段 → `Linear(3→512)` / `Linear(6→512)` / `Linear(90→512)` → stack **(B,3,512)**，加可学习 token 类型嵌入 (3,512)；
- **时间嵌入**：t×1000 → Sinusoidal(128) → MLP → **c (B,512)**；
- **条件投影**：`cond_proj` Linear(1024→512)，context = (B,256,512)；
- **6 × AdaLNCrossAttnBlock**（`transformer.py:140-196`）：每块 = AdaLN 调制自注意力（3 token 间）→ 交叉注意力（3 grasp token ↔ 256 场景 token）→ AdaLN FFN；调制向量由 c 生成 7×512（scale/shift/gate ×2 + cross gate），**全部 zero-init**（训练初期等价恒等映射，DiT 标准稳定化技巧）；
- **输出头**：final RMSNorm + 调制后，三 token 分别过 `Linear(512→3/6/90)` → 拼回 **v (B,99)** 速度场。

**采样**：`x ~ N(0,I) (B,99)`，50 步 Euler 从 t=1 积到 t=0，出口 `denormalize` → 米制 99D（归一化细节见 `HUG_norm_stats_analysis.md`）。

## 6. MANO 解码（冻结）

文件：`src/models/mano.py:48-101`

- 99D 拆开：t (B,3) + R_6d (B,6) + pose_6d (B,15,6)；
- 6D → 旋转矩阵 → 轴角：pose_coeffs (B,48) = 手腕轴角(3) + 15 关节轴角(45)；
- `ManoLayer(pose_coeffs, β)`，β 固定为 canonical 值（10 维 buffer，跨采集者统一手型）；`center_idx=0` 手腕为原点；
- 输出：**landmarks_3d (B,21,3)**（`+t` 平移到相机系）、**vertices (B,778,3)**。

### 附：landmarks 是什么

手部 **21 个 3D 关节关键点**（"手的骨架"）：16 个运动学关节（手腕 1 + 5 指 × 3 关节 MCP/PIP/DIP）+ 5 个指尖。来源：Aria 眼镜跟踪输出 21 个 3D landmarks → 拟合 MANO → 正算存回数据集。用途：

1. **L3D 训练监督**（λ=20）：监督 21 个语义关键点比监督 778 个顶点更聚焦指尖/关节；
2. **landmarks_2d (21,2)**：经 K 投影到图像，数据采集时做"手接触物体 mask"校验；
3. **FC error 评估指标**：拇指尖 + 最近支撑指尖到物体表面的距离。

对比 vertices (778,3)：mesh 表面顶点（"手的皮肤"），用于可视化、仿真加载、穿透类指标。

## 7. 训练前向的完整损失路径

文件：`src/models/grasp_model.py:206-229`

```
GT 99D → normalize → 与 eps 插值得 x_t ──→ DiT → pred_v (B,99)
                          │                          │
   速度目标 v*=eps−x0 ◄───┘          Lv = MSE(pred_v, v*)        λ=1
                          │
   recover_x0: x̂0 = x_t − t·pred_v → denormalize → MANO
                          │
   L3D = L1(landmarkŝ (B,21,3), GT landmarks)，权重 (1−t)         λ=20
```

总损失（论文 Eq.1）：`L = λv·Lv + λ3D·(1−t)·L3D`，λv=1，λ3D=20。(1−t) 权重把几何监督集中到接近干净样本的步（x̂0 仅在低噪声时有意义）。

## 8. 关键维度速查表

| 符号 | 张量 | 维度 |
|---|---|---|
| 输入图像 | rgb | (B, 3, 224, 224) |
| 深度图（仅数据阶段） | depth | (224, 224) uint16 毫米 |
| 输入点云（位置/颜色） | pcl_xyz / pcl_rgb | (B, 4096, 3) |
| query 点 | point_uv → point | (B, 3) → (B, 3) 米制 |
| RGB tokens | rgb_patches | (B, 256, 768) |
| 点云 tokens | depth_patches | (B, 256, 512) |
| 质心 | centroids | (B, 256, 3) |
| 绘制后拼接 | cat[painted, depth] | (B, 256, 1280) |
| 融合条件 | cond | (B, 256, 1024) |
| grasp tokens | h | (B, 3, 512) |
| DiT context | cond_proj(cond) | (B, 256, 512) |
| 时间嵌入 | c | (B, 512) |
| 速度场/状态 | v / x | (B, 99) = 3+6+90 |
| MANO 输出 | landmarks / vertices | (B, 21, 3) / (B, 778, 3) |

## 9. 值得注意的设计细节

1. **计算量刻意压低**：所有注意力为 Flash SDPA + QK-norm；DiT 仅 3 个 token（自注意力几乎免费，交叉注意力 3×256 很轻），重计算全在融合层 256 token；
2. **可训练部分只有 PointNeXt + 融合 + DiT**：DINOv2 与 MANO 均冻结（论文 §4.2："Only the PointNeXt encoder, RGB-PC fusion module, and flow transformer are optimized"）；
3. **K 只用于几何运算**（反投影/投影），从不作为可学习输入——跨相机内参泛化（论文 App. C.2）；
4. **β 固定 canonical**：消除采集者手型差异，同一 θ 表示同一抓取；
5. **DiT 输出头 zero-init**：训练起点速度场为零，配合 rectified flow 的线性插值，初期训练稳定；
6. **颜色是辅助信号而非依赖项**：灰度帧训练（数据层面免疫灰度相机）+ `pcl_use_rgb` 构造时开关（结构层面支持纯几何模型），见 §3.3、§3.4。

## 附：关键代码位置

| 文件 | 位置 | 作用 |
|---|---|---|
| `src/models/encoders.py:12-43` | `DINOv2Encoder` | 冻结 RGB 编码 |
| `src/models/encoders.py:46-82` | `PointNeXtEncoder` | 点云编码封装 |
| `src/models/pointnext.py:195-285` | `PointNeXt` | SA/InvResMLP/FP U-Net |
| `src/models/pointnext.py:29` | `_fps_indices` | 最远点采样（FPS） |
| `src/models/pointnext.py:218` | stem 构造 | `6 if use_rgb else 3` 静态维度分支 |
| `src/models/pointnext.py:266-270` | stem forward | 颜色拼接 / 纯几何两路径 |
| `src/models/grasp_model.py:36-46` | 配置级联 | pointpainting / 2D point / pcl_use_rgb 依赖处理 |
| `src/models/fusion.py:21-48` | `FourierPosEmbed` | RFF + 线性 bypass |
| `src/models/fusion.py:123-161` | `_project_centroids` / `_paint` | 点绘制 |
| `src/models/fusion.py:163-215` | `PatchFusion.forward` | 融合主流程 |
| `src/models/grasp_flow.py:28-162` | `TokenPerGroupDiT` | flow 去噪网络 |
| `src/models/grasp_flow.py:211-226` | `sample` | 50 步 Euler 采样 |
| `src/models/transformer.py:140-196` | `AdaLNCrossAttnBlock` | DiT 块（AdaLN + 交叉注意） |
| `src/models/mano.py:48-101` | `decode_mano_params` / `forward` | 99D → MANO mesh |
| `src/models/grasp_model.py:129-176` | `encode_scene` | 编码+融合总装 |
| `src/models/grasp_model.py:206-229` | `forward` | 训练损失路径 |
