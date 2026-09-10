# HUG 数据归一化（norm_stats）分析

> 分析对象：`assets/norm_stats.json` 及其在 `src/models/grasp_flow.py` 中的使用。
> 结论先行：99D grasp 状态的归一化不是通用的"神经网络好习惯"，而是 **flow matching 的硬性要求**；归一化统计量**只在训练集上计算一次**，测试集既不参与统计、也没有任何测试数据被归一化。

## 1. norm_stats.json 是什么

文件包含**三组**统计量，对应 99D grasp 状态 `x = [t, R6d, θ6d]` 的三段切分（论文 §4.1）：

| 组 | 维度 | 内容 | 实测均值/标准差（前几维） |
|---|---|---|---|
| `translation` | 3 | 手腕平移（米，相机系） | mean=[0.086, −0.016, **0.538**]，std≈[0.156, 0.179, 0.216] |
| `wrist_rot` | 6 | 手腕旋转（6D 连续表示） | std≈[0.373, 0.453, 0.289] |
| `finger_rot` | 90 | 15 个 MANO 手指关节的 6D 旋转 | mean 含 0.78 / −0.52 等大分量 |

归一化方式为**逐组 z-score 标准化**（`grasp_flow.py:108-120`）：

```python
trans   = (x[:, :3]  - trans_mean)   / trans_std
wrist   = (x[:, 3:9] - wrist_mean)   / wrist_std
fingers = (x[:, 9:99]- finger_mean)  / finger_std
```

注意 `translation.mean z ≈ 0.538`：物体典型位于相机前方 0.54 米（第一人称一臂距离），原始数据离零中心非常远。

## 2. 为什么必须归一化

### 2.1 最根本：flow matching 的数学结构要求数据端近似 N(0, I)

训练流程（`grasp_flow.py:236-249`）：

```python
x_norm = self.denoise_fn.normalize(x_start)  # GT → 归一化空间
eps    = torch.randn_like(x_norm)            # 噪声端 ~ N(0, I)
x_t    = (1 - t_) * x_norm + t_ * eps        # 数据端与噪声端线性插值
target = eps - x_norm                        # 速度场目标
```

Flow matching 在**数据分布与标准高斯之间**插值构建 ODE，隐含假设数据端也大致零均值、单位方差量级。若数据均值在 0.5 开外（如平移 z 分量），插值两端尺度严重不匹配，噪声相对信号微不足道，ODE 几何病态。

### 2.2 三组量纲/尺度悬殊，归一化保证梯度均衡

99D 中混合两类物理量：平移（**米**，std≈0.15–0.22）与旋转（**无量纲** 6D 分量，std≈0.18–0.45）。不归一化时，3 维平移的误差量级会压过 90 维手指关节，模型学会放手腕却学不会捏合手指。

论文 §4.1 原话：*"separate tokens keep geometrically distinct components from over-mixing and **balance the gradient signal across groups**"*——先逐组归一化拉到同一量级，再分别投成三个 token。消融实验（论文 Table 2）中去掉 3D loss 会使 test SR 掉 40+ 个点，侧面印证指尖级精度对这类精细化监督的依赖。

### 2.3 推理闭环：归一化空间积分，出口处反归一化

`grasp_flow.py:211-226`：50 步 Euler 积分全程在归一化空间进行，积出干净状态后 `denormalize` 回**米制相机系**，再交 MANO 解码。因此 norm_stats 是模型权重的一部分——`inference.py:107` 直接从 checkpoint 读取 `norm_stats`，保证训练/推理使用同一份统计量。

## 3. 归一化施加在谁身上（训练 vs 测试）

两个子问题必须分开：**统计量从哪算**、**归一化操作作用于谁**。

### 3.1 统计量：只用训练集算，一次算好，全程冻结

- 在 train split（128 万条带标注样本）上离线计算，随 checkpoint 存储；
- 测试集**不参与**计算——也算不了，因为测试集 `grasp = None`（标注被扣留，见 README）；
- 原则上也不能用被评数据统计，否则构成数据泄漏。

### 3.2 归一化操作：训练和测试走两条不同的路

关键前提：**norm_stats 只管 99D grasp 状态**，不管模型输入（RGB 走 DINOv2 自带 transform；点云是米制 XYZ）。

| 阶段 | 谁被 normalize | 谁被 denormalize | 用的统计量 |
|---|---|---|---|
| 训练 | 训练集 GT grasp | （算 3D loss 时）预测的 x0 | 训练集统计量 |
| 测试/推理 | **无**——测试样本没有 grasp，输入 RGB-D 不经此归一化 | 模型自己生成的预测 | **同一份**训练集统计量 |

推理代码（`grasp_flow.py:211-226`）显示流程从纯噪声出发，天然处于归一化空间，无需归一化任何输入：

```python
x = torch.randn(b, self.d_mano, ...)   # 起点即 N(0,I)
for i in reversed(range(steps)):       # ODE 积分，全程归一化空间
    ...
x = self.denoise_fn.denormalize(x)     # 唯一出口：变回米制
```

### 3.3 边界情况

若用带 GT 的数据做评估（如 train 留出集），GT 也必须用**同一份训练统计量** normalize——任何情况下都不会、也不该使用被评数据自己的统计量。

## 4. 一句话总结

> "归一化"在 HUG 中不是对所有数据集做一遍预处理，而是**模型内部的工作坐标系**：训练时把 GT 拽进该坐标系（normalize），推理时从噪声出发在该坐标系内积分、出口处拽回真实单位（denormalize）。进出用同一把尺子——训练集算出的 mean/std——它随 checkpoint 一起存储，保证训练与推理永远一致。

## 附：关键代码位置

| 文件 | 位置 | 作用 |
|---|---|---|
| `assets/norm_stats.json` | — | 三组 mean/std |
| `src/utils/data_keys.py:13` | `NORM_STATS_FILE` | 文件路径常量 |
| `src/models/grasp_flow.py:90-105` | `_register_norm_stats` | 注册为 buffer（随模型保存/加载） |
| `src/models/grasp_flow.py:108-120` | `normalize` / `denormalize` | 逐组 z-score 正/反变换 |
| `src/models/grasp_flow.py:236-249` | `forward` | 训练：normalize(GT) → 插值 → 速度目标 |
| `src/models/grasp_flow.py:211-226` | `sample` | 推理：噪声出发 → Euler 积分 → denormalize |
| `src/models/grasp_flow.py:199-209` | `recover_x0` | 3D loss 用：恢复 x0 后 denormalize |
| `src/inference.py:107` | — | 从 checkpoint 读取 norm_stats |
