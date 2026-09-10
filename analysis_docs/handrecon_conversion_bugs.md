# 手部重建数据转换 Bug 排查记录

> 日期：2026-08-29
> 状态：**已定位并修复，验证通过**；待全量重转换 + 重训
> 影响范围：`scripts/conversion_common.py`、`scripts/convert_dexycb.py`、
> `scripts/convert_ho3d.py` 及全部已转换数据（DexYCB ~50.8 万 pkl、HO3D ~8.2 万 pkl）

## 1. 背景与症状

v1（30k 步，恒定 LR）与 v2（25k 步，LR 衰减 + 2D 监督）训练后，官方测试集指标出现
无法由训练配置解释的异常：

| 指标 | DexYCB s0_test | HO3D_v3 eval | 异常点 |
|---|---|---|---|
| PA-MPJPE | 6.8-7.7 mm | 24.0 mm | HO3D 始终卡在 24mm（step 1000 起就不变） |
| MPJPE | 58-64 mm | 169-170 mm | 与 PA 差 7-9 倍，平移/深度系统性偏差 |
| MPVPE（修复前） | 133 mm | **500 mm** | 顶点比关节差 4 倍，物理上不可能 |

其他线索：

- **预训练 HUG 模型（未微调）在 HO3D eval 上 PA-MPJPE = 15.8mm，优于我们
  微调后的 24.0mm** -- finetune 把 HO3D 能力"训没了"
- DexYCB 上 query 点深度均值 1.62m，而手部实际在 0.83m
- HO3D eval 平移误差存在 -86mm 的系统性 x 方向偏差

## 2. Bug A：MANO 平移参考点错位 ~9.6cm（致命）

### 根因

MANO 模板中**腕关节不在原点**：模板腕关节位置为

```
J0 = J_regressor @ v_template = [0.0957, 0.0064, 0.0062]  (米，掌心区域)
```

两个数据集的官方平移标注（DexYCB `pose_m[48:51]`、HO3D `handTrans`）指的都是
**MANO 模板原点**位置，官方关节满足：

```
official_wrist = MANO_translation + J0   (J0 不经手部旋转，直接相加)
```

多序列/多相机/多帧实测该规则误差 **0.7-5mm**（对比直接平移的 93-101mm）。

但转换代码用「腕部居中关节 + 原始平移」，导致**每只手的 GT 相对图像证据偏移
~9.6cm**（方向随手部朝向旋转）。

### 证据链

1. 逐帧对比官方 `joint_3d`：我们的腕部位置与官方差 (+96, -1, +33)mm，
   **逐帧完全恒定** → 系统性偏移而非估计噪声
2. 误差向量换算到世界系 = (-0.095, -0.006, -0.006)m，**与 J0 精确吻合**
3. 转换产物的 mask 与官方手部 seg 的 **IoU 仅 0.01**（偏移 ~90px）
4. query 点从错位 mask 采样 → 落在背景（深度 1.62m vs 手部 0.83m）
5. PCL 按错误的 query 中心裁剪 → **装的是背景点云，不是手**
   （深度通道在 DexYCB 训练中等同虚设）

### 修复

- `scripts/conversion_common.py`：新增 `mano_wrist_offset()`（从 MANO 资产
  计算 J0 并缓存）
- `scripts/convert_dexycb.py`：`t_cam = R_e @ (trans_w + J0) + t_e`
- `scripts/convert_ho3d.py`：`t_cam = ho3d_to_std(handTrans + J0)`
- `src/models/mano.py`：保持 `center_idx=0` 并补充注释（约定：99D 的 t
  **就是腕部位置**，模型侧无需改动）

## 3. Bug B：6D 旋转编码与模型解码不兼容（致命）

### 根因

编码端（`scripts/conversion_common.py:rotmat_to_6d`）输出**列拼接**：

```python
concatenate([R[:, 0], R[:, 1]])   # [R00,R10,R20, R01,R11,R21]
```

解码端（`src/utils/transform_utils.py:six_d_to_rotation_matrix`，模型 MANO
前向使用）做 `six_d.reshape(3, 2)`（行优先）后 Gram-Schmidt 两列，期望的是：

```python
R[:, :2].flatten()                 # [R00,R01,R10,R11,R20,R21]
```

两者不是互逆运算：**往返误差 ~1.7（应≈0）**。所有存储的腕部旋转（R_6d）与
15 个手指旋转（pose_6d）经解码后全部被扰乱，实测为约 **70° 的恒定朝向错误 +
21mm PA 残差**。

### 修复

`rotmat_to_6d` 改为 `R[:, :2].reshape(-1)`，往返误差降至 **1.2e-07**。

## 4. 为什么两个 Bug 长期未暴露

**自洽性掩盖**：转换端与模型端使用同一套错误约定，训练 / val / DexYCB 测试
全部自洽：

- PA 指标（Procrustes 对齐）天然消除恒定平移偏移
- 旋转扰乱对 pred 与 GT 一致，相对比较不可见
- 训练中的 val 用的是我们自己的转换数据，同样自洽

只有**官方 GT**（HO3D evaluation_xyz.json、DexYCB 官方 labels）使用正确约定，
异常只在那里显现。预训练模型（正确约定数据训练）在 HO3D eval 上反而更优，
正是数据约定错误的信号。

## 5. 修复验证（端到端重转换 504 帧）

| 指标 | 修复前 | 修复后 |
|---|---|---|
| 关节 vs 官方 joint_3d（腕部） | 93-101 mm | **2-5 mm** |
| mask vs 官方 seg IoU | 0.01 | **0.39** |
| 6D 往返误差 | ~1.7 | **1.2e-07** |

注：landmarks 对官方剩余 ~17mm 为 canonical betas（HUG 约定形状）与受试者
betas 的形状差异，属设计预期（模型 pred/GT 同用 canonical 形状，自洽）；
mask 已改用受试者 betas 生成以更好覆盖真实手部。

## 6. 后续计划

1. [x] DexYCB 全量重转换至 `dexycb_v2_canonical_right`（508,384 pkl）
2. [x] 重算混合 norm_stats（`norm_stats_handrecon_v2.json`，训练 n=477,518）（`scripts/compute_norm_stats.py`，t 与旋转分布均变化）
3. [x] 基于 v2 目录重跑 `scripts/filter_empty_masks.py`（train=394,193，val=21,785，test=76,845）（mask 位置已变）
4. [ ] 重训 25k 步（配置已切换，尚未启动）（LR 衰减 + 2D 监督 + 采样式 val 均已就位）
5. [ ] 重跑官方测试集全量评测

预期收益：深度通道从背景噪声变为真实手部点云（平移/MPJPE 应大幅改善），
旋转姿态与官方 GT 对齐（HO3D eval PA-MPJPE 应显著下降）。

## 附录 A：同轮排查中修复的其他问题

| 问题 | 位置 | 说明 |
|---|---|---|
| MPVPE 顶点缺平移 | `src/models/grasp_model.py:mano_forward` | vertices 未加 t，HO3D eval 上 MPVPE 虚高至 500mm |
| HO3D eval 关节顺序 | `src/dataloader/grasp_dataset.py:HO3D_RAW_TO_STD` | 官方 evaluation_xyz 为 MANO 原始运动学顺序，与 manotorch 输出顺序不同，加载时重排 |
| 假 EMA checkpoint | `src/train.py:save_checkpoint` | EMA 未启动（<ema_start_step）时存入的是预训练初始权重；改为存 null 并回退 |
| val 指标乐观偏差 | `src/train.py:run_val` | 旧 val 走 forward 单步 x0 恢复（~2x 乐观）；改为 50 步 ODE 完整采样、去掉 loss |
| val NCCL 超时 | `src/train.py:build_loaders/setup_ddp` | rank-0 单卡跑采样式 val 超 600s 看门狗；改为多卡分片 + all_reduce，NCCL 超时放宽 30min |
| cuSOLVER 崩溃 | `src/metrics.py:compute_similarity_transform` | 显存压力下 cusolverDnCreate 失败；(B,3,3) SVD 改到 CPU 计算 |
| val 口径版本 | `src/train.py` | checkpoint 存 `val_metric` 版本标记，resume 时口径不一致自动重置 best_val |

## 附录 B：排查过程中被排除的假设

- ~~HO3D 是 ego 数据、中心裁剪错位~~：HO3D 为第三人称（最多 5 相机），
  实测手部基本居中（eval 集 0% 关节出画）
- ~~HO3D eval GT 存在反射/镜像~~：镜像 GT 指标更差（42.9 vs 40.6）
- ~~图像与 GT 帧错位~~：GT 邻帧平滑度 0.25mm，深度-投影一致性通过
- ~~模型输出塌缩~~：预测姿态多样性 35.7mm ≈ GT 32.7mm
- ~~flow 采样噪声~~：8 次重采样 std 仅 1.38mm，取均值无收益
- ~~query 点类型（腕部 vs mask 随机）~~：同域实测无差异（12.78 vs 12.79mm）
- ~~HO3D handPose 的 PCA 解释 / flat_hand_mean / 旋转基变换~~：均不成立；
  最终证实 handPose 处理本身正确，仅平移参考点错误（"指尖错位 10cm"实为
  排查者自己对比时关节顺序不一致的分析假象）
- ~~EMA 步数不足~~：0.999 decay 下 ~5k 步已充分收敛

## 附录 C：教训

1. **与官方标注的端到端对齐验证必须在数据转换完成后立即做**
   （投影 vs seg IoU、官方 joint_3d 直接对比），不能依赖自洽指标
2. 旋转参数化必须写成**往返单元测试**（encode→decode 恒等），
   6D 布局（行优先/列拼接）是经典坑
3. PA（Procrustes）指标会掩盖恒定坐标系错误，绝对指标（MPJPE）同样要看
4. "训练后的模型不如预训练模型"是数据/约定问题的强信号，不应先归因为遗忘
