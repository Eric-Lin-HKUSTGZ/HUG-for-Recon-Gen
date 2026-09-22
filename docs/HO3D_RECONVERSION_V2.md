> 2026-09-20 更新：划分已统一到 splits_v2；旧 splits 和独立 HO3D 划分目录已移除。真实条件缓存现已生成。当前训练入口为 configs/train_ho3d.yaml，smoke 入口为 configs/train_ho3d_smoke.yaml。旧 99D train_ho3d 配置与历史列表的压缩存档位于 /root/code/vepfs/HUG-for-Recon-Gen/ho3d_smoke_20260920/historical_split_lists.tar.gz。

# HO3D 重新转换与缓存操作说明

日期：2026-09-19。以下命令在远程服务器执行。全量转换、正式条件缓存由人工启动；本次只进行了小样本 CPU 检查，没有启动全量转换或训练。

## 1. 本次重新转换解决什么

旧目录为 `/root/code/vepfs/dataset/hand_recon_hug/ho3d`、`ho3d_eval`。不覆盖这两个目录，也不改动 DexYCB 数据。

| 项目 | 旧转换 | 新转换 |
|---|---|---|
| 训练 shape | 保存了真实 `shape_gt`，但 `shape` 和关节点/网格使用固定 HUG shape | `shape`、`shape_gt` 均为该帧原始 `handBeta`，网格也使用该 beta |
| 腕部位置 | `handTrans + 零 shape 模板腕部偏移` | `handTrans + J_regressor(v_template + shapedirs × beta)[0]`，逐帧核对官方腕部 |
| 指尖 | manotorch 默认顶点 | HO3D 官方顶点 `[744,333,444,555,672]`，依次为拇指、食指、中指、无名指、小指 |
| RGB / Depth | 中心裁剪到正方形，缩为 224 | 保留原始分辨率，RGB 保存原 JPG 字节；Depth 解码后保存 uint16 毫米 PNG |
| 相机内参 | 已随旧裁剪缩放 | 保留原始 K；训练时随增强/crop 更新 |
| 推理条件 | 旧 eval 使用 GT wrist 作 query | 不写 GT query，不用 GT mesh 制造部署输入；WiLoR detector + RTMPose 单独缓存 |
| 评测 GT | joints/verts | 继续使用官方 joints/verts，不伪造评测集 shape/pose |
| 完整性 | 旧 clean eval 列表仅 17,224 帧 | 以官方 evaluation.txt 的 20,137 帧及对应 JSON 行号为准，检测失败也保留 |

**重要澄清：并非旧训练 PKL 完全没有真实 shape。问题是标签几何生成没有使用它，而且腕部偏移也未随 shape 更新。**

初步均匀抽查 64 帧时，旧固定腕部偏移的平均误差约 2.54 mm；manotorch 与 HO3D 的四个不同指尖定义带来约 4–6 mm 的平均指尖差异。这些数字是抽查，不是全量统计，也不是训练模型的性能改善预测。完整转换报告会给出全量腕部误差统计。

HO3D 本次按原始右手 MANO 处理；不套用 DexYCB 左手镜像修复。相机转换为 `C=diag(1,-1,-1)`，它是绕 X 轴旋转 180°，不是左手镜像：

- 点坐标：`X_cv = C X_ho3d`，单位米；
- 全局旋转：`R_cv = C R_ho3d`；15 个手指局部旋转不变；
- 6D 旋转：取旋转矩阵前两列，再按行展开；
- 109D 标签：腕部平移 3 + 腕部旋转 6 + 手指旋转 90 + 原始 beta 10；
- 训练 joints 保存标准顺序，eval joints 保存官方 raw 顺序，Dataset 读取 eval 时只重排一次；
- 评测 JSON 与原始 evaluation.txt 索引绑定，禁止先筛选或排序列表再按新索引读取 GT。

## 2. 远程已新增/修改的代码

- `scripts/prepare_ho3d_v2.py`：转换、原始数据对照检查、划分、严格 109D 统计、缓存核验和可视化。
- `scripts/verify_ho3d_loader.py`：CPU 数据加载、原生 MANO 几何回解、shape 梯度、评测无参数标签路径、缓存与增强集成检查。
- `src/dataloader/grasp_dataset.py`：识别 HO3D native schema，提供右手标记和关节点定义标记，支持 eval 无 MANO 参数。
- `src/models/native_mano.py`、`grasp_model.py`：按样本选择 HO3D 官方指尖定义；DexYCB 定义不变。这是输出关节点的取点规则，不是额外模型输入或新预测参数。
- `src/train.py`、`src/eval_test.py`：无参数 eval 传递原生几何元数据；支持数据集条目分别覆盖 `geometry_overlay` 和 `hand_crop`，以免把 DexYCB overlay/cache 用在 HO3D 上。

没有修改 MANO 权重资产，没有启动训练。

## 3. 环境与路径

本地连接：

```bash
ssh -p 58225 root@115.190.90.101
```

以下在同一个远程 Bash 会话中执行。建议放在你自己的持久终端会话中运行。

```bash
cd /root/code/vepfs/repos/HUG-for-Recon-Gen
export PY=/root/code/vepfs/miniconda3/envs/hug_mediapipe/bin/python
export RAW=/root/code/vepfs/dataset/HO3D_v3
export DATA=/root/code/vepfs/dataset/hand_recon_hug
export TRAIN_NEW="$DATA/ho3d_v2_fullres_native"
export EVAL_NEW="$DATA/ho3d_eval_v2_fullres_native"
export SPLITS_NEW="$DATA/splits_v2"
export AUDIT=/root/code/vepfs/HUG-for-Recon-Gen/ho3d_reconversion_v2
export CACHE=/root/code/vepfs/HUG-for-Recon-Gen/condition_cache/ho3d_v2
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
mkdir -p "$AUDIT" "$SPLITS_NEW" "$CACHE"

test -f "$RAW/train.txt"
test -f "$RAW/evaluation.txt"
test -f "$RAW/evaluation_xyz.json"
test -f "$RAW/evaluation_verts.json"
wc -l "$RAW/train.txt" "$RAW/evaluation.txt"
df -h "$DATA"
"$PY" scripts/prepare_ho3d_v2.py --help
```

官方列表当前为 train 83,325 帧、evaluation 20,137 帧。实际计数以脚本去除空行后的报告为准。最终转换约 10.3 万个 PKL，CPU/文件系统负载为主；evaluation JSON 为约 1 GB，读取时会有数 GB 的临时内存开销。

## 4. 小样本冒烟检查

以下 `*_final` 小样本目录是本次已生成并验证的 32 帧结果，覆盖列表不同位置。命令相同可重复运行；已有合格 PKL 会复用。**不要在小样本目录内去掉 `--limit` 做全量转换。**

```bash
"$PY" scripts/prepare_ho3d_v2.py convert \
  --raw-root "$RAW" --split train --limit 32 --workers 2 \
  --output "$AUDIT/smoke_train_final"

"$PY" scripts/prepare_ho3d_v2.py convert \
  --raw-root "$RAW" --split evaluation --limit 32 --workers 2 \
  --output "$AUDIT/smoke_eval_final"

"$PY" scripts/prepare_ho3d_v2.py check \
  --raw-root "$RAW" --dataset-root "$AUDIT/smoke_train_final" \
  --samples 32 --visualize 20 --output "$AUDIT/smoke_train_check"

"$PY" scripts/prepare_ho3d_v2.py check \
  --raw-root "$RAW" --dataset-root "$AUDIT/smoke_eval_final" \
  --samples 32 --visualize 20 --output "$AUDIT/smoke_eval_check"

"$PY" scripts/verify_ho3d_loader.py \
  --dataset-root "$AUDIT/smoke_train_final" \
  --samples-file "$AUDIT/smoke_train_final/samples.txt" \
  --augmentation-config configs/train_handrecon_v32_native_rgbd_gt_aug_lora.yaml \
  --output "$AUDIT/smoke_train_loader"

"$PY" scripts/verify_ho3d_loader.py \
  --dataset-root "$AUDIT/smoke_eval_final" \
  --samples-file "$AUDIT/smoke_eval_final/samples.txt" \
  --output "$AUDIT/smoke_eval_loader"
```

未传 `--cache` 时，loader 检查会创建名为 `TEST_FIXTURE_all_detector_misses.npz` 的全检测失败测试夹具，仅验证 fallback 与几何链路，**不是实际检测结果，不能给正式训练使用**。正式缓存生成后，第 9 步会使用真实缓存再验证。

`check` 生成的 20 张图中，绿色骨架表示官方 GT 投影；这是标签检查图，不是增强图，也没有把 GT 作为部署输入。

## 5. 全量重新转换

通过第 4 步后人工执行。建议顺序运行，便于检查 CPU/I/O 负载。

```bash
"$PY" scripts/prepare_ho3d_v2.py convert \
  --raw-root "$RAW" --split train --workers 8 \
  --output "$TRAIN_NEW"

"$PY" scripts/prepare_ho3d_v2.py convert \
  --raw-root "$RAW" --split evaluation --workers 8 \
  --output "$EVAL_NEW"

cat "$TRAIN_NEW/conversion_report.json"
cat "$EVAL_NEW/conversion_report.json"
```

验收要求：`status=complete`、`failed=0`，训练 `converted=83325`、评测 `converted=20137`。每一训练帧都对照官方 joints 检查，最大关节点误差超过 0.1 mm 会报错；每一 eval 帧检查官方指尖与顶点一致性。

缺 RGB/Depth/meta、空参数、非有限值、几何不匹配都会记录到 `errors`，最终返回非零退出码。**不要把失败报告当作“转换结束”继续下游步骤。**修复源文件后，原命令可续跑。输出目录带转换身份记录；脚本、MANO 资产、列表或 split/limit 改变时拒绝混写，需使用新版本目录。原始 RGB/meta/depth 文件本身应保持只读不变；不会逐个给十万帧源文件做内容哈希。

旧 224 数据已经丢失空间细节，因此 HO3D 本次需要从原始数据重写图像/深度 PKL，不能只给旧 PKL 加一个 shape overlay。

## 6. 生成新划分，抽查原始文件

保留旧 `ho3d_val.clean.txt` 中的验证序列身份，以这些序列划分新转换的全部训练帧，不沿用旧 clean 的逐帧删除结果。评测使用全部官方 eval 帧。

```bash
# 2026-09-20 已完成划分并合并到 splits_v2，不必再次生成。
# 训练 77564 / 验证 5761 / 官方评测 20137 帧。

cat "$SPLITS_NEW/ho3d_split_report.json"

"$PY" scripts/prepare_ho3d_v2.py check \
  --raw-root "$RAW" --dataset-root "$TRAIN_NEW" \
  --samples 512 --visualize 20 --output "$AUDIT/train_geometry_check"

"$PY" scripts/prepare_ho3d_v2.py check \
  --raw-root "$RAW" --dataset-root "$EVAL_NEW" \
  --samples 512 --visualize 20 --output "$AUDIT/eval_geometry_check"
```

产生 `ho3d_train.txt`、`ho3d_val.txt`、`ho3d_eval.txt`、`ho3d_trainval.txt`。最后一个仅供共同缓存全部有标注帧使用，**不用于统计或作为训练集**。此划分保留现有实验的按序列留出策略，并不声称达到按对象/受试者完全独立的额外协议。官方 evaluation 仅用于最终测试，不用于挑选 checkpoint。

## 7. 只用训练划分计算 109D 归一化统计

```bash
"$PY" scripts/prepare_ho3d_v2.py stats \
  --dataset-root "$TRAIN_NEW" \
  --samples-file "$SPLITS_NEW/ho3d_train.txt" \
  --output "$SPLITS_NEW/norm_stats_ho3d_v2.json"

cat "$SPLITS_NEW/norm_stats_ho3d_v2.audit.json"
```

检查 `n` 与 `split_report.json` 的 train 数量相同。任何文件读取/参数错误直接失败，不静默跳过；stats 含真实 shape 的 10 维。

这个文件适合 HO3D-only 实验。后续 DexYCB+HO3D 混训可以明确沿用当前 DexYCB native 的固定归一化，或另算混合训练统计；**不能把 DexYCB 旧 PKL 的未修正参数拿来合并统计**。若重算，应从 DexYCB native overlay 的 `params.npy` 按其 train list 取值，加上 HO3D 新训练参数；val/eval 都不参与。改变统计后不要当成原 checkpoint 的无缝 resume。

## 8. 缓存检测框与 RTMPose

先确认 GPU 空闲。本次检查时 4 张 GPU 都在运行训练，没有替你启动 GPU 缓存任务。

```bash
nvidia-smi

CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY" -m src.cache_train_conditions \
  --dataset-root "$TRAIN_NEW" \
  --samples-file "$SPLITS_NEW/ho3d_trainval.txt" \
  --output "$CACHE/trainval" \
  --workers 4 --batch-size 64 --chunk-size 1024 --launch

CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY" -m src.cache_train_conditions \
  --dataset-root "$EVAL_NEW" \
  --samples-file "$SPLITS_NEW/ho3d_eval.txt" \
  --output "$CACHE/evaluation" \
  --workers 4 --batch-size 64 --chunk-size 1024 --launch

cat "$CACHE/trainval/summary.json"
cat "$CACHE/evaluation/summary.json"
```

如果只有 1 张空闲卡，把 `CUDA_VISIBLE_DEVICES` 改为该卡物理编号，并使用 `--workers 1`。缓存的 worker rank 是可见卡的逻辑编号。中断后用**相同参数**重跑，可复用完成的 chunks；更换数据、权重或参数需新缓存目录。

缓存模型沿用现有 DexYCB 流程：

- detector：`/root/code/vepfs/GPGFormer/weights/detector.pt`；
- RTMPose 配置：`/root/code/vepfs/third_party/mmpose-1.3.2/configs/hand_2d_keypoint/rtmpose/hand5/rtmpose-m_8xb256-210e_hand5-256x256.py`；
- RTMPose 权重：`/root/code/vepfs/HUG-for-Recon-Gen/rtmpose/rtmpose-m-hand5-256x256.pth`。

缓存包含原图宽高、原始检测框、扩大 1.5 倍的正方形 crop、21 个预测点、置信度、检测命中标记。只读 RGB，不用 GT 标签构造条件。train/val 共同生成一个只读缓存是允许的，因为这里没有学习参数或汇总标签；Dataset 按 stem 选择对应行。

检测失败：保留该帧，缓存零关节点/零置信度，RGB 回退全图；不使用 GT 补框，不剔除难例。应记录 miss rate，它影响最终重建性能。

## 9. 缓存核验、随机可视化、真实 loader 检查

```bash
"$PY" scripts/prepare_ho3d_v2.py cache-check \
  --dataset-root "$TRAIN_NEW" --samples-file "$SPLITS_NEW/ho3d_trainval.txt" \
  --cache "$CACHE/trainval/conditions.npz" \
  --samples 512 --visualize 20 --output "$AUDIT/train_cache_check"

"$PY" scripts/prepare_ho3d_v2.py cache-check \
  --dataset-root "$EVAL_NEW" --samples-file "$SPLITS_NEW/ho3d_eval.txt" \
  --cache "$CACHE/evaluation/conditions.npz" \
  --samples 512 --visualize 20 --output "$AUDIT/eval_cache_check"

"$PY" scripts/verify_ho3d_loader.py \
  --dataset-root "$TRAIN_NEW" --samples-file "$SPLITS_NEW/ho3d_train.txt" \
  --cache "$CACHE/trainval/conditions.npz" --samples 128 \
  --augmentation-config configs/train_handrecon_v32_native_rgbd_gt_aug_lora.yaml \
  --output "$AUDIT/train_real_cache_loader"

"$PY" scripts/verify_ho3d_loader.py \
  --dataset-root "$EVAL_NEW" --samples-file "$SPLITS_NEW/ho3d_eval.txt" \
  --cache "$CACHE/evaluation/conditions.npz" --samples 128 \
  --output "$AUDIT/eval_real_cache_loader"
```

缓存可视化：**红框=检测器原始框；绿框=扩大后的 RGB crop；紫色骨架=RTMPose 预测（绘制置信度 ≥0.1 的点）；白字 hit=False=检测失败。**绿色框可以超出图像边界，裁剪时补边。每组图片附 README；缓存图不叠 GT，以免混淆。

检查器验证缓存行数/顺序、列表哈希、数据集根目录、数值有限性、检测失败的回退字段；随机抽查图片宽高。loader 检查验证真实读取、增强后的 K 与 GT 投影、109D 回解误差及 shape 梯度。没有重跑检测器，因此不能证明任意外部缓存的像素坐标语义；应配合图片人工检查。

## 10. 训练配置如何接入

模型必须是 `d_mano: 109`、`mano_geometry: native_side_v1`，并启用 `hand_crop`。HO3D 新 PKL 已有正确几何，`geometry_overlay: null`；不能继承 DexYCB 的 overlay 路径。

下面是**需要合入新实验配置的条目示例，不是完整可启动训练配置**。没有替你改动正在使用的 v32 配置或启动新的训练。

```yaml
# trainer.data.datasets 中新增 HO3D 条目
- path: /root/code/vepfs/dataset/hand_recon_hug/ho3d_v2_fullres_native
  train_samples: /root/code/vepfs/dataset/hand_recon_hug/splits_v2/ho3d_train.txt
  val_samples: /root/code/vepfs/dataset/hand_recon_hug/splits_v2/ho3d_val.txt
  geometry_overlay: null
  hand_crop:
    enabled: true
    keypoint_source: rtmpose_cache
    train_keypoint_source: rtmpose_cache
    eval_keypoint_source: rtmpose_cache
    condition_cache: /root/code/vepfs/HUG-for-Recon-Gen/condition_cache/ho3d_v2/trainval/conditions.npz

# trainer.val.datasets 中使用训练池留出的 HO3D 序列
- name: ho3d_val
  path: /root/code/vepfs/dataset/hand_recon_hug/ho3d_v2_fullres_native
  samples: /root/code/vepfs/dataset/hand_recon_hug/splits_v2/ho3d_val.txt
  geometry_overlay: null
  hand_crop:
    enabled: true
    keypoint_source: rtmpose_cache
    train_keypoint_source: rtmpose_cache
    eval_keypoint_source: rtmpose_cache
    condition_cache: /root/code/vepfs/HUG-for-Recon-Gen/condition_cache/ho3d_v2/trainval/conditions.npz

# trainer.test.datasets 中使用官方 evaluation
- name: ho3d_eval
  path: /root/code/vepfs/dataset/hand_recon_hug/ho3d_eval_v2_fullres_native
  samples: /root/code/vepfs/dataset/hand_recon_hug/splits_v2/ho3d_eval.txt
  geometry_overlay: null
  hand_crop:
    enabled: true
    keypoint_source: rtmpose_cache
    train_keypoint_source: rtmpose_cache
    eval_keypoint_source: rtmpose_cache
    condition_cache: /root/code/vepfs/HUG-for-Recon-Gen/condition_cache/ho3d_v2/evaluation/conditions.npz
```

注意同时显式覆盖 train/eval keypoint source：v31/v32 的 GT oracle 设置不能自动继承到这个部署口径实验。GT 关键点消融应另设配置并明确标注，不能与预测关键点结果混报。

数据增强仍放在在线 Dataset。原图上的缓存坐标跟随同一个仿射变换，RGB/Depth/K/2D 标签同步更新；3D 标签保持相机坐标不变。修改亮度、颜色、现有仿射采样范围，不需要重做原图条件缓存。不过“变换原图预测点”并不等于“在增强图上重跑 RTMPose”；这是当前明确采用的缓存训练策略。未来若加入水平翻转、透视变换或非刚性变形，需要先增加对应标签/缓存映射支持。

新全量官方 eval 与旧 17,224 帧 clean eval 不是同一测试集合，分数不能直接当成仅 shape 修复的消融对比；若要比较，另取二者相同 stem 子集，并同时报告完整官方评测结果。

## 11. 输出验收清单

1. 两份 conversion report 都 complete、failed=0，数量与官方列表一致。
2. 训练/验证按原有验证序列划分，无 stem 重叠；eval 顺序与官方 JSON 对齐。
3. 两组 geometry check 均通过，各至少 20 张随机可视化。
4. stats 的 n 等于 train 数量，包含 109D 四组统计，不含 val/eval。
5. 两份 condition cache complete、无缺行，检测失败也在列表里。
6. 两组 cache check 与真实缓存 loader check 通过，各 20 张缓存图人工检查。
7. 训练配置使用新根目录/列表/cache，HO3D 不挂 DexYCB overlay；GT oracle 与部署评测口径明确区分。

全部完成后才进行 HO3D-only 基线或 DexYCB+HO3D 混训实验。


## 12. 本次已执行的验证结果

- 原始训练列表均匀选取 32 帧，转换 32/32、失败 0；逐帧 MANO 与官方 joints 最大误差 0.0001681 mm，旧固定模板腕部偏移在这 32 帧的平均误差 2.62186 mm。
- 原始 evaluation 列表均匀选取 32 帧，转换 32/32、失败 0；官方指尖与官方顶点最大误差 0.0001367 mm。
- 训练/评测各 32 帧与原始 RGB、Depth、K、GT 对照检查通过，各随机可视化 20 张并检查了拼图。图像在 smoke_train_check、smoke_eval_check 下；绿色 GT 骨架与可见手部位置相符，遮挡关节点按官方标签保留。
- 训练 loader 使用 v32 增强配置及“检测全部失败”的测试缓存：32 帧通过，109D 回解关节点最大误差 0.0001681 mm，网格最大误差 0.0000760 mm，shape 梯度与增强后投影检查通过。
- eval loader 的 32 帧无 MANO 标签路径通过；未读取或填充 GT shape。
- 15 项已有 native MANO、geometry overlay、augmented condition cache 测试通过；另检查了按样本切换指尖定义保持 DexYCB 不变、保留序列划分、拒绝未完成转换，以及缓存检查器的全检测失败路径。
- 全量转换、真实检测/RTMPose 缓存、真实缓存增强 loader 检查及完整模型训练尚未执行。真实缓存命令应在 GPU 空闲时人工启动。
