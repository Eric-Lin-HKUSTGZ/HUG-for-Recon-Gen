# DexYCB train 离线条件缓存

在服务器 `/root/code/HUG-for-Recon-Gen` 中运行。数据来源为
`splits_v2/dexycb_train.clean.txt`，共 394,193 张。输入为原始 RGB，依次使用
当前 YOLO detector 和 RTMPose-m Hand5；不使用 GT 生成 bbox 或关键点。

## 全量运行

```bash
cd /root/code/HUG-for-Recon-Gen
/root/code/vepfs/miniconda3/envs/hug_mediapipe/bin/python -u -m src.cache_train_conditions \
  --output /root/code/vepfs/HUG-for-Recon-Gen/condition_cache/dexycb_train_detector_rtmpose_v1 \
  --launch --workers 4 --batch-size 64 --chunk-size 1024
```

这是前台命令，请在你自己的 tmux/screen 会话中运行。4 个 worker 使用 GPU 0–3，
每 1024 张原子保存一个数据块。每卡日志为 `worker_0.log` 至 `worker_3.log`，
进度为 `progress_0.json` 至 `progress_3.json`。

中断后执行完全相同的命令即可续跑；已校验的数据块会被跳过。
输入清单、模型权重、生成脚本或关键参数变动时，元数据检查会拒绝混用旧缓存，
应选择新的输出目录。读取或推理异常会使任务失败，不会伪装成 detector 漏检。

结束后自动按 train 清单顺序合并为 `conditions.npz`，校验全部样本，生成
`summary.json`，其中 `status: complete`、`n_samples: 394193` 表示全量完成。
`metadata.json` 记录数据清单、权重和脚本哈希以及坐标约定。

## 缓存字段

| 字段 | 形状 | 含义 |
|---|---|---|
| sample / index | N | 数据集根目录下的相对样本名 / 清单中的行索引 |
| image_size_wh | N×2 | 原图宽、高 |
| detector_bbox_xyxy | N×4 | detector 原始框；漏检时为零，须结合 detector_hit 使用 |
| crop_bbox_xyxy | N×4 | 与现有部署一致的 1.5 倍正方形框；漏检时为全图框 |
| detector_hit / detector_score / detector_class | N | 是否检测到手、框置信度、类别；漏检类别为 -1 |
| keypoints_xy | N×21×2 | RTMPose 原图坐标；漏检时为零 |
| keypoint_scores | N×21 | RTMPose 原始逐点分数，未阈值过滤、未截断 |
| pose_returned | N | 是否成功产生 RTMPose 结果 |

xy 和 xyxy 都是原图像素坐标。关节顺序为 wrist、thumb×4、index×4、middle×4、
ring×4、pinky×4。原始分数方便后续训练选择阈值，低分关键点仍保留坐标。
detector 选择最高分右手框，没有右手框时选最高分其他手框；conf=0.25、iou=0.7，
imgsz=512（匹配当前 detector 权重携带的部署默认值）。
RTMPose 输入框与当前部署一致：已经扩大 1.5 倍的框，再通过其既有预处理流程。

读取示例：

```python
import numpy as np
with np.load('/root/code/vepfs/HUG-for-Recon-Gen/condition_cache/dexycb_train_detector_rtmpose_v1/conditions.npz',
             allow_pickle=False) as cache:
    stems = cache['sample']
    keypoints = cache['keypoints_xy']
    scores = cache['keypoint_scores']
    hits = cache['detector_hit']
```

NPZ 按列压缩，训练时应一次性载入所需数组，按 sample 建索引，避免每次取样解压整列。
缓存准备阶段不做随机 skeleton 丢弃；后续训练再施加 10% dropout。

## Smoke test

```bash
cd /root/code/HUG-for-Recon-Gen
/root/code/vepfs/miniconda3/envs/hug_mediapipe/bin/python -u -m src.cache_train_conditions \
  --output /root/code/vepfs/HUG-for-Recon-Gen/condition_cache/dexycb_train_smoke_512_20260915 \
  --launch --limit 128 --chunk-size 32 --batch-size 16 --workers 4
/root/code/vepfs/miniconda3/envs/hug_mediapipe/bin/python -u -m scripts.verify_condition_cache_smoke \
  --output /root/code/vepfs/HUG-for-Recon-Gen/condition_cache/dexycb_train_smoke_512_20260915
```

验证内容：4 卡推理和合并、样本顺序/字段完整性、与现有在线 loader 的框和原图关键点对齐、
强制 detector 漏检、拒绝错位 sample ID、重跑不改写已有数据块。

批量 GPU 推理不保证与单张推理逐位相同。验证会分别报告 detector 框差异、
端到端关键点差异，以及固定同一个缓存框后的 RTMPose 差异。
极小的框差异会通过图像重采样和 SimCC 最大值解码放大为少量像素差异；
这不改变缓存的原图坐标定义。固定框后测试容差为 0.5 px，框容差为 0.05 px。
