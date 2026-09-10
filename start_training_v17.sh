#!/bin/bash

# 启动v17训练脚本
# 方案A：激进数据增强

cd /root/code/HUG-for-Recon-Gen

echo "=========================================="
echo "启动 HUG 训练 - v17 激进数据增强"
echo "=========================================="
echo ""
echo "配置文件: configs/train_handrecon_v17_aggressive_aug.yaml"
echo "输出目录: /root/code/vepfs/HUG-for-Recon-Gen/hand_recon/20260909_v17_aggressive_aug"
echo "日志文件: /root/code/vepfs/HUG-for-Recon-Gen/logs/hand_recon/20260909_v17_aggressive_aug.jsonl"
echo ""
echo "核心改进："
echo "  ✅ Depth增强: 50% → 85% 概率"
echo "  ✅ Pointcloud dropout: 10% → 20%"
echo "  ✅ 启用Affine变换"
echo ""
echo "预期效果："
echo "  - Train/Val Gap: 12mm → 5-8mm"
echo "  - Val PA-MPJPE: 7.5mm → 6.0-6.5mm"
echo ""
echo "预计训练时间: ~11小时"
echo "=========================================="
echo ""

# 检查配置文件
if [ ! -f "configs/train_handrecon_v17_aggressive_aug.yaml" ]; then
    echo "❌ 错误: 配置文件不存在"
    exit 1
fi

# 创建输出目录
mkdir -p /root/code/vepfs/HUG-for-Recon-Gen/hand_recon/20260909_v17_aggressive_aug
mkdir -p /root/code/vepfs/HUG-for-Recon-Gen/logs/hand_recon

echo "✅ 配置检查通过"
echo ""
echo "开始训练..."
echo ""

# 启动训练
torchrun --nproc_per_node=4 -m src.train \
  --config configs/train_handrecon_v17_aggressive_aug.yaml

echo ""
echo "训练完成或中断"
