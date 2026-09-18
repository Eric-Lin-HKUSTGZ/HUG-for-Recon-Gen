#!/usr/bin/env bash
set -euo pipefail

# 在无外网服务器上准备 RTMPose-m Hand5。依赖包和权重均从 vepfs 读取，
# 不会访问 Hugging Face、OpenMMLab 或 PyPI。
archive="${MMPOSE_ARCHIVE:-/root/code/vepfs/mmpose-v1.3.2.tar.gz}"
third_party_root="${THIRD_PARTY_ROOT:-/root/code/vepfs/third_party}"
source_dir="${MMPOSE_SOURCE_DIR:-${third_party_root}/mmpose-1.3.2}"
pose_python="${POSE_PYTHON:-/root/code/vepfs/miniconda3/envs/pose/bin/python}"
checkpoint="${RTMPOSE_CHECKPOINT:-/root/code/vepfs/HUG-for-Recon-Gen/rtmpose/rtmpose-m-hand5-256x256.pth}"

archive_sha256="fcf1e7ac3f5bbb6f3f98849bb579d63184cb3c7ba944ee1a894dc52084d901b3"
checkpoint_sha256="b74fb5941684fe13c337b8d4fce644293e12903fed5407f8b27921f107dc6003"

if [[ ! -x "$pose_python" ]]; then
  echo "pose 环境 Python 不存在或不可执行：$pose_python" >&2
  exit 1
fi
if [[ ! -f "$archive" ]]; then
  echo "MMPose 离线源码包不存在：$archive" >&2
  exit 1
fi
if [[ ! -f "$checkpoint" ]]; then
  echo "RTMPose-m Hand5 权重不存在：$checkpoint" >&2
  exit 1
fi

printf '%s  %s\n' "$archive_sha256" "$archive" | sha256sum --check --status
printf '%s  %s\n' "$checkpoint_sha256" "$checkpoint" | sha256sum --check --status

mkdir -p "$third_party_root"
if [[ ! -f "$source_dir/mmpose/__init__.py" ]]; then
  tar -xzf "$archive" -C "$third_party_root"
fi

# 当前 pose 环境安装的是无 CUDA 扩展的 mmcv-lite。RTMPose 不需要 EDPose，
# 但 MMPose 的 heads/__init__.py 会无条件导入 EDPose 并连带导入 mmcv._ext。
# 将这个无关模块改成可选导入，保持 RTMPose 本身实现不变。下面按代码块
# 边界重写，因此脚本可重复执行，也能修复被中断或重复修改的旧环境。
heads_init="$source_dir/mmpose/models/heads/__init__.py"
"$pose_python" - "$heads_init" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
lines = path.read_text().splitlines()
end = next((i for i, line in enumerate(lines) if line.startswith("__all__")), None)
if end is None:
    raise RuntimeError(f"无法识别 {path} 中的 __all__")
candidates = [
    i for i, line in enumerate(lines[:end])
    if line.strip() == "try:" or "transformer_heads import EDPoseHead" in line
]
if not candidates:
    raise RuntimeError(f"无法识别 {path} 中的 EDPoseHead 导入块")
start = min(candidates)
replacement = [
    "try:",
    "    from .transformer_heads import EDPoseHead",
    "except (ImportError, ModuleNotFoundError):",
    "    EDPoseHead = None",
    "",
]
path.write_text("\n".join(lines[:start] + replacement + lines[end:]) + "\n")
PY

"$pose_python" -m pip install --no-index --no-deps --no-build-isolation -e "$source_dir"

TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 "$pose_python" - "$source_dir" "$checkpoint" <<'PY'
from pathlib import Path
import sys

source_dir = Path(sys.argv[1])
checkpoint = Path(sys.argv[2])
config = source_dir / "configs/hand_2d_keypoint/rtmpose/hand5/rtmpose-m_8xb256-210e_hand5-256x256.py"
from mmpose.apis import init_model
import mmpose

assert config.is_file(), config
assert checkpoint.is_file(), checkpoint
print(f"MMPose {mmpose.__version__} 可用")
print(f"配置：{config}")
print(f"权重：{checkpoint}")
print("RTMPose-m Hand5 离线环境准备完成")
PY
