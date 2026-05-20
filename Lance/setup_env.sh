#!/bin/bash

# install runtime dependencies for Lance inference
# 用法：./multi_pip_install.sh [python_path]
# 遇到任何错误会立即退出。

set -euo pipefail  # 启用严格模式，任何错误立即退出

# --- 配置区 ---
PYTHON=${1:-python3}
TIMEOUT=300

KEY_PACKAGES=(
    "torch==2.5.1+cu124"
    "torchvision==0.20.1+cu124"
)

# --- 主流程 ---
# 卸载pynvml（如果存在）
echo ">>> 开始卸载pynvml..."
$PYTHON -m pip uninstall -y pynvml || true

echo ">>> 开始安装关键软件包..."
for pkg in "${KEY_PACKAGES[@]}"; do
    echo "--- 正在安装: $pkg ---"
    timeout $TIMEOUT $PYTHON -m pip install --upgrade --no-cache-dir "$pkg" --index-url https://download.pytorch.org/whl/cu124
done

echo ">>> 开始从requirements.txt安装推理依赖..."
timeout $TIMEOUT $PYTHON -m pip install --upgrade --no-cache-dir -r requirements.txt

# 3. 成功结束
echo "✓ 所有包均已成功安装或更新。"
