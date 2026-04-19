#!/bin/bash
# ============================================================================
# Colab Environment Setup — Avatar 3D Pipeline
# ============================================================================
# Usage (paste into first Colab cell):
#   !bash scripts/setup_colab.sh
#
# What this script does:
#   1. Installs Python dependencies
#   2. Installs PyTorch Geometric (for GNN)
#   3. Installs PyTorch3D (for 3D rendering)
#   4. Creates symlinks to Google Drive weights/data
#   5. Verifies GPU availability
# ============================================================================

set -e

echo "============================================"
echo "🚀 Avatar 3D Pipeline — Colab Setup"
echo "============================================"

# --- 1. Detect CUDA version ---
CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "none")
TORCH_VERSION=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "none")
echo "📌 PyTorch: $TORCH_VERSION | CUDA: $CUDA_VERSION"

# --- 2. Install core dependencies ---
echo ""
echo "📦 Installing core dependencies..."
pip install -q \
    fastapi uvicorn[standard] \
    trimesh pyglet \
    opencv-python-headless \
    Pillow \
    scikit-image \
    lpips \
    tqdm \
    tensorboard

# --- 3. Install PyTorch Geometric ---
echo ""
echo "📦 Installing PyTorch Geometric..."
TORCH_SHORT=$(python -c "import torch; v=torch.__version__.split('+')[0]; print(v)")
pip install -q torch-geometric
pip install -q pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f "https://data.pyg.org/whl/torch-${TORCH_SHORT}+${CUDA_VERSION//.}.html" 2>/dev/null || \
    echo "⚠️  Some PyG extensions failed — GNN fallback mode will be used"

# --- 4. Install PyTorch3D (optional, for rendering) ---
echo ""
echo "📦 Installing PyTorch3D..."
pip install -q "git+https://github.com/facebookresearch/pytorch3d.git@stable" 2>/dev/null || \
    echo "⚠️  PyTorch3D install failed — rendering utils will be limited"

# --- 5. Setup Google Drive symlinks ---
DRIVE_BASE="/content/drive/MyDrive"
WEIGHTS_DIR="${DRIVE_BASE}/avatar_weights"
DATA_DIR="${DRIVE_BASE}/avatar_data"
CKPT_DIR="${DRIVE_BASE}/avatar_checkpoints"

echo ""
echo "📁 Setting up Google Drive paths..."

# Create Drive directories if they don't exist
mkdir -p "${WEIGHTS_DIR}/deca"
mkdir -p "${WEIGHTS_DIR}/flame2020"
mkdir -p "${WEIGHTS_DIR}/ffhq"
mkdir -p "${WEIGHTS_DIR}/facescape"
mkdir -p "${DATA_DIR}/selfies"
mkdir -p "${DATA_DIR}/albedo"
mkdir -p "${DATA_DIR}/normal"
mkdir -p "${DATA_DIR}/vietnamese/selfies"
mkdir -p "${CKPT_DIR}/geometry"
mkdir -p "${CKPT_DIR}/texture"
mkdir -p "${CKPT_DIR}/texture_lora"

# Symlink weights directory
if [ -d "${WEIGHTS_DIR}" ]; then
    rm -rf weights 2>/dev/null || true
    ln -sf "${WEIGHTS_DIR}" weights
    echo "  ✅ weights → ${WEIGHTS_DIR}"
else
    echo "  ⚠️ ${WEIGHTS_DIR} not found — create it on Google Drive"
fi

# Symlink data directory
if [ -d "${DATA_DIR}" ]; then
    rm -rf data 2>/dev/null || true
    ln -sf "${DATA_DIR}" data
    echo "  ✅ data → ${DATA_DIR}"
else
    echo "  ⚠️ ${DATA_DIR} not found — create it on Google Drive"
fi

# --- 6. Verify setup ---
echo ""
echo "============================================"
echo "🔍 Verification"
echo "============================================"

python -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
    print(f'  VRAM: {mem:.1f} GB')

try:
    import torch_geometric
    print(f'  PyG: {torch_geometric.__version__}')
except ImportError:
    print('  PyG: ❌ Not installed (GNN fallback mode)')

try:
    import pytorch3d
    print(f'  PyTorch3D: ✅')
except ImportError:
    print('  PyTorch3D: ❌ Not installed')

import pathlib
w = pathlib.Path('weights')
d = pathlib.Path('data')
print(f'  Weights dir: {\"✅\" if w.exists() else \"❌\"} ({w.resolve()})')
print(f'  Data dir: {\"✅\" if d.exists() else \"❌\"} ({d.resolve()})')

# Check for key weight files
weight_files = [
    'weights/deca/deca_model.tar',
    'weights/flame2020/head_template.obj',
    'weights/flame2020/FLAME_texture.npz',
]
print()
print('  Weight files status:')
for wf in weight_files:
    status = '✅' if pathlib.Path(wf).exists() else '❌ MISSING'
    print(f'    {wf}: {status}')
"

echo ""
echo "============================================"
echo "✅ Setup complete!"
echo "============================================"
