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
# Install ninja FIRST to enable parallel C++ compilation (turns 30 min build into 3 mins)
pip install -q ninja
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

# --- 4b. Install DECA (decalib) ---
echo ""
echo "📦 Setting up DECA (decalib)..."
if [ ! -d "/content/DECA" ]; then
    git clone https://github.com/yfeng95/DECA.git /content/DECA 2>/dev/null && \
        echo "  ✅ DECA repo cloned" || echo "  ⚠️  Clone failed"
fi
# Install DECA dependencies explicitly (requirements.txt often incomplete)
pip install -q kornia face-alignment yacs chumpy 2>/dev/null || true
pip install -q -r /content/DECA/requirements.txt 2>/dev/null || true
# --- 5. Fix DECA Environment for Python 3.12 ---
# DECA has no setup.py — add to Python path via .pth file
PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "/content/DECA" > "/usr/local/lib/python${PYTHON_VERSION}/dist-packages/deca_path.pth"

# 5a. Patch chumpy package directly (it uses removed features from Python 3.12 & Numpy 2.x)
echo "🔧 Patching chumpy for compatibility..."
python -c "
import site, os
for sp in site.getsitepackages():
    ch_init = os.path.join(sp, 'chumpy', '__init__.py')
    if os.path.exists(ch_init):
        with open(ch_init, 'r') as f:
            c = f.read()
        if 'from numpy import bool' in c:
            c = c.replace(
                'from numpy import bool, int, float, complex, object, unicode, str, nan, inf',
                'from numpy import nan, inf'
            )
            with open(ch_init, 'w') as f:
                f.write(c)
    ch_py = os.path.join(sp, 'chumpy', 'ch.py')
    if os.path.exists(ch_py):
        with open(ch_py, 'r') as f:
            c2 = f.read()
        if 'getargspec' in c2 and 'getfullargspec' not in c2:
            c2 = c2.replace('inspect.getargspec', 'inspect.getfullargspec')
            with open(ch_py, 'w') as f:
                f.write(c2)
"

# 5b. Copy missing FLAME topology files into DECA/data 
echo "📂 Setting up FLAME topology files..."
python -c "
import shutil, os
flame_src = '/content/drive/MyDrive/avatar_project/avatar_weights/flame2020'
deca_data = '/content/DECA/data'
os.makedirs(deca_data, exist_ok=True)
if os.path.isdir(flame_src):
    for f in os.listdir(flame_src):
        dst = os.path.join(deca_data, f)
        if not os.path.exists(dst):
            shutil.copy2(os.path.join(flame_src, f), dst)
"
# --- 5. Setup Google Drive paths ---
# NOTE: Google Drive (FUSE) does not support symlinks.
# We create symlinks on LOCAL disk (/content/) pointing TO Drive instead.
DRIVE_BASE="/content/drive/MyDrive/avatar_project"
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

# Create symlinks on LOCAL disk /content/ (supports symlinks) → pointing to Drive
if [ -d "${WEIGHTS_DIR}" ]; then
    rm -rf /content/weights 2>/dev/null || true
    ln -sf "${WEIGHTS_DIR}" /content/weights
    echo "  ✅ /content/weights → ${WEIGHTS_DIR}"
else
    echo "  ⚠️ ${WEIGHTS_DIR} not found — create it on Google Drive first"
fi

if [ -d "${DATA_DIR}" ]; then
    rm -rf /content/data 2>/dev/null || true
    ln -sf "${DATA_DIR}" /content/data
    echo "  ✅ /content/data → ${DATA_DIR}"
else
    echo "  ⚠️ ${DATA_DIR} not found — create it on Google Drive first"
fi

# Also expose weights & data in current pipeline directory via env vars
export WEIGHTS_DIR="${WEIGHTS_DIR}"
export DATA_DIR="${DATA_DIR}"
export CKPT_DIR="${CKPT_DIR}"
echo "  ✅ Env vars: WEIGHTS_DIR, DATA_DIR, CKPT_DIR exported"

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
    mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
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
