# 📓 Hướng Dẫn Chạy Trên Google Colab

## Bước 0: Chuẩn bị Google Drive

Tạo cấu trúc thư mục trên Google Drive:

```
📁 My Drive/
├── 📁 avatar_weights/        ← Bỏ file checkpoint vào đây
│   ├── 📁 deca/
│   │   └── deca_model.tar
│   ├── 📁 flame2020/
│   │   ├── head_template.obj
│   │   ├── FLAME_texture.npz
│   │   └── generic_model.pkl
│   ├── 📁 ffhq/              ← Sau khi train stage 1
│   └── 📁 facescape/         ← Sau khi train stage 2-4
├── 📁 avatar_data/           ← Training data
│   ├── 📁 ffhq_raw/         ← Ảnh FFHQ gốc
│   ├── 📁 selfies/          ← Ảnh selfie đã xử lý
│   ├── 📁 albedo/           ← Albedo maps
│   ├── 📁 normal/           ← Normal maps
│   ├── 📁 vietnamese/
│   │   └── 📁 selfies/     ← 100-200 ảnh selfie VN
│   └── manifest.csv
└── 📁 avatar_checkpoints/    ← Training output
    ├── 📁 geometry/
    ├── 📁 texture/
    └── 📁 texture_lora/
```

---

## Notebook 1: Setup & Verify Environment

Tạo Colab notebook mới, paste từng cell:

### Cell 1: Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

### Cell 2: Clone repo & Install deps
```python
!git clone https://github.com/<YOUR_REPO>/Lucky_Ly_AI_Models.git /content/repo
%cd /content/repo/avatar_3d_pipeline
!bash scripts/setup_colab.sh
```

### Cell 3: Verify weights
```python
from pathlib import Path

weight_files = [
    "weights/deca/deca_model.tar",
    "weights/flame2020/head_template.obj",
    "weights/flame2020/FLAME_texture.npz",
]

for wf in weight_files:
    p = Path(wf)
    status = "✅" if p.exists() else "❌ MISSING"
    size = f"({p.stat().st_size / 1024 / 1024:.1f} MB)" if p.exists() else ""
    print(f"  {wf}: {status} {size}")
```

---

## Notebook 2: Sinh Dữ Liệu Training (Generate Pairs)

### Cell 1: Setup (như Notebook 1)

### Cell 2: Download FFHQ subset (nếu chưa có)
```python
# Option A: Dùng FFHQ thumbnails từ HuggingFace
!pip install -q datasets
from datasets import load_dataset
import os

output_dir = "/content/drive/MyDrive/avatar_data/ffhq_raw"
os.makedirs(output_dir, exist_ok=True)

ds = load_dataset("mattymchen/ffhq-1024", split="train[:5000]")
for i, sample in enumerate(ds):
    sample["image"].save(f"{output_dir}/{i:05d}.png")
    if (i + 1) % 500 == 0:
        print(f"  Downloaded {i+1} images...")

print(f"✅ Downloaded {len(ds)} FFHQ images")
```

### Cell 3: Generate synthetic pairs
```python
!python training/generate_synthetic_pairs.py \
    --input-dir /content/drive/MyDrive/avatar_data/ffhq_raw \
    --output-dir /content/drive/MyDrive/avatar_data \
    --weights-dir weights \
    --image-size 512 \
    --max-samples 5000
```

### Cell 4: Validate dataset
```python
!python training/validate_dataset.py \
    --data-root /content/drive/MyDrive/avatar_data \
    --min-size 256 --fix
```

---

## Notebook 3: Train Geometry VAE

### Cell 1: Setup (như Notebook 1)

### Cell 2: Chuẩn bị mesh data
```python
# NOTE: Geometry training cần mesh data (vertices + faces)
# Nếu chưa có, DECA reconstruction sẽ tạo mesh từ ảnh

import numpy as np
from pathlib import Path

mesh_dir = Path("/content/drive/MyDrive/avatar_data/meshes")
mesh_dir.mkdir(parents=True, exist_ok=True)

# Kiểm tra nếu đã có mesh data
existing = list(mesh_dir.glob("*.npy"))
print(f"📊 Found {len(existing)} mesh files")

if len(existing) == 0:
    print("⚠️ Không tìm thấy mesh data.")
    print("   Cần chạy DECA reconstruction trên ảnh để tạo mesh trước.")
    print("   Hoặc dùng FaceScape 3D scans.")
```

### Cell 3: Train
```python
!python training/train_geometry_vae.py \
    --data-root /content/drive/MyDrive/avatar_data/meshes \
    --output-dir /content/drive/MyDrive/avatar_checkpoints/geometry \
    --epochs 100 \
    --lr 1e-4 \
    --save-every 1 \
    --w-recon 1.0 \
    --w-kl 0.001 \
    --w-laplacian 0.1 \
    --w-edge 0.05
```

### Cell 4: Kiểm tra training progress
```python
import json
from pathlib import Path

metrics_path = Path("/content/drive/MyDrive/avatar_checkpoints/geometry/metrics_latest.json")
if metrics_path.exists():
    metrics = json.loads(metrics_path.read_text())
    print(f"📊 Latest checkpoint:")
    print(f"   Epoch: {metrics.get('epoch', '?')}")
    for k, v in metrics.items():
        if k != 'epoch':
            print(f"   {k}: {v:.6f}")
```

---

## Notebook 4: Train Texture Networks

### Cell 1: Setup

### Cell 2: Stage 1 — Train G (texture prior)
```python
!python training/train_texture_full.py \
    --data-root /content/drive/MyDrive/avatar_data \
    --weights-dir /content/drive/MyDrive/avatar_weights \
    --output-dir /content/drive/MyDrive/avatar_checkpoints/texture \
    --stage 1 \
    --epochs 50 \
    --batch-size 2 \
    --image-size 512 \
    --save-every 1
```

### Cell 3: Stage 2 — Train G_A (skin tone)
```python
# Chạy SAU khi stage 1 complete
!python training/train_texture_full.py \
    --data-root /content/drive/MyDrive/avatar_data \
    --weights-dir /content/drive/MyDrive/avatar_checkpoints/texture \
    --output-dir /content/drive/MyDrive/avatar_checkpoints/texture \
    --stage 2 \
    --epochs 30 \
    --batch-size 2 \
    --save-every 1
```

### Cell 4: Stage 3+4 — Train G_C + G_E
```python
# Stage 3: Reflectance
!python training/train_texture_full.py \
    --data-root /content/drive/MyDrive/avatar_data \
    --weights-dir /content/drive/MyDrive/avatar_checkpoints/texture \
    --output-dir /content/drive/MyDrive/avatar_checkpoints/texture \
    --stage 3 --epochs 30 --batch-size 2 --save-every 1

# Stage 4: PBR extraction
!python training/train_texture_full.py \
    --data-root /content/drive/MyDrive/avatar_data \
    --weights-dir /content/drive/MyDrive/avatar_checkpoints/texture \
    --output-dir /content/drive/MyDrive/avatar_checkpoints/texture \
    --stage 4 --epochs 20 --batch-size 2 --save-every 1
```

---

## Notebook 5: LoRA Fine-Tune cho Vietnamese Faces

### Cell 1: Setup

### Cell 2: Sinh pairs cho ảnh VN
```python
!python training/generate_synthetic_pairs.py \
    --input-dir /content/drive/MyDrive/avatar_data/vietnamese/selfies \
    --output-dir /content/drive/MyDrive/avatar_data/vietnamese \
    --weights-dir weights \
    --image-size 512
```

### Cell 3: Fine-tune
```python
!python training/fine_tune_texture.py \
    --data-root /content/drive/MyDrive/avatar_data/vietnamese \
    --weights-dir /content/drive/MyDrive/avatar_checkpoints/texture \
    --output-dir /content/drive/MyDrive/avatar_checkpoints/texture_lora \
    --batch-size 1 \
    --grad-accum-steps 8 \
    --lora-rank 8 \
    --lora-alpha 16 \
    --epochs 25
```

---

## Notebook 6: Evaluate

### Cell 1: Setup

### Cell 2: Run evaluation
```python
!python training/evaluate.py \
    --pred-dir /content/drive/MyDrive/avatar_data/albedo \
    --gt-dir /content/drive/MyDrive/avatar_data/albedo \
    --metrics all \
    --image-size 512 \
    --report-path /content/drive/MyDrive/avatar_checkpoints/eval_report.json
```

---

## ⚠️ Xử Lý Khi Session Bị Ngắt

1. **Đừng hoảng** — tất cả checkpoint đã lưu trên Google Drive
2. **Mở lại notebook** → chạy Cell 1 (Mount Drive) + Cell 2 (Clone + Install)
3. **Chạy lại training** — script sẽ **tự động tìm và resume** từ checkpoint cuối cùng
4. **Không tốn lại epoch** — training tiếp tục từ chỗ bị ngắt

## 💡 Tips Tối Ưu Colab

1. **Giữ tab mở** — Colab ngắt nhanh hơn nếu bạn chuyển tab
2. **Dùng Colab Pro** cho texture training (stage 2-4) vì cần nhiều VRAM hơn
3. **Kiểm tra GPU**: Runtime → Change runtime type → GPU (T4 hoặc tốt hơn)
4. **Dùng `--save-every 1`** luôn — Colab có thể ngắt bất cứ lúc nào
