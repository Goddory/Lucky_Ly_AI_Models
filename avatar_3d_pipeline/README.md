# Avatar 3D Pipeline

Image-to-3D face avatar pipeline with:
- Phase 1: DECA/EMOCA base reconstruction
- Phase 2: Part-based VAE-GNN mesh refinement
- Phase 3: Geometry-aware texture generation and PBR extraction
- Phase 4: FastAPI service + GLB export

## Project Structure

```
avatar_3d_pipeline/
├── api/
├── core/
├── models/
├── training/
├── utils/
├── weights/
├── outputs/
├── requirements.txt
└── environment.yml
```

## Environment Setup (RTX 4060, CUDA 12.1)

```bash
conda create -n avatar_env python=3.10 -y
conda activate avatar_env

# Install PyTorch with CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install pipeline dependencies
pip install -r requirements.txt
```

## Download Pre-trained Weights

```bash
python utils/download_weights.py --weights-dir weights
```

If required assets fail because of licensing or mirror availability, set the corresponding environment variables listed in `utils/download_weights.py` and run again.

## Run API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

- Health check: `GET /health`
- Avatar generation: `POST /generate-avatar` with multipart form-data field `image`
- Generated assets are served from `/outputs/...`

## Fine-tune Texture Modules with LoRA

```bash
python training/fine_tune_texture.py \
  --data-root ./data \
  --manifest ./data/manifest.csv \
  --weights-dir ./weights \
  --output-dir ./checkpoints/texture_lora \
  --batch-size 1 \
  --grad-accum-steps 8 \
  --lora-rank 8 \
  --lora-alpha 16
```

Dataset formats are documented in `utils/dataset.py` and support either CSV/JSON manifest or folder scanning.
