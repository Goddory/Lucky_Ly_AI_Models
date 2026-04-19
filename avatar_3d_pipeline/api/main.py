from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.schemas import AvatarGenerationResponse, HealthResponse
from core.exporter import AvatarExporter
from core.geometry_gnn import GeometryRefiner
from core.reconstructor import DECAReconstructor
from core.texture_gen import TextureSynthesisPipeline

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

ROOT_DIR = Path(__file__).resolve().parents[1]
WEIGHTS_DIR = ROOT_DIR / "weights"
OUTPUT_DIR = ROOT_DIR / "outputs"
TMP_DIR = OUTPUT_DIR / "tmp"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEXTURE_OUTPUT_SIZE = int(os.getenv("AVATAR_TEXTURE_SIZE", "512"))
ALLOW_MISSING_WEIGHTS = os.getenv("AVATAR_ALLOW_MISSING_WEIGHTS", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

if ALLOW_MISSING_WEIGHTS:
    LOGGER.warning(
        "AVATAR_ALLOW_MISSING_WEIGHTS is enabled. Running in non-strict test mode."
    )

reconstructor = DECAReconstructor(
    weights_dir=WEIGHTS_DIR,
    device=DEVICE,
    use_emoca=False,
    fallback_on_error=ALLOW_MISSING_WEIGHTS,
)
geometry_refiner = GeometryRefiner(
    weights_dir=WEIGHTS_DIR,
    device=DEVICE,
    strict_checkpoint=not ALLOW_MISSING_WEIGHTS,
)
texture_pipeline = TextureSynthesisPipeline(
    weights_dir=WEIGHTS_DIR,
    device=DEVICE,
    strict_pretrained=not ALLOW_MISSING_WEIGHTS,
)
exporter = AvatarExporter(output_dir=OUTPUT_DIR)

app = FastAPI(
    title="Image-to-3D Face Avatar API",
    version="1.0.0",
    description="Single-image 3D avatar generation pipeline with mesh refinement and PBR textures.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", device=DEVICE)


@app.post("/generate-avatar", response_model=AvatarGenerationResponse)
async def generate_avatar(
    image: UploadFile = File(...),
    alpha: float = Form(0.55),
    deterministic: bool = Form(True),
    smooth_iterations: int = Form(2),
) -> AvatarGenerationResponse:
    if image.content_type is None or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    job_id = f"avatar_{uuid.uuid4().hex[:10]}"
    file_suffix = Path(image.filename or "selfie.jpg").suffix or ".jpg"
    temp_path = TMP_DIR / f"{job_id}{file_suffix}"

    try:
        payload = await image.read()
        temp_path.write_bytes(payload)

        recon = reconstructor.reconstruct(temp_path)
        refined = geometry_refiner.refine(
            vertices=recon.vertices,
            faces=recon.faces,
            deterministic=deterministic,
            smooth_iterations=max(0, int(smooth_iterations)),
        )
        try:
            tex = texture_pipeline.generate(
                uv_texture=recon.uv_texture,
                z_g=refined.z_g,
                alpha=float(alpha),
                output_size=max(128, TEXTURE_OUTPUT_SIZE),
            )
        except torch.OutOfMemoryError:
            # Retry at lower UV resolution for 8GB GPUs when texture attention spikes.
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            tex = texture_pipeline.generate(
                uv_texture=recon.uv_texture,
                z_g=refined.z_g,
                alpha=float(alpha),
                output_size=256,
            )

        export_result = exporter.export_glb(
            vertices=refined.refined_vertices,
            faces=refined.faces,
            uv_coords=recon.uv_coords,
            textures=tex.to_dict(),
            output_name=job_id,
        )

        texture_urls = {
            key: f"/outputs/{job_id}_textures/{path.name}"
            for key, path in export_result.texture_paths.items()
        }

        return AvatarGenerationResponse(
            job_id=job_id,
            glb_url=f"/outputs/{export_result.glb_path.name}",
            textures=texture_urls,
            shape_params=recon.shape_params.astype(float).tolist(),
            expression_params=recon.expression_params.astype(float).tolist(),
            camera_pose=recon.camera_pose.astype(float).tolist(),
            message="Avatar generated successfully.",
        )
    except HTTPException:
        raise
    except Exception as exc:
        LOGGER.exception("Avatar generation failed for %s", job_id)
        raise HTTPException(status_code=500, detail=f"Avatar generation failed: {exc}") from exc
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
