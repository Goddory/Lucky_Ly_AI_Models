from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(default="ok")
    device: str


class AvatarGenerationResponse(BaseModel):
    job_id: str
    glb_url: str
    textures: Dict[str, str]
    shape_params: List[float]
    expression_params: List[float]
    camera_pose: List[float]
    message: Optional[str] = None
