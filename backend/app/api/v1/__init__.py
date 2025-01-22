from fastapi import APIRouter
from .endpoints import health, videos, images

v1_router = APIRouter(prefix="/v1")

# Automatically include all routers
v1_router.include_router(health.router, tags=["health"])
v1_router.include_router(videos.router, prefix="/videos", tags=["videos"]) 
v1_router.include_router(images.router, prefix="/images", tags=["images"])