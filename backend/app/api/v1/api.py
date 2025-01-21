from fastapi import APIRouter
from .endpoints import health, videos

api_router = APIRouter()
api_router.include_router(health.router, prefix="/health", tags=["health"])
api_router.include_router(videos.router, prefix="/videos", tags=["videos"]) 