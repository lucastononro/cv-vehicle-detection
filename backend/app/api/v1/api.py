from fastapi import APIRouter
from .endpoints import health, videos, images, labelling

api_router = APIRouter()
api_router.include_router(health.router, prefix="/health", tags=["health"])
api_router.include_router(videos.router, prefix="/videos", tags=["videos"]) 
api_router.include_router(images.router, prefix="/images", tags=["images"])
api_router.include_router(labelling.router, prefix="/labelling", tags=["labelling"])