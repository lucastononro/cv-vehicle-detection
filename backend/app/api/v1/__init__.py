from fastapi import APIRouter
from .endpoints import health, users, videos

v1_router = APIRouter(prefix="/v1")

# Automatically include all routers
v1_router.include_router(health.router, tags=["health"])
v1_router.include_router(users.router, prefix="/users", tags=["users"])
v1_router.include_router(videos.router, prefix="/videos", tags=["videos"]) 