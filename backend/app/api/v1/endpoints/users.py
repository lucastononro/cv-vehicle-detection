from fastapi import APIRouter, Depends
from ....services import BaseService
from typing import List

router = APIRouter()

# Example of how to use with dependency injection
def get_user_service() -> BaseService:
    # You would implement proper DI here
    pass

@router.get("/")
async def get_users(
    service: BaseService = Depends(get_user_service)
) -> List[dict]:
    return await service.get_all()

@router.get("/me")
async def get_current_user():
    return {"user": "current_user"} 