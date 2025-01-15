from typing import TypeVar, Generic, Optional, List
from ..repositories import BaseRepository

T = TypeVar('T')

class BaseService(Generic[T]):
    """
    Base Service class that provides common CRUD operations
    """
    
    def __init__(self, repository: BaseRepository[T]):
        self.repository = repository
    
    async def get(self, id: str) -> Optional[T]:
        return await self.repository.get(id)
    
    async def get_all(self) -> List[T]:
        return await self.repository.get_all()
    
    async def create(self, data: T) -> T:
        return await self.repository.create(data)
    
    async def update(self, id: str, data: T) -> Optional[T]:
        return await self.repository.update(id, data)
    
    async def delete(self, id: str) -> bool:
        return await self.repository.delete(id) 