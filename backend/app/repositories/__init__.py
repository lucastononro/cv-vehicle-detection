from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Optional, List

T = TypeVar('T')

class BaseRepository(ABC, Generic[T]):
    """
    Abstract Base Repository defining common operations
    """
    
    @abstractmethod
    async def get(self, id: str) -> Optional[T]:
        pass
    
    @abstractmethod
    async def get_all(self) -> List[T]:
        pass
    
    @abstractmethod
    async def create(self, data: T) -> T:
        pass
    
    @abstractmethod
    async def update(self, id: str, data: T) -> Optional[T]:
        pass
    
    @abstractmethod
    async def delete(self, id: str) -> bool:
        pass 