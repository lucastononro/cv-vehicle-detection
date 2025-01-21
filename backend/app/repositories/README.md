# Repositories

This directory implements the repository pattern for data access in the Vehicle Detection system.

## Structure

```
repositories/
├── __init__.py      # Base repository interface
└── s3.py            # S3 storage implementation
```

## Base Repository

The `BaseRepository` class in `__init__.py` defines the standard interface for all repositories:

```python
from app.repositories import BaseRepository

class BaseRepository(Generic[T]):
    async def get(self, id: str) -> Optional[T]
    async def get_all(self) -> List[T]
    async def create(self, data: T) -> T
    async def update(self, id: str, data: T) -> Optional[T]
    async def delete(self, id: str) -> bool
```

## S3 Repository

The `S3Repository` class provides file storage operations using AWS S3 or compatible services (like MinIO):

### Configuration

```python
# Required environment variables
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
S3_ENDPOINT_URL=your_s3_endpoint  # e.g., http://localhost:9000 for MinIO
```

### Features

- File upload (both streaming and bytes)
- File download
- File listing
- Byte-level operations
- Error handling with graceful fallbacks

### Usage Examples

```python
from app.repositories.s3 import S3Repository

# Initialize repository
s3_repo = S3Repository()

# Upload a file
with open('video.mp4', 'rb') as file:
    s3_repo.upload_file(file, 'video.mp4')

# Upload bytes directly
s3_repo.upload_bytes(frame_bytes, 'frame_001.jpg')

# Download a file
s3_repo.download_file('video.mp4', '/tmp/video.mp4')

# Get file as bytes
frame_data = s3_repo.get_file_bytes('frame_001.jpg')

# List all files
files = s3_repo.list_files()
```

### Bucket Structure

The repository uses the following bucket structure:
```
s3://videos/                  # Main bucket
├── raw/                     # Original uploaded videos
│   └── video_001.mp4
├── processed/               # Processed videos
│   └── video_001_detected.mp4
├── frames/                  # Extracted video frames
│   └── video_001/
│       ├── frame_001.jpg
│       └── frame_002.jpg
└── results/                # Detection results
    └── video_001.json
```

## Adding New Storage Providers

To implement a new storage provider:

1. Create a new file (e.g., `mongo_repository.py`)
2. Implement the `BaseRepository` interface
3. Add required configuration to `core/config.py`
4. Example:

```python
from app.repositories import BaseRepository

class MongoRepository(BaseRepository[T]):
    def __init__(self):
        # Initialize connection
        pass

    async def get(self, id: str) -> Optional[T]:
        # Implement get operation
        pass

    # Implement other required methods
``` 