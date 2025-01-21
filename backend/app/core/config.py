import os
import json
from pydantic_settings import BaseSettings
from typing import Optional, List

class Settings(BaseSettings):
    PROJECT_NAME: str = "Vehicle Detection API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # AWS Configuration
    AWS_ACCESS_KEY_ID: str = "test"
    AWS_SECRET_ACCESS_KEY: str = "test"
    AWS_DEFAULT_REGION: str = "us-east-1"
    S3_ENDPOINT_URL: str = "http://localstack:4566"
    
    # Database Configuration
    DATABASE_URL: str
    
    # CORS Settings
    CORS_ORIGINS: List[str] = json.loads(
        os.getenv("CORS_ORIGINS", '["http://localhost:4200"]')
    )
    
    class Config:
        case_sensitive = True

settings = Settings() 