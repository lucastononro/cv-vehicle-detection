import boto3
import os
from typing import BinaryIO, List, Dict, Any, Optional
import io

class S3Repository:
    def __init__(self):
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            endpoint_url=os.getenv('S3_ENDPOINT_URL')
        )
        self.bucket_name = "videos"
    
    def upload_file(self, file: BinaryIO, filename: str) -> None:
        """Upload a file to S3"""
        self.s3_client.upload_fileobj(file, self.bucket_name, filename)
    
    def upload_bytes(self, data: bytes, key: str) -> None:
        """Upload bytes data to S3"""
        self.s3_client.upload_fileobj(io.BytesIO(data), self.bucket_name, key)
    
    def get_file_bytes(self, key: str) -> Optional[bytes]:
        """Get file as bytes from S3"""
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
            return response['Body'].read()
        except:
            return None
    
    def download_file(self, filename: str, local_path: str) -> None:
        """Download a file from S3"""
        with open(local_path, 'wb') as f:
            self.s3_client.download_fileobj(self.bucket_name, filename, f)
    
    def list_files(self) -> List[Dict[str, Any]]:
        """List all files in the bucket"""
        response = self.s3_client.list_objects_v2(Bucket=self.bucket_name)
        if 'Contents' in response:
            return response['Contents']
        return [] 