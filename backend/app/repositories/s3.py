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
        # Initialize buckets
        self.buckets = {
            "videos": "videos",
            "images": "images",
            "thumbnails": "thumbnails",
            "labels": "labels"  # New bucket for OCR labels
        }
        self._ensure_buckets_exist()
    
    def _ensure_buckets_exist(self):
        """Ensure all required buckets exist"""
        for bucket_name in self.buckets.values():
            try:
                self.s3_client.head_bucket(Bucket=bucket_name)
            except:
                self.s3_client.create_bucket(Bucket=bucket_name)
    
    def upload_file(self, file: BinaryIO, filename: str, bucket_type: str = "videos") -> None:
        """Upload a file to S3"""
        bucket = self.buckets.get(bucket_type, self.buckets["videos"])
        self.s3_client.upload_fileobj(file, bucket, filename)
    
    def upload_bytes(self, data: bytes, key: str, bucket_type: str = "videos") -> None:
        """Upload bytes data to S3"""
        bucket = self.buckets.get(bucket_type, self.buckets["videos"])
        self.s3_client.upload_fileobj(io.BytesIO(data), bucket, key)
    
    def get_file_bytes(self, key: str, bucket_type: str = "videos") -> Optional[bytes]:
        """Get file as bytes from S3"""
        try:
            bucket = self.buckets.get(bucket_type, self.buckets["videos"])
            response = self.s3_client.get_object(Bucket=bucket, Key=key)
            return response['Body'].read()
        except:
            return None
    
    def download_file(self, filename: str, local_path: str, bucket_type: str = "videos") -> None:
        """Download a file from S3"""
        bucket = self.buckets.get(bucket_type, self.buckets["videos"])
        with open(local_path, 'wb') as f:
            self.s3_client.download_fileobj(bucket, filename, f)
    
    def list_files(self, bucket_type: str = "videos") -> List[Dict[str, Any]]:
        """List all files in the specified bucket"""
        bucket = self.buckets.get(bucket_type, self.buckets["videos"])
        response = self.s3_client.list_objects_v2(Bucket=bucket)
        if 'Contents' in response:
            return response['Contents']
        return [] 