from datetime import datetime, timezone
from typing import Any, Dict

def to_camel(string: str) -> str:
    """Convert snake_case to camelCase"""
    components = string.split('_')
    return components[0] + ''.join(x.title() for x in components[1:])

def to_snake(string: str) -> str:
    """Convert camelCase to snake_case"""
    return ''.join(['_' + c.lower() if c.isupper() else c for c in string]).lstrip('_')

def utc_now() -> datetime:
    """Get current UTC datetime"""
    return datetime.now(timezone.utc)

def remove_none_values(d: Dict[str, Any]) -> Dict[str, Any]:
    """Remove None values from a dictionary"""
    return {k: v for k, v in d.items() if v is not None} 