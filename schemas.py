from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class DocumentCreate(BaseModel):
    title: str
    content: str

class DocumentResponse(BaseModel):
    id: int
    title: str
    content: str
    created_at: datetime
    # 旧数据库中没有 updated_at，需设为可选避免序列化报错
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True
