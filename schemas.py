from pydantic import BaseModel
from datetime import datetime
from typing import Optional
from typing import Literal

class DocumentCreate(BaseModel):
    title: str
    content: str

class DocumentResponse(BaseModel):
    id: int
    title: str
    content: str
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True

class ChatRecordResponse(BaseModel):
    """返回给前端的一条聊天记录（短期记忆）。"""
    id: int
    role: Literal["user", "assistant"]
    content: str
    created_at: datetime

    class Config:
        from_attributes = True

class ChatRequest(BaseModel):
    # 历史由后端自行管理，前端只发当前这一句
    message: str


class RegisterRequest(BaseModel):
    """注册新用户：用户名 + 密码（密码到后端立刻搅成哈希，绝不落明文）。"""
    username: str
    password: str