from sqlalchemy import Column, Integer, String, Text, DateTime
from sqlalchemy.sql import func
from database import Base

class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(200), nullable=False)
    content = Column(Text, nullable=False)
    # 物理文件路径（UUID 命名，与 title 分离：title 存原始展示名，file_path 存磁盘上的实际路径）
    file_path = Column(String(500), nullable=True)
    # 文件内容指纹（SHA256）：内容相同指纹就相同，用于上传时识别重复文件
    file_hash = Column(String(64), nullable=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    # 缺少 updated_at 列会导致 update_document 中 db_doc.updated_at 赋值后不持久化
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
