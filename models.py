from sqlalchemy import Column, Integer, String, Text, DateTime
from sqlalchemy.sql import func
from database import Base

class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(200), nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    # 缺少 updated_at 列会导致 update_document 中 db_doc.updated_at 赋值后不持久化
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
