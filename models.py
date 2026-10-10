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
    # 归属用户（隔离标签）：入库时由服务端从 token 查出写入，前端无法指定。
    # 不加 ForeignKey：老库靠 ALTER TABLE 补列，SQLite 无法给旧表补外键
    user_id = Column(Integer, nullable=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    # 缺少 updated_at 列会导致 update_document 中 db_doc.updated_at 赋值后不持久化
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


class ChatRecord(Base):
    """聊天记录（短期记忆）：只保留最近若干轮完整对话，超出的旧对话迁移到向量库长期记忆。"""
    __tablename__ = "chat_records"

    id = Column(Integer, primary_key=True, index=True)
    role = Column(String(20), nullable=False)  # user / assistant
    content = Column(Text, nullable=False)
    # 归属用户（隔离标签）：聊天记忆按人隔离，查记忆时只对上自己的暗号
    user_id = Column(Integer, nullable=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class User(Base):
    """注册用户（花名册）：身份层的核心表。密码只存哈希，绝不存明文。"""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), nullable=False, unique=True, index=True)
    # PBKDF2 哈希（格式：盐$哈希值）：数据库泄露也无法还原原密码
    password_hash = Column(String(200), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
