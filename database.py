from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# 数据库连接地址
SQLALCHEMY_DATABASE_URL = "sqlite:///./kb_app.db"

# 创建数据库引擎
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)

# 创建会话工厂
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 基类，所有模型都要继承它
Base = declarative_base()

# 获取数据库会话的函数
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

