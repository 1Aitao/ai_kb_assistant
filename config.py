"""配置中心：集中管理所有环境变量与默认值，禁止在业务代码中直接读环境变量。"""
from functools import lru_cache
import os

from dotenv import load_dotenv

load_dotenv()


class Settings:
    """应用配置项。敏感信息来自 .env，其余提供合理默认值。"""

    def __init__(self) -> None:
        # 大模型（DeepSeek）
        self.DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
        self.DEEPSEEK_BASE_URL: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        self.DEEPSEEK_MODEL: str = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        self.LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))
        # 和风天气
        self.QWEATHER_KEY: str = os.getenv("QWEATHER_KEY", "")
        self.QWEATHER_BASE_URL: str = os.getenv("QWEATHER_BASE_URL", "https://devapi.qweather.com")
        self.QWEATHER_GEO_BASE_URL: str = os.getenv("QWEATHER_GEO_BASE_URL", self.QWEATHER_BASE_URL)
        # 向量库与嵌入模型
        self.CHROMA_DIR: str = os.getenv("CHROMA_DIR", "./chroma_db")
        self.CHROMA_COLLECTION: str = os.getenv("CHROMA_COLLECTION", "documents")
        self.EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
        # 分块策略
        self.CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
        self.CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))
        self.SEARCH_RESULTS: int = int(os.getenv("SEARCH_RESULTS", "3"))
        # OCR 渲染倍率：实测 1.2 比 2.0 快约 28% 且识别质量相当；低质量扫描件可调回 2.0
        self.OCR_RENDER_SCALE: float = float(os.getenv("OCR_RENDER_SCALE", "1.2"))
        # 上传目录
        self.UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "uploads")


@lru_cache
def get_settings() -> Settings:
    """返回全局单例配置，避免重复读取环境变量。"""
    return Settings()


settings = get_settings()
