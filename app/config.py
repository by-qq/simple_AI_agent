from pydantic import BaseModel
from dotenv import load_dotenv
import os

load_dotenv()


class Settings(BaseModel):
    # DeepSeek配置（使用OpenAI兼容的字段名）
    openai_api_key: str = os.getenv("DEEPSEEK_API_KEY", "")
    model_name: str = os.getenv("MODEL_NAME", "deepseek-chat")
    openai_api_base: str = os.getenv("OPENAI_API_BASE", "https://api.deepseek.com/v1")

    # # 对于本地Ollama
    # model_name:str = "gpt-oss:20b"  # 您的本地模型名
    # openai_api_base:str = "http://localhost:11434/v1"
    # openai_api_key:str = "ollama"  # 伪密钥

    # ChromaDB配置
    chroma_dir: str = os.getenv("CHROMA_DIR", "./data/chroma")
    chroma_host: str = os.getenv("CHROMA_HOST", "localhost")
    chroma_port: int = int(os.getenv("CHROMA_PORT", "8000"))
    collection_name: str = os.getenv("COLLECTION_NAME", "knowledge_base")
    audio_collection_name: str = os.getenv("AUDIO_COLLECTION_NAME", "audio_base")

    # 文本处理配置
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "800"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "120"))

    # 嵌入模型配置
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "embedding-3")
    # qianwen_api_key: str = os.getenv("DASHSCOPE_API_KEY", "")
    zhipu_api_key: str = os.getenv("ZHIPU_API_KEY", "")
    # zhipu_api_base: str = os.getenv("ZHIPU_API_BASE", "https://open.bigmodel.cn/api/paas/v4" )

    MYSQL_HOST:str = os.getenv("MYSQL_HOST", "localhost")
    MYSQL_PORT:int = int(os.getenv("MYSQL_PORT", "3306"))
    MYSQL_USER:str = os.getenv("MYSQL_USER", "tom")
    MYSQL_PASSWORD:int = os.getenv("MYSQL_PASSWORD", "123456")  # 可配置到系统环境中
    MYSQL_DB:str = os.getenv("MYSQL_DB", "enterprise_kb")
    # MySQL 连接池配置
    MYSQL_POOL_MIN_SIZE:int = 1
    MYSQL_POOL_MAX_SIZE:int = 10

    REDIS_HOST:str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT:int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_TTL_SECONDS:int = 604800  # 7 days 键多久会自动过期

    JWT_SECRET:str = os.getenv("JWT_SECRET", "dev-only-changeme")
    JWT_ALG:str = os.getenv("JWT_ALG", "HS256")
    JWT_EXPIRE_MINUTES:int = int(os.getenv("JWT_EXPIRE_MINUTES", 120))

    CILLECTION_NAME:str = os.getenv("AUDIO_COLLECTION_NAME", "audio_base")

    celery_broker_url: str = os.getenv("CELERY_BROKER_URL", "amqp://peter:123456@127.0.0.1:5672/%2F")  # ⚠️改自己的用户名和密码
    celery_audio_queue: str = "audio"  # 消息队列的名字

    audio_dir: str = "data/audio"
    audio_wav_dir: str = "data/audio_wav"
    audio_clip_dir: str = "data/audio_clips"

settings = Settings()