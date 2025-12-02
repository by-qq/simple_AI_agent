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

    # 文本处理配置
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "800"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "120"))

    # 嵌入模型配置
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "embedding-3")
    # qianwen_api_key: str = os.getenv("DASHSCOPE_API_KEY", "")
    zhipu_api_key: str = os.getenv("ZHIPU_API_KEY", "")
    # zhipu_api_base: str = os.getenv("ZHIPU_API_BASE", "https://open.bigmodel.cn/api/paas/v4" )



settings = Settings()