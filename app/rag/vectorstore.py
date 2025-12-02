import chromadb
from langchain_chroma import Chroma # langchain和chromadb结合需要使用的依赖
from app.config import settings

# 获得向量存储的一个方法
def get_vectorstore(embeddings):

    # 获得一个指向chromadb的连接
    client = chromadb.HttpClient(
        host=settings.chroma_host,  # IP
        port=settings.chroma_port,  # 端口号

    )
    # chromadb.api.client._DEFAULT_TIMEOUT = 30  # 单位：秒，按需调整

    return Chroma(
        client=client,
        collection_name=settings.collection_name,   # 数据库名
        embedding_function=embeddings,              # 嵌入方法
    )


