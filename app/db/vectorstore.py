import chromadb
from langchain_chroma import Chroma
from app.config import settings

def get_client():
    return chromadb.HttpClient(
        host=settings.chroma_host,
        port=settings.chroma_port
    )

def get_vectorstore(embeddings):
    return Chroma(
        client=get_client(),
        collection_name=settings.collection_name,
        embedding_function=embeddings,
    )

def get_audio_vectorstore(embeddings):
    return Chroma(
        client=get_client(),
        collection_name=settings.audio_collection_name,
        embedding_function=embeddings,
    )
