from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from app.config import settings
from app.rag.vectorstore import get_vectorstore
from langchain_community.chat_models import ChatOpenAI

def get_llm():
    """获取大语言模型 - 返回ChatOpenAI实例"""
    return ChatOpenAI(
        model=settings.model_name,
        api_key=settings.openai_api_key,
        base_url=settings.openai_api_base,  # 关键：设置DeepSeek的API地址
        temperature=0.2,
        streaming=True,
    )

# def get_embeddings():
#     """获取嵌入模型 - 使用OpenAI兼容接口"""
#     return OpenAIEmbeddings(
#         api_key=settings.openai_api_key,
#         base_url=settings.openai_api_base,  # 使用DeepSeek的嵌入模型
#         model=settings.embedding_model,
#     )

def get_embeddings():
    """获取嵌入模型 - 使用OpenAI兼容接口"""

    return ZhipuAIEmbeddings(
        model=settings.embedding_model,  # 智谱AI的embedding模型名称
        api_key=settings.zhipu_api_key,

        # api_key="bffeafd8bc604868a43277dcfb30bb24.JS5GyTxie2tkQlR1",  # 替换为您的智谱API密钥
        # base_url=settings.zhipu_api_base # 智谱AI的API地址
    )


def get_vs():
    """获取向量存储实例"""

    return get_vectorstore(get_embeddings())

# 测试
if __name__=="__main__":
    print(get_llm())
#     # print("________________________________")
#     # print(get_vs())