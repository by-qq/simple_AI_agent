from __future__ import annotations

from functools import lru_cache

import torch # PyTorch框架主要负责张量计算、模型推理等
from transformers import AutoModelForSequenceClassification, AutoTokenizer
# AutoTokenizer根据模型名自动加载对应的tokenizer分词器/编码器
# AutoModelForSequenceClassification自动加载序列分类模型，比如reranker就是这种：输入query和doc，输出相关性分数或分类logits
from app.config import settings

@lru_cache(maxsize=1) # 这个函数加载的tokenizer避免每次rerank都重新加载大模型非常慢
def _load_reranker():
    model_name = settings.audio_rerank_model
    tok = AutoTokenizer.from_pretrained(model_name) # 从配置里读reranker模型名/路径，赋给model_name
    # 用模型名加载tokenizer模型，默认走HuggingFace缓存目录，没缓存则下载，所以略长
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()  # 用模型名加载序列分类模型，权重同理会下载
    return tok, model  # 返回tokenizer和模型

def rerank_scores(query: str, texts: list[str]) -> list[float]:
    """
    当前这个函数会把query和每条候选text组成一对，喂给一个Transformers的序列分类模型reranker
    输出每条候选的相关性分数，这个分数是float列表，顺序与texts一一对应
    """
    if not texts:
        return []

    tok, model = _load_reranker()

    bs = int(getattr(settings, "audio_rerank_batch_size", 16))  # batch size是批大小，即模型一次推理处理多少条候选文本
    # 比如text有40条，bs=16，切分批次就是：16+16条+8条
    # bs越大越快但越吃内存
    out: list[float] = []  # out是一个收集最终分数的列表
    # out的长度最终应该等于len(texts)，out[j]对应texts[j]的rerank分数

    with torch.no_grad():  # 关闭梯度计算，这里推理不需要提督，更省内存也更快
        for i in range(0, len(texts), bs):
            chunk = texts[i : i + bs]
            enc = tok(  # 这一段是tokenizer编码
                [query] * len(chunk),  # 把同一个query复制成batch大小，让它与每个候选文本一一配对
                chunk,  # 候选文本列表
                padding=True,  # 把batch内序列padding成同长度，便于组成张量，要求张量大小一致，想像成规则二位数组
                truncation=True,  # 超长就截断
                max_length=int(getattr(settings, "audio_rerank_max_len", 512)),  # 截断的最大token长度默认512
                return_tensors="pt",  # 返回PyTorch张量字典，通常包含input_ids和attention_mask等等
            )
            logits = model(**enc).logits  # model(**enc)把tokenizer输出的张量字典当参数喂给模型。.logits是模型原始输出分数张量

            if logits.dim() == 2 and logits.size(-1) >= 1:  # 如果是二维[B, C]且C>=1
                s = logits[:, 0]  # 取第0列logits[:, 0]当作分数向量s
            else:
                s = logits.view(-1)  # 否则直接view(-1)拉平成一维

            out.extend([float(x) for x in s.cpu().tolist()])  # s.cpu()确保张量在CPU上，然后把tensor转成float加入out
    return out

# todo 单独测试，打断点，看每次的输出结果格式
if __name__ ==  "__main__":
    text = [
        """音乐陶冶身心：
荔枝春
河图""",
        """游玩增长见识：身未动，心已远。
一个人的旅行，在路上遇见最真实的自己。
梦想，并不奢侈，只要勇敢地迈出第一步。
旅行要学会随遇而安，淡然一点，走走停停。
每一次旅行，都是从内心到风景的巡礼。""",
        "说点啥阿，不知道"
    ]
    reranker = rerank_scores("游玩相关的句子", text)
    print(reranker)




