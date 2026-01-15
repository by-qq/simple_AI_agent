from __future__ import annotations

from dataclasses import dataclass

from app.db.es_audio_admin import keyword_search
from app.deps import get_audio_vs
from app.tools.torch_reranker import rerank_scores


@dataclass
class Candidate:  # 候选片段，一个音频的一个segment，用Candidate
    audio_id: str
    segment_id: str
    start_ms: int
    end_ms: int
    text: str

    vec_rank: int | None = None  # 来自向量召回的排名，1是最好
    es_rank: int | None = None  # 来自es关键词召回的排名
    rrf_score: float = 0.0  # RRF融合后的分数，越大越靠前，es和vec融合后的分数
    rerank_score: float | None = None  # reranker给的最终重排分数


def _rrf_add(score: float, rank: int, k0: int) -> float:
    """RRF（Reciprocal Rank Fusion倒数排名融合）公式：
    它是一种把多个检索器的排名结果融合在一起的办法，不直接用各自的分数，而是只看排名
	对某个候选，如果在某个检索器里排名是rank，则加分1/(k0+rank)
	k0为超参数 越大，排名差距的影响越平滑；越小，前排优势越明显。rank为排名
	这里是把向量排名和ES排名都累加到同一个rrf_score上，实现融合。"""
    return score + 1.0 / float(k0 + rank)


def hybrid_search(
    *,
    q: str,
    k: int,  # 最终返回条数，强制在1~50
    allowed_visibilities: list[str],
    top_v: int = 50,  # 向量召回数量至少>=k，最多200
    top_b: int = 50,  # ES关键词召回数量至少>=k，最多200
    top_n_for_rerank: int = 30,  # rerank的候选数量，至少>=k，最多200
    rrf_k0: int = 60,  # RRF超参数，至少1
    min_rerank_score: float | None = None,
) -> list[Candidate]:
    """ hybrid_search()里混合了三步：
    向量检索（语义召回）+ 关键词检索（ES精确召回）+ 融合RRF、重排reranker
    语义召回：
    关键词检索：ES中的 倒排索引 无法做到关联 ，可以看成数据库的缓存
    """
    k = max(1, min(int(k), 50))
    top_v = max(k, min(int(top_v), 200))
    top_b = max(k, min(int(top_b), 200))
    top_n_for_rerank = max(k, min(int(top_n_for_rerank), 200))
    rrf_k0 = max(1, int(rrf_k0))

    vs = get_audio_vs()
    where = {"visibility": {"$in": allowed_visibilities}}

    docs_scores = vs.similarity_search_with_score(q, k=top_v, filter=where)

    # 去es里查关键词，返回top_b条命中
    es_hits = keyword_search(q=q, k=top_b, allowed_visibilities=allowed_visibilities)

    merged: dict[str, Candidate] = {}  # 把两路结果合并到merged

    for i, (doc, _score) in enumerate(docs_scores, start=1):
        md = getattr(doc, "metadata", None) or {}
        audio_id = str(md.get("audio_id") or "").strip()
        segment_id = str(md.get("segment_id") or "").strip()
        if not audio_id:
            continue
        if not segment_id:
            seg_idx = md.get("segment_idx")
            segment_id = f"{audio_id}:{seg_idx}" if seg_idx is not None else f"{audio_id}:unknown"

        key = segment_id
        start_ms = int(md.get("start_ms") or 0)
        end_ms = int(md.get("end_ms") or 0)
        text = (getattr(doc, "page_content", "") or "").strip()

        c = merged.get(key)
        if not c:
            c = Candidate(
                audio_id=audio_id,
                segment_id=segment_id,
                start_ms=start_ms,
                end_ms=end_ms,
                text=text,
                vec_rank=i,
                es_rank=None,
                rrf_score=0.0,
            )
            merged[key] = c
        c.vec_rank = c.vec_rank or i
        c.rrf_score = _rrf_add(c.rrf_score, i, rrf_k0)

    for j, h in enumerate(es_hits, start=1):
        if not h.audio_id:
            continue
        key = h.segment_id or f"{h.audio_id}:{h.segment_idx}"
        c = merged.get(key)
        if not c:
            c = Candidate(
                audio_id=h.audio_id,
                segment_id=key,
                start_ms=h.start_ms,
                end_ms=h.end_ms,
                text=(h.text or "").strip(),
                vec_rank=None,
                es_rank=j,
                rrf_score=0.0,
            )
            merged[key] = c
        c.es_rank = c.es_rank or j
        c.rrf_score = _rrf_add(c.rrf_score, j, rrf_k0)

    if not merged:
        return []

    # RRF排序 + 截断到rerank的候选数
    candidates = sorted(merged.values(), key=lambda x: x.rrf_score, reverse=True)  # 先按融合分数rrf_score从高到低排
    candidates = candidates[:top_n_for_rerank] # 截断：只取前top_n_for_rerank条进入rerank

    texts = [c.text for c in candidates]  # 把候选的text拿出来组成列表
    scores = rerank_scores(q, texts)  # rerank_scores返回等长的分数列表

    for c, s in zip(candidates, scores):  # 把每个分数写回Candidate.rerank_score
        c.rerank_score = float(s)

    # 按rerank_score重新排序，如果分数太低-1e9就放到后面
    candidates.sort(key=lambda x: (x.rerank_score if x.rerank_score is not None else -1e9), reverse=True)

    if min_rerank_score is not None:  # 如果设置了阈值：删掉rerank分数太低的候选
        candidates = [c for c in candidates if (c.rerank_score or -1e9) >= float(min_rerank_score)]

    return candidates[:k]