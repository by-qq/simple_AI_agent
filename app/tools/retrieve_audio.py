from __future__ import annotations


from app.deps import get_vs
from app.security.kb_visibility import compute_allowed_kb_visibilities



def audio_similarity_search_for_user(query: str, user, k: int = 6):
    vs = get_vs()
    allowed = compute_allowed_kb_visibilities(user)
#     这里主要是版本不一定，我们到时候看具体怎么处理，主要是参数名都不一样
    try:
        docs = vs.similarity_search(query, k=k, filter={"visibility": {"$in": allowed}})
    except TypeError:
        docs = vs.similarity_search(query, k=k, where={"visibility": {"$in": allowed}})

    return docs, allowed
    # docs = vs.similarity_search(query, k=k, filter={"visibility": {"$in": ['public']}})
    #
    # return docs, ['public']

