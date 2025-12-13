import json
import redis
from typing import Any
from app.config import settings

r = redis.Redis(
    host=settings.REDIS_HOST,
    port=settings.REDIS_PORT,
    decode_responses=True,
)


# Keys that often contain non-JSON-serializable objects (LangChain Documents, Messages, etc.)
DROP_KEYS = {"docs", "messages", "chat_history", "retrieved_docs"}

# 复杂对象没办法序列化而准备的，序列化就是将xx内容变成字节的序列，反序列化就是将字节的系列再变回来
def _safe_dumps(obj: Any) -> str:
    """Dump to JSON, falling back to str() for unknown objects."""
    return json.dumps(obj, ensure_ascii=False, default=str)


def load_session(session_id: str) -> dict | None:
    s = r.get(session_id)
    return json.loads(s) if s else None


def save_session(session_id: str, state: dict) -> None:
    # 过滤键值对
    safe_state = {k: v for k, v in state.items() if k not in DROP_KEYS}
    r.setex(session_id, settings.REDIS_TTL_SECONDS, _safe_dumps(safe_state))
