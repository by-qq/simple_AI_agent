from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterable

from elasticsearch import Elasticsearch, helpers

from app.config import settings


AUDIO_INDEX = getattr(settings, "es_audio_index", "audio_segments_v1")
# ES中的索引名字，读取es_audio_index的值，没有就取audio_segments_v1

@lru_cache(maxsize=1)
def es_client() -> Elasticsearch:
    return Elasticsearch(
        hosts=[settings.es_url],
        request_timeout=30,
        retry_on_timeout=True,
        max_retries=3,
    )


def ensure_audio_index() -> None:
    es = es_client()
    if es.indices.exists(index=AUDIO_INDEX):
        return

    mappings = {
        "properties": {
            "audio_id": {"type": "keyword"},
            "segment_id": {"type": "keyword"},
            "segment_idx": {"type": "integer"},
            "start_ms": {"type": "integer"},
            "end_ms": {"type": "integer"},
            "visibility": {"type": "keyword"},  # 精确用来检索的，不分词
            "text": {
                "type": "text",  # 人类读，分词
                "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
            },
        }
    }

    settings_body = {
        "index": {
            "number_of_shards": 1,  # 整个索引只有一个主分片
            "number_of_replicas": 0,  # 不设置副本
            "refresh_interval": "1s", # 定期刷新索引使新写入的数据可以被搜索到
        }
    }

    es.indices.create(index=AUDIO_INDEX, mappings=mappings, settings=settings_body)


def reset_audio_index() -> None:
    es = es_client()
    if es.indices.exists(index=AUDIO_INDEX):
        es.indices.delete(index=AUDIO_INDEX)
    ensure_audio_index()


def upsert_audio_segments(*, audio_id: str, rows: list[dict[str, Any]]) -> int:
    """把某个音频文件的多个segments批量写入或更新到es中"""
    ensure_audio_index()
    es = es_client()

    actions = []
    for r in rows:
        seg_id = str(r.get("segment_id") or "").strip()
        seg_idx = int(r.get("segment_idx"))
        actions.append(
            {
                "_op_type": "index",
                "_index": AUDIO_INDEX,
                "_id": f"{audio_id}:{seg_idx}",
                "_source": {
                    "audio_id": audio_id,
                    "segment_id": seg_id or f"{audio_id}:{seg_idx}",
                    "segment_idx": seg_idx,
                    "start_ms": int(r.get("start_ms") or 0),
                    "end_ms": int(r.get("end_ms") or 0),
                    "text": str(r.get("text") or ""),
                    "visibility": str(r.get("visibility") or "public"),
                },
            }
        )

    if not actions:
        return 0

    ok, _ = helpers.bulk(es, actions, refresh=True)
    return int(ok)


def delete_by_audio_id(audio_id: str) -> int:
    ensure_audio_index()
    es = es_client()
    resp = es.delete_by_query(
        index=AUDIO_INDEX,
        query={"term": {"audio_id": audio_id}},  # term查询是精确匹配，不分词
        refresh=True,  # 操作完成后立即刷新索引
        conflicts="proceed",  # 冲突时继续执行，不报错
    )
    return int(resp.get("deleted") or 0)


def delete_many_audio_ids(audio_ids: Iterable[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for aid in audio_ids:
        aid = (aid or "").strip()
        if not aid:
            continue
        try:
            out[aid] = int(delete_by_audio_id(aid))
        except Exception:
            out[aid] = 0
    return out


def update_visibility_by_audio_id(audio_id: str, visibility: str) -> int:
    ensure_audio_index()
    es = es_client()
    resp = es.update_by_query(
        index=AUDIO_INDEX,
        query={"term": {"audio_id": audio_id}},
        script={
            "source": "ctx._source.visibility = params.v",
            "lang": "painless",  # Painless是es官方默认的脚本语言
            "params": {"v": visibility},
        },
        refresh=True,
        conflicts="proceed",
    )
    return int(resp.get("updated") or 0)


@dataclass(frozen=True)  # 初始化后不能再修改它的属性
class ESKeywordHit:  # 这里主要存放es命中结果
    audio_id: str
    segment_id: str
    segment_idx: int
    start_ms: int
    end_ms: int
    text: str
    score: float

def keyword_search(*, q: str, k: int, allowed_visibilities: list[str]) -> list[ESKeywordHit]:
    ensure_audio_index()
    es = es_client()

    k = max(1, min(int(k), 200))
    allowed = [v for v in (allowed_visibilities or []) if v]

    body = {
        "size": k,
        "query": {
            "bool": {
                "must": [  # WHERE条件
                    {
                        "simple_query_string": {  # 全文搜索匹配, 类似于mysql的MATCH(text) AGAINST()
                            "query": q,
                            "fields": ["text"],
                            "default_operator": "and",  # 默认操作用AND visibility IN (...)
                        }
                    }
                ],
                "filter": [{"terms": {"visibility": allowed}}] if allowed else [],
            }
        },
    }  # 类似于sql SELECT * FROM audio_segments WHERE MATCH(text) AGAINST (:q IN BOOLEAN MODE)AND visibility IN (:allowed) LIMIT :k;

    resp = es.search(index=AUDIO_INDEX, **body)
    hits = resp.get("hits", {}).get("hits", []) or []

    out: list[ESKeywordHit] = []
    for h in hits:
        src = h.get("_source") or {}
        out.append(
            ESKeywordHit(
                audio_id=str(src.get("audio_id") or ""),
                segment_id=str(src.get("segment_id") or ""),
                segment_idx=int(src.get("segment_idx") or 0),
                start_ms=int(src.get("start_ms") or 0),
                end_ms=int(src.get("end_ms") or 0),
                text=str(src.get("text") or ""),
                score=float(h.get("_score") or 0.0),
            )
        )
    return out


if __name__ == "__main__":
    import sys
    import time

    # 测试前确保配置正确
    if not hasattr(settings, "es_url"):
        # 为测试设置一个默认的 ES URL
        settings.es_url = "http://localhost:9200"


    def test_ensure_audio_index():
        """测试索引创建"""
        print("测试 ensure_audio_index...")
        try:
            ensure_audio_index()
            print("✓ 索引创建/检查成功")
        except Exception as e:
            print(f"✗ 索引创建失败: {e}")
        print()


    def test_reset_audio_index():
        """测试重置索引"""
        print("测试 reset_audio_index...")
        try:
            reset_audio_index()
            print("✓ 索引重置成功")
        except Exception as e:
            print(f"✗ 索引重置失败: {e}")
        print()


    def test_upsert_audio_segments():
        """测试批量插入/更新音频片段"""
        print("测试 upsert_audio_segments...")

        # 测试数据
        rows = [
            {
                "segment_id": "seg_001",
                "segment_idx": 1,
                "start_ms": 0,
                "end_ms": 5000,
                "text": "这是一个测试音频片段，包含一些测试文本",
                "visibility": "public"
            },
            {
                "segment_id": "seg_002",
                "segment_idx": 2,
                "start_ms": 5000,
                "end_ms": 10000,
                "text": "第二个音频片段，继续测试功能",
                "visibility": "private"
            },
            {
                "segment_id": "seg_003",
                "segment_idx": 3,
                "start_ms": 10000,
                "end_ms": 15000,
                "text": "测试 Elasticsearch 搜索功能",
                "visibility": "public"
            }
        ]

        try:
            audio_id = "test_audio_001"
            count = upsert_audio_segments(audio_id=audio_id, rows=rows)
            print(f"✓ 成功插入 {count} 个音频片段 (audio_id: {audio_id})")
        except Exception as e:
            print(f"✗ 插入失败: {e}")
        print()


    def test_keyword_search():
        """测试关键词搜索"""
        print("测试 keyword_search...")
        try:
            # 等待 ES 刷新索引
            time.sleep(1)

            # 搜索测试
            results = keyword_search(
                q="测试音频",
                k=10,
                allowed_visibilities=["public"]
            )

            print(f"✓ 搜索到 {len(results)} 个结果:")
            for i, hit in enumerate(results[:3]):  # 只显示前3个结果
                print(f"  结果 {i + 1}: audio_id={hit.audio_id}, text={hit.text[:50]}...")

        except Exception as e:
            print(f"✗ 搜索失败: {e}")
        print()


    def test_update_visibility_by_audio_id():
        """测试更新音频可见性"""
        print("测试 update_visibility_by_audio_id...")
        try:
            audio_id = "test_audio_001"
            updated = update_visibility_by_audio_id(
                audio_id=audio_id,
                visibility="private"
            )
            print(f"✓ 成功更新 {updated} 个文档的可见性")

            # 验证更新效果
            time.sleep(1)
            results = keyword_search(
                q="测试",
                k=10,
                allowed_visibilities=["public"]
            )
            print(f"  更新后，public 可见的文档数: {len(results)}")

        except Exception as e:
            print(f"✗ 更新失败: {e}")
        print()


    def test_delete_by_audio_id():
        """测试删除音频"""
        print("测试 delete_by_audio_id...")
        try:
            audio_id = "test_audio_001"
            deleted = delete_by_audio_id(audio_id)
            print(f"✓ 成功删除 {deleted} 个音频片段")

            # 验证删除效果
            time.sleep(1)
            results = keyword_search(
                q="测试",
                k=10,
                allowed_visibilities=["public", "private"]
            )
            print(f"  删除后，剩余文档数: {len(results)}")

        except Exception as e:
            print(f"✗ 删除失败: {e}")
        print()


    def test_delete_many_audio_ids():
        """测试批量删除音频"""
        print("测试 delete_many_audio_ids...")

        # 先插入一些测试数据
        test_audios = ["test_audio_002", "test_audio_003"]
        for audio_id in test_audios:
            rows = [
                {
                    "segment_idx": 1,
                    "start_ms": 0,
                    "end_ms": 5000,
                    "text": f"测试数据 {audio_id}",
                    "visibility": "public"
                }
            ]
            upsert_audio_segments(audio_id=audio_id, rows=rows)

        time.sleep(1)

        try:
            # 批量删除
            results = delete_many_audio_ids(test_audios)
            print(f"✓ 批量删除结果: {results}")

        except Exception as e:
            print(f"✗ 批量删除失败: {e}")
        print()


    def test_es_client_cache():
        """测试 ES 客户端缓存"""
        print("测试 es_client 缓存...")
        try:
            client1 = es_client()
            client2 = es_client()

            if client1 is client2:
                print("✓ ES 客户端已正确缓存 (单例模式)")
            else:
                print("✗ ES 客户端未正确缓存")

        except Exception as e:
            print(f"✗ 测试失败: {e}")
        print()


    def run_all_tests():
        """运行所有测试"""
        print("=" * 50)
        print("开始 Elasticsearch 接口测试")
        print("=" * 50)
        print()

        # 注意：测试顺序很重要
        test_es_client_cache()
        test_reset_audio_index()  # 先重置索引，清理环境
        test_ensure_audio_index()  # 确保索引存在
        test_upsert_audio_segments()  # 插入测试数据
        test_keyword_search()  # 测试搜索
        test_update_visibility_by_audio_id()  # 测试更新可见性
        test_delete_by_audio_id()  # 测试删除单个音频
        test_delete_many_audio_ids()  # 测试批量删除

        print("=" * 50)
        print("测试完成！")
        print("=" * 50)


    # 运行测试
    run_all_tests()