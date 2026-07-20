# -*- coding: utf-8 -*-
"""
linker.py

实体对齐模块。

职责边界：
    1. 接收 mention。
    2. 优先通过 alias_map 做高置信精确匹配。
    3. alias_map 未命中时，调用 retrieval/entity_vector_store.py 做实体向量候选召回。
    4.配。
    3. alias_map 未命中时，调用 retrieval/entity_vector_store.py 做实体向量候选召回。
    4. 返回统一候选结果，供 graph/nodes/entity_linking_node.py 使用。

注意：
    本文件属于 entity_resolution 领域能力层，但不直接实现 Chroma 查询、embedding 加载、
    LangGraph 状态更新、KG 路径搜索、LLM 调用或最终回答生成。
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .normalizer import (
    DEFAULT_ALIAS_PATH,
    load_alias_map,
    normalize_entity_name,
    normalize_mention,
    resolve_project_path,
)
from ..retrieval.embedding import DEFAULT_EMBEDDING_MODEL
from ..retrieval.entity_vector_store import (
    DEFAULT_CHROMA_DIR,
    DEFAULT_COLLECTION_NAME,
    EntityVectorStore,
)


# =========================================================
# 1. 默认配置
# =========================================================

DEFAULT_MODEL_NAME = DEFAULT_EMBEDDING_MODEL


# =========================================================
# 2. 通用工具
# =========================================================

def clip_score(score: Any) -> float:
    """
    将分数限制在 [0, 1]。
    """

    try:
        value = float(score)
    except Exception:
        return 0.0

    if value < 0:
        return 0.0

    if value > 1:
        return 1.0

    return value


def safe_float(value: Any, default: float = 0.0) -> float:
    """
    安全转 float。
    """

    try:
        return float(value)
    except Exception:
        return default


def first_non_empty_str(
    item: Dict[str, Any],
    keys: List[str],
) -> str:
    """
    从 dict 中按顺序获取第一个非空字符串。
    """

    for key in keys:
        value = item.get(key)

        if value is None:
            continue

        value = str(value).strip()

        if value:
            return value

    return ""


# =========================================================
# 3. alias 原始信息加载
# =========================================================

def load_alias_entries(alias_path: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """
    加载 alias_map.json，并保留 entity_id / node_key / aliases 等信息。

    与 entity_normalizer.load_alias_map() 的区别：
        load_alias_map():
            返回 alias -> entity_name

        load_alias_entries():
            返回 alias -> entity_info

    支持格式：
        1. alias -> entity_name
        2. entity_name -> [alias1, alias2, ...]
        3. alias -> {entity_id, entity_name, node_key, aliases, ...}
    """

    path = resolve_project_path(alias_path or DEFAULT_ALIAS_PATH)

    if not path.exists():
        return {}

    try:
        with open(path, "r", encoding="utf-8") as file:
            raw_alias_map = json.load(file)
    except Exception:
        return {}

    if not isinstance(raw_alias_map, dict):
        return {}

    entries: Dict[str, Dict[str, Any]] = {}

    for key, value in raw_alias_map.items():
        key_text = normalize_entity_name(key)

        if not key_text:
            continue

        # -------------------------------------------------
        # 格式 1：alias -> entity_name
        # -------------------------------------------------
        if isinstance(value, str):
            entity_name = normalize_entity_name(value)

            if not entity_name:
                continue

            info = {
                "entity_id": "",
                "entity_name": entity_name,
                "node_key": "",
                "aliases": [key_text],
                "source": "alias_map",
                "metadata": {},
            }

            _add_alias_entry(entries, key_text, info)
            _add_alias_entry(entries, entity_name, info)

            continue

        # -------------------------------------------------
        # 格式 2：entity_name -> aliases
        # -------------------------------------------------
        if isinstance(value, list):
            entity_name = normalize_entity_name(key_text)

            if not entity_name:
                continue

            aliases = [
                normalize_entity_name(alias)
                for alias in value
                if normalize_entity_name(alias)
            ]

            info = {
                "entity_id": "",
                "entity_name": entity_name,
                "node_key": "",
                "aliases": aliases,
                "source": "alias_map",
                "metadata": {},
            }

            _add_alias_entry(entries, entity_name, info)

            for alias in aliases:
                _add_alias_entry(entries, alias, info)

            continue

        # -------------------------------------------------
        # 格式 3：alias -> entity info dict
        # -------------------------------------------------
        if isinstance(value, dict):
            entity_name = first_non_empty_str(
                value,
                [
                    "entity_name",
                    "name",
                    "entity",
                    "title",
                    "label",
                ],
            )
            entity_name = normalize_entity_name(entity_name or key_text)

            if not entity_name:
                continue

            entity_id = first_non_empty_str(
                value,
                [
                    "entity_id",
                    "id",
                    "node_id",
                    "qid",
                    "key",
                ],
            )

            node_key = first_non_empty_str(
                value,
                [
                    "node_key",
                    "graph_node",
                    "graph_key",
                    "key",
                ],
            )

            aliases = value.get("aliases", [])

            if isinstance(aliases, str):
                aliases = [aliases]

            if not isinstance(aliases, list):
                aliases = []

            aliases = [
                normalize_entity_name(alias)
                for alias in aliases
                if normalize_entity_name(alias)
            ]

            if key_text not in aliases:
                aliases.append(key_text)

            if entity_name not in aliases:
                aliases.append(entity_name)

            info = {
                "entity_id": entity_id,
                "entity_name": entity_name,
                "node_key": node_key,
                "aliases": aliases,
                "source": str(value.get("source", "alias_map") or "alias_map"),
                "type": str(value.get("type", "") or ""),
                "score": clip_score(value.get("score", 1.0)),
                "metadata": {
                    item_key: item_value
                    for item_key, item_value in value.items()
                    if item_key not in {
                        "entity_id",
                        "id",
                        "node_id",
                        "qid",
                        "key",
                        "entity_name",
                        "name",
                        "entity",
                        "title",
                        "label",
                        "node_key",
                        "graph_node",
                        "graph_key",
                        "aliases",
                        "source",
                        "type",
                        "score",
                    }
                },
            }

            # 当前 key。
            _add_alias_entry(entries, key_text, info)

            # 标准实体名。
            _add_alias_entry(entries, entity_name, info)

            # entity_id / node_key 也作为可命中的 alias。
            if entity_id:
                _add_alias_entry(entries, entity_id, info)

            if node_key:
                _add_alias_entry(entries, node_key, info)

            # aliases。
            for alias in aliases:
                _add_alias_entry(entries, alias, info)

            continue

    return entries


def _add_alias_entry(
    entries: Dict[str, Dict[str, Any]],
    alias: str,
    info: Dict[str, Any],
) -> None:
    """
    向 alias entries 中加入一个 alias。
    """

    alias_key = normalize_mention(alias)

    if not alias_key:
        return

    entries[alias_key] = dict(info)


# =========================================================
# 4. EntityLinker
# =========================================================

class EntityLinker:
    """
    实体对齐器。

    使用示例：
        linker = EntityLinker()
        result = linker.link("美国", top_k=5)

    返回格式：
        {
            "mention": "美国",
            "normalized_mention": "美国",
            "candidates": [...],
            "selected_entity": "United States",
            "selected_entity_id": "united_states",
            "confidence": 1.0,
            "need_confirmation": False,
            "linking_method": "alias_map",
            "message": "对象名称已命中别名映射。"
        }
    """

    def __init__(
        self,
        chroma_dir: Optional[str] = None,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        model_name: str = DEFAULT_MODEL_NAME,
        alias_path: Optional[str] = None,
        auto_select_threshold: float = 0.72,
        margin_threshold: float = 0.05,
        local_files_only: bool = True,
        lazy_load: bool = True,
    ) -> None:
        """
        Args:
            chroma_dir:
                实体向量库目录。只传给 retrieval/entity_vector_store.py 使用。

            collection_name:
                实体向量库 collection 名称。

            model_name:
                embedding 模型名称。只传给 retrieval/entity_vector_store.py 使用。

            alias_path:
                alias_map.json 路径。

            auto_select_threshold:
                Top-1 分数超过该阈值时，允许自动选择。

            margin_threshold:
                Top-1 与 Top-2 分数差距超过该阈值时，认为候选足够明确。

            local_files_only:
                是否只加载本地模型。只传给 retrieval/entity_vector_store.py 使用。

            lazy_load:
                是否延迟加载实体向量库。
        """

        self.chroma_dir = str(chroma_dir or DEFAULT_CHROMA_DIR)
        self.collection_name = collection_name
        self.model_name = model_name
        self.alias_path = str(resolve_project_path(alias_path or DEFAULT_ALIAS_PATH))

        self.auto_select_threshold = float(auto_select_threshold)
        self.margin_threshold = float(margin_threshold)
        self.local_files_only = bool(local_files_only)
        self.lazy_load = bool(lazy_load)

        # alias_map:
        #   alias -> entity_name
        self.alias_map = load_alias_map(self.alias_path)

        # alias_entries:
        #   alias -> {entity_id, entity_name, node_key, aliases, ...}
        self.alias_entries = load_alias_entries(self.alias_path)

        self.vector_store: Optional[EntityVectorStore] = None
        self.last_vector_error: Optional[str] = None

        if not self.lazy_load:
            self._ensure_vector_store_ready()

    # =====================================================
    # 4.1 对外接口
    # =====================================================

    def link(
        self,
        mention: str,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        对单个 mention 进行实体对齐。
        """

        original_mention = normalize_entity_name(mention)
        normalized = normalize_mention(original_mention)

        if not normalized:
            return self._empty_link_result(
                mention=original_mention,
                normalized_mention=normalized,
                method="empty_input",
                message="输入为空，无法识别具体对象。",
            )

        top_k = max(int(top_k or 5), 1)

        # -------------------------------------------------
        # 1. alias_map / alias_entries 精确命中
        # -------------------------------------------------
        alias_info = self.alias_entries.get(normalized)

        if alias_info is None and normalized in self.alias_map:
            alias_info = {
                "entity_id": "",
                "entity_name": self.alias_map[normalized],
                "node_key": "",
                "aliases": [original_mention],
                "source": "alias_map",
                "metadata": {},
            }

        if alias_info is not None:
            entity_name = normalize_entity_name(alias_info.get("entity_name", ""))

            candidates = self._build_alias_candidates(
                mention=original_mention,
                alias_info=alias_info,
                top_k=top_k,
            )

            selected_entity_id = str(alias_info.get("entity_id", "") or "")

            if not selected_entity_id and candidates:
                selected_entity_id = str(candidates[0].get("entity_id", "") or "")

            return {
                "mention": original_mention,
                "normalized_mention": normalized,
                "candidates": candidates,
                "selected_entity": entity_name,
                "selected_entity_id": selected_entity_id,
                "confidence": 1.0,
                "score_margin": 1.0,
                "need_confirmation": False,
                "linking_method": "alias_map",
                "message": "对象名称已命中别名映射。",
            }

        # -------------------------------------------------
        # 2. 实体向量库候选召回
        # -------------------------------------------------
        vector_candidates = self._safe_query_entity_vector_store(
            mention=original_mention,
            top_k=top_k,
        )

        if not vector_candidates:
            return self._empty_link_result(
                mention=original_mention,
                normalized_mention=normalized,
                method="entity_vector_store",
                message="没有找到足够可靠的候选对象。",
            )

        top1 = vector_candidates[0]
        top2 = vector_candidates[1] if len(vector_candidates) > 1 else None

        top1_score = clip_score(top1.get("score", 0.0))
        top2_score = clip_score(top2.get("score", 0.0)) if top2 else 0.0
        margin = top1_score - top2_score

        can_auto_select = (
            top1_score >= self.auto_select_threshold
            and margin >= self.margin_threshold
        )

        if can_auto_select:
            selected_entity = str(
                top1.get("entity_name")
                or top1.get("entity")
                or ""
            )
            selected_entity_id = str(top1.get("entity_id") or "")
            need_confirmation = False
            message = "已找到高置信候选对象。"
        else:
            selected_entity = None
            selected_entity_id = None
            need_confirmation = True
            message = "候选对象相似度不足或差距较小，需要进一步确认。"

        return {
            "mention": original_mention,
            "normalized_mention": normalized,
            "candidates": vector_candidates,
            "selected_entity": selected_entity,
            "selected_entity_id": selected_entity_id,
            "confidence": top1_score,
            "score_margin": margin,
            "need_confirmation": need_confirmation,
            "linking_method": "entity_vector_store",
            "message": message,
        }

    def search(
        self,
        mention: str,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        link() 的别名，方便上层兼容调用。
        """

        return self.link(mention, top_k=top_k)

    def retrieve(
        self,
        mention: str,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        link() 的别名，方便上层兼容调用。
        """

        return self.link(mention, top_k=top_k)

    def batch_link(
        self,
        mentions: List[str],
        top_k: int = 5,
    ) -> Dict[str, Dict[str, Any]]:
        """
        批量实体对齐。
        """

        results: Dict[str, Dict[str, Any]] = {}

        for mention in mentions:
            mention_text = normalize_entity_name(mention)

            if not mention_text:
                continue

            results[mention_text] = self.link(mention_text, top_k=top_k)

        return results

    # =====================================================
    # 4.2 alias 命中候选构造
    # =====================================================

    def _build_alias_candidates(
        self,
        *,
        mention: str,
        alias_info: Dict[str, Any],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """
        alias_map 命中后构造候选。

        逻辑：
            1. 先构造一个 1.0 分的 alias 候选。
            2. 保留 entity_id / node_key / aliases。
            3. 如果实体向量库可用，再查询标准实体，补充更多候选。
        """

        entity_name = normalize_entity_name(alias_info.get("entity_name", ""))
        entity_id = str(alias_info.get("entity_id", "") or "")
        node_key = str(alias_info.get("node_key", "") or "")
        aliases = alias_info.get("aliases", []) or []

        if not isinstance(aliases, list):
            aliases = []

        aliases = [
            normalize_entity_name(alias)
            for alias in aliases
            if normalize_entity_name(alias)
        ]

        candidate = {
            "rank": 1,
            "entity": entity_name,
            "entity_name": entity_name,
            "entity_id": entity_id,
            "document": entity_name,
            "score": 1.0,
            "distance": 0.0,
            "source": "alias_map",
            "aliases": aliases,
            "metadata": {
                "alias": mention,
                "selected": True,
                "node_key": node_key,
                "graph_node": node_key,
                "type": alias_info.get("type", ""),
                "aliases": aliases,
                "alias_metadata": alias_info.get("metadata", {}) or {},
            },
        }

        vector_candidates = self._safe_query_entity_vector_store(
            mention=entity_name,
            top_k=top_k,
            suppress_error=True,
        )

        if not vector_candidates:
            return [candidate]

        # 优先找名称完全匹配项，补 entity_id / document / node_key。
        entity_key = normalize_mention(entity_name)

        for item in vector_candidates:
            item_name = normalize_entity_name(
                item.get("entity_name")
                or item.get("entity")
                or ""
            )

            if normalize_mention(item_name) != entity_key:
                continue

            if not candidate["entity_id"]:
                candidate["entity_id"] = item.get("entity_id", "") or ""

            candidate["document"] = item.get("document", entity_name) or entity_name

            item_metadata = item.get("metadata", {}) or {}

            if isinstance(item_metadata, dict):
                if not candidate["metadata"].get("node_key"):
                    candidate["metadata"]["node_key"] = (
                        item_metadata.get("node_key")
                        or item_metadata.get("graph_node")
                        or item_metadata.get("graph_key")
                        or ""
                    )

            break

        # 合并其他候选，避免只返回一个。
        merged = [candidate]
        seen = {normalize_mention(entity_name)}

        for item in vector_candidates:
            item_name = normalize_entity_name(
                item.get("entity_name")
                or item.get("entity")
                or ""
            )

            key = normalize_mention(item_name)

            if not key or key in seen:
                continue

            seen.add(key)
            merged.append(item)

        return merged[:top_k]

    # =====================================================
    # 4.3 retrieval 层调用
    # =====================================================

    def _safe_query_entity_vector_store(
        self,
        *,
        mention: str,
        top_k: int,
        suppress_error: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        安全调用 retrieval/entity_vector_store.py。

        注意：
            这里不直接加载 Chroma，不直接加载 embedding 模型，
            只调用 retrieval 层已经封装好的实体向量库接口。
        """

        try:
            self.last_vector_error = None

            return self._query_entity_vector_store(
                mention=mention,
                top_k=top_k,
            )

        except Exception as exc:
            self.last_vector_error = str(exc)

            if suppress_error:
                return []

            return []

    def _query_entity_vector_store(
        self,
        *,
        mention: str,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """
        调用实体向量库查询 Top-K 候选。
        """

        mention = normalize_entity_name(mention)

        if not mention:
            return []

        self._ensure_vector_store_ready()

        if self.vector_store is None:
            return []

        results = self.vector_store.query(
            mention,
            top_k=int(top_k),
        )

        return self._normalize_vector_results(results)

    def _ensure_vector_store_ready(self) -> None:
        """
        延迟初始化 EntityVectorStore。
        """

        if self.vector_store is not None:
            return

        self.vector_store = EntityVectorStore(
            chroma_dir=self.chroma_dir,
            collection_name=self.collection_name,
            model_name=self.model_name,
            local_files_only=self.local_files_only,
            create_if_missing=False,
            lazy_load=True,
        )

    def _normalize_vector_results(
        self,
        results: Any,
    ) -> List[Dict[str, Any]]:
        """
        将 retrieval/entity_vector_store.py 的结果规范成 EntityLinker 候选格式。
        """

        if not isinstance(results, list):
            return []

        candidates: List[Dict[str, Any]] = []

        for idx, item in enumerate(results):
            if not isinstance(item, dict):
                continue

            raw_metadata = item.get("metadata", {})

            if not isinstance(raw_metadata, dict):
                raw_metadata = {}

            entity_name = normalize_entity_name(
                item.get("entity_name")
                or item.get("entity")
                or item.get("name")
                or raw_metadata.get("entity_name")
                or raw_metadata.get("name")
                or item.get("document")
                or ""
            )

            if not entity_name:
                continue

            entity_id = str(
                item.get("entity_id")
                or item.get("id")
                or item.get("node_id")
                or item.get("vector_id")
                or raw_metadata.get("entity_id")
                or raw_metadata.get("id")
                or raw_metadata.get("node_id")
                or ""
            )

            node_key = str(
                item.get("node_key")
                or item.get("graph_node")
                or item.get("graph_key")
                or raw_metadata.get("node_key")
                or raw_metadata.get("graph_node")
                or raw_metadata.get("graph_key")
                or ""
            )

            aliases = (
                item.get("aliases")
                or raw_metadata.get("aliases")
                or []
            )

            if isinstance(aliases, str):
                aliases = [aliases]

            if not isinstance(aliases, list):
                aliases = []

            score = clip_score(item.get("score", 0.0))
            distance = safe_float(item.get("distance", 1.0), default=1.0)
            document = str(item.get("document") or entity_name)
            source = str(item.get("source") or "entity_vector_store")

            candidates.append(
                {
                    "rank": idx + 1,
                    "entity": entity_name,
                    "entity_name": entity_name,
                    "entity_id": entity_id,
                    "document": document,
                    "score": score,
                    "distance": distance,
                    "source": source,
                    "aliases": aliases,
                    "metadata": {
                        "rank": idx + 1,
                        "vector_id": str(item.get("vector_id") or ""),
                        "node_key": node_key,
                        "graph_node": node_key,
                        "aliases": aliases,
                        "raw_metadata": raw_metadata,
                    },
                }
            )

        return candidates

    # =====================================================
    # 4.4 空结果
    # =====================================================

    def _empty_link_result(
        self,
        *,
        mention: str,
        normalized_mention: str,
        method: str,
        message: str,
    ) -> Dict[str, Any]:
        """
        构造空实体对齐结果。
        """

        return {
            "mention": mention,
            "normalized_mention": normalized_mention,
            "candidates": [],
            "selected_entity": None,
            "selected_entity_id": None,
            "confidence": 0.0,
            "score_margin": 0.0,
            "need_confirmation": True,
            "linking_method": method,
            "message": message,
        }


# =========================================================
# 5. 便捷函数
# =========================================================

_GLOBAL_LINKER: Optional[EntityLinker] = None


def get_default_linker() -> EntityLinker:
    """
    获取全局默认 EntityLinker。

    避免每次 link_entity() 都重新初始化实体向量库。
    """

    global _GLOBAL_LINKER

    if _GLOBAL_LINKER is None:
        _GLOBAL_LINKER = EntityLinker()

    return _GLOBAL_LINKER


def link_entity(
    mention: str,
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    简单函数接口。
    """

    linker = get_default_linker()

    return linker.link(mention, top_k=top_k)


def batch_link_entities(
    mentions: List[str],
    top_k: int = 5,
) -> Dict[str, Dict[str, Any]]:
    """
    批量实体对齐函数接口。
    """

    linker = get_default_linker()

    return linker.batch_link(mentions, top_k=top_k)


# =========================================================
# 6. 快速测试入口
# =========================================================

if __name__ == "__main__":
    linker = EntityLinker()

    test_mentions = [
        "Barack Obama",
        "Obama",
        "奥巴马",
        "Michelle Obama",
        "美国",
        "United States",
        "Hawaii",
        "not exist entity",
    ]

    for test_mention in test_mentions:
        print("\n" + "=" * 80)
        print(f"mention = {test_mention}")
        test_result = linker.link(test_mention, top_k=5)
        print(json.dumps(test_result, ensure_ascii=False, indent=2))