# -*- coding: utf-8 -*-
"""
normalizer.py

对象名称标准化模块。

作用：
    1. 统一处理 mention / entity name / node key 的文本规范化。
    2. 加载 alias_map.json，并提供别名映射能力。
    3. 提供标准对象名称、节点 key、候选对象的规范化工具。
    4. 为 entity_linker.py、entity_grounding_node.py、graph_builder.py 提供通用能力。

本文件属于 kg 底层能力层：
    kg/
        entity_normalizer.py

它不负责：
    1. 向量检索。
    2. 图路径查询。
    3. 最终回答生成。
    4. LangGraph 节点调度。
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


# =========================================================
# 1. 默认路径
# =========================================================

DEFAULT_ALIAS_PATH = "data/demo/processed/alias_map.json"


# =========================================================
# 2. 项目路径工具
# =========================================================

def get_project_root() -> Path:
    """
    获取项目根目录。

    当前文件：
        src/kg_rag_agent/kg/entity_normalizer.py

    parents:
        0 -> kg/
        1 -> kg_rag_agent/
        2 -> src/
        3 -> 项目根目录
    """

    return Path(__file__).resolve().parents[3]


def resolve_project_path(path: str | Path) -> Path:
    """
    解析项目路径。

    支持：
        1. 绝对路径
        2. 相对于当前工作目录
        3. 相对于项目根目录
    """

    input_path = Path(path)

    if input_path.is_absolute():
        return input_path

    cwd_path = Path.cwd() / input_path

    if cwd_path.exists():
        return cwd_path

    return get_project_root() / input_path


# =========================================================
# 3. 基础文本标准化
# =========================================================

def normalize_text(
    text: Any,
    *,
    lowercase: bool = True,
    remove_accents: bool = True,
    normalize_space: bool = True,
    replace_underscore: bool = True,
    replace_hyphen: bool = False,
) -> str:
    """
    通用文本标准化。
    """

    if text is None:
        return ""

    value = str(text).strip()

    if not value:
        return ""

    value = unicodedata.normalize("NFKC", value)

    if remove_accents:
        value = _remove_accents(value)

    if replace_underscore:
        value = value.replace("_", " ")

    if replace_hyphen:
        value = value.replace("-", " ")

    if normalize_space:
        value = re.sub(r"\s+", " ", value).strip()

    if lowercase:
        value = value.lower()

    return value


def normalize_mention(text: Any) -> str:
    """
    mention 标准化。

    用于：
        1. 用户输入对象名称。
        2. alias_map key。
        3. 模糊匹配 key。
    """

    value = normalize_text(
        text,
        lowercase=True,
        remove_accents=True,
        normalize_space=True,
        replace_underscore=True,
        replace_hyphen=False,
    )

    value = strip_outer_punctuation(value)

    return value


def normalize_entity_name(text: Any) -> str:
    """
    标准对象名称清洗。

    与 normalize_mention 不同：
        1. 保留大小写。
        2. 只做空格、Unicode、首尾标点清理。
    """

    if text is None:
        return ""

    value = str(text).strip()
    value = unicodedata.normalize("NFKC", value)
    value = re.sub(r"\s+", " ", value).strip()
    value = strip_outer_punctuation(value)

    return value


def normalize_node_key(text: Any) -> str:
    """
    图节点 key 标准化。

    用于构建 node index。
    """

    return normalize_text(
        text,
        lowercase=True,
        remove_accents=True,
        normalize_space=True,
        replace_underscore=True,
        replace_hyphen=True,
    )


def strip_outer_punctuation(text: str) -> str:
    """
    去掉首尾常见标点。
    """

    if not text:
        return ""

    return text.strip(
        " \t\r\n"
        ".,;:!?，。；：！？"
        "\"'“”‘’"
        "()（）[]【】{}<>《》"
        "、"
    )


def _remove_accents(text: str) -> str:
    """
    去除重音符号。

    示例：
        Beyoncé -> Beyonce
    """

    normalized = unicodedata.normalize("NFD", text)

    return "".join(
        char for char in normalized
        if unicodedata.category(char) != "Mn"
    )


# =========================================================
# 4. mention 清洗
# =========================================================

def clean_mention_text(text: Any) -> str:
    """
    清洗从用户问题中抽取出来的 mention。

    用于 mention_extraction_node 或 entity_linker 前置处理。
    """

    value = normalize_entity_name(text)

    if not value:
        return ""

    prefix_patterns = [
        r"^请问",
        r"^帮我看看",
        r"^帮我查一下",
        r"^帮我分析一下",
        r"^我想知道",
        r"^我想问",
        r"^关于",
        r"^有关",
        r"^介绍一下",
        r"^介绍",
        r"^解释一下",
        r"^解释",
        r"^说明一下",
        r"^说明",
        r"^the\s+",
        r"^a\s+",
        r"^an\s+",
    ]

    for pattern in prefix_patterns:
        value = re.sub(pattern, "", value, flags=re.IGNORECASE).strip()

    suffix_patterns = [
        r"是什么$",
        r"是谁$",
        r"有哪些$",
        r"有什么$",
        r"怎么样$",
        r"怎么回事$",
        r"的关系$",
        r"之间的关系$",
        r"相关内容$",
        r"相关对象$",
        r"的信息$",
        r"的情况$",
        r"\?$",
        r"？$",
        r"。$",
        r"\.$",
    ]

    for pattern in suffix_patterns:
        value = re.sub(pattern, "", value, flags=re.IGNORECASE).strip()

    value = strip_outer_punctuation(value)

    return value


def is_valid_entity_text(text: Any) -> bool:
    """
    判断文本是否适合作为对象名称。
    """

    value = normalize_entity_name(text)

    if not value:
        return False

    if len(value) > 120:
        return False

    if len(value) <= 1 and not re.match(r"[A-Z]", value):
        return False

    if re.fullmatch(r"\d+", value):
        return False

    if re.fullmatch(r"[\W_]+", value):
        return False

    if normalize_mention(value) in STOP_ENTITY_TEXTS:
        return False

    return True


STOP_ENTITY_TEXTS = {
    "谁",
    "什么",
    "哪里",
    "哪个",
    "哪些",
    "多少",
    "如何",
    "为什么",
    "怎么",
    "怎么办",
    "怎么样",
    "关系",
    "路径",
    "实体",
    "对象",
    "内容",
    "信息",
    "情况",
    "介绍",
    "说明",
    "解释",
    "分析",
    "这个",
    "那个",
    "它",
    "他",
    "她",
    "who",
    "what",
    "where",
    "which",
    "why",
    "how",
    "relation",
    "relationship",
    "path",
    "entity",
    "object",
    "thing",
    "information",
    "about",
    "explain",
    "describe",
    "tell",
    "me",
}


# =========================================================
# 5. alias_map 加载与标准化
# =========================================================

def load_alias_map(alias_path: Optional[str] = None) -> Dict[str, str]:
    """
    加载 alias_map.json。

    支持格式 1：alias -> entity
        {
            "美国": "United States",
            "usa": "United States"
        }

    支持格式 2：entity -> aliases
        {
            "United States": ["美国", "usa", "US"]
        }

    支持格式 3：alias -> entity info dict
        {
            "Obama": {
                "entity_id": "barack_obama",
                "entity_name": "Barack Obama",
                "node_key": "barack_obama",
                "aliases": ["Barack Obama", "Obama", "奥巴马"]
            }
        }

    返回格式：
        {
            normalize_mention(alias): standard_entity_name
        }
    """

    path = resolve_project_path(alias_path or DEFAULT_ALIAS_PATH)

    if not path.exists():
        return {}

    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return normalize_alias_map(data)

def save_alias_map(
    alias_map: Dict[str, Any],
    alias_path: Optional[str] = None,
) -> Path:
    """
    保存 alias_map.json。

    注意：
        这里不会强制改变用户传入结构，只做 JSON 保存。
    """

    path = resolve_project_path(alias_path or DEFAULT_ALIAS_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(
            alias_map,
            file,
            ensure_ascii=False,
            indent=2,
        )

    return path


def normalize_alias_map(raw_alias_map: Dict[str, Any]) -> Dict[str, str]:
    """
    将任意 alias_map 结构统一转成 alias -> entity。

    兼容三种常见结构：
        1. alias -> entity_name
        2. entity_name -> [alias1, alias2, ...]
        3. alias -> {entity_id, entity_name/name, node_key, aliases, ...}
    """

    normalized: Dict[str, str] = {}

    if not isinstance(raw_alias_map, dict):
        return normalized

    for key, value in raw_alias_map.items():
        # 格式 1：alias -> entity
        if isinstance(value, str):
            alias_key = normalize_mention(key)
            entity_name = normalize_entity_name(value)

            if alias_key and entity_name:
                normalized[alias_key] = entity_name

            continue

        # 格式 2：entity -> aliases
        if isinstance(value, list):
            entity_name = normalize_entity_name(key)

            if entity_name:
                entity_key = normalize_mention(entity_name)
                if entity_key:
                    normalized[entity_key] = entity_name

            for alias in value:
                alias_key = normalize_mention(alias)

                if alias_key and entity_name:
                    normalized[alias_key] = entity_name

            continue

        # 格式 3：alias -> entity info dict
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
            entity_name = normalize_entity_name(entity_name or key)

            if not entity_name:
                continue

            # 当前 key 本身就是 alias。
            alias_key = normalize_mention(key)
            if alias_key:
                normalized[alias_key] = entity_name

            # 标准实体名也加入 alias_map。
            entity_key = normalize_mention(entity_name)
            if entity_key:
                normalized[entity_key] = entity_name

            # entity_id / node_key 也加入，方便后续直接输入节点 key。
            for id_key in ["entity_id", "id", "node_id", "node_key", "key"]:
                alias_value = value.get(id_key)
                alias_key = normalize_mention(alias_value)

                if alias_key:
                    normalized[alias_key] = entity_name

            # aliases 字段加入。
            aliases = value.get("aliases", [])

            if isinstance(aliases, str):
                aliases = [aliases]

            if isinstance(aliases, list):
                for alias in aliases:
                    alias_key = normalize_mention(alias)

                    if alias_key:
                        normalized[alias_key] = entity_name

            continue

    return normalized


def resolve_alias(
    text: Any,
    alias_map: Dict[str, str],
) -> Optional[str]:
    """
    根据 alias_map 解析标准对象名称。
    """

    key = normalize_mention(text)

    if not key:
        return None

    return alias_map.get(key)


# =========================================================
# 6. EntityNormalizer 类
# =========================================================

class EntityNormalizer:
    """
    对象名称标准化器。

    用法：
        normalizer = EntityNormalizer()
        entity = normalizer.resolve("美国")
    """

    def __init__(
        self,
        alias_path: Optional[str] = None,
        alias_map: Optional[Dict[str, str]] = None,
    ) -> None:
        self.alias_path = alias_path or DEFAULT_ALIAS_PATH

        if alias_map is not None:
            self.alias_map = normalize_alias_map(alias_map)
        else:
            self.alias_map = load_alias_map(self.alias_path)

    def normalize_mention(self, text: Any) -> str:
        """
        标准化 mention。
        """

        return normalize_mention(text)

    def normalize_entity_name(self, text: Any) -> str:
        """
        清洗标准对象名称。
        """

        return normalize_entity_name(text)

    def normalize_node_key(self, text: Any) -> str:
        """
        标准化节点 key。
        """

        return normalize_node_key(text)

    def clean_mention(self, text: Any) -> str:
        """
        清洗 mention 文本。
        """

        return clean_mention_text(text)

    def is_valid(self, text: Any) -> bool:
        """
        判断是否是有效对象文本。
        """

        return is_valid_entity_text(text)

    def resolve(self, text: Any) -> Optional[str]:
        """
        通过 alias_map 解析标准对象名称。
        """

        return resolve_alias(text, self.alias_map)

    def normalize_or_resolve(self, text: Any) -> str:
        """
        优先通过 alias_map 解析；
        如果没有命中，则返回清洗后的对象名称。
        """

        resolved = self.resolve(text)

        if resolved:
            return resolved

        return normalize_entity_name(text)

    def add_alias(
        self,
        alias: str,
        entity_name: str,
    ) -> None:
        """
        添加别名映射到内存中。
        """

        alias_key = normalize_mention(alias)
        entity_name = normalize_entity_name(entity_name)

        if alias_key and entity_name:
            self.alias_map[alias_key] = entity_name

    def batch_resolve(
        self,
        texts: Iterable[Any],
    ) -> Dict[str, str]:
        """
        批量解析对象名称。
        """

        results: Dict[str, str] = {}

        for text in texts:
            original = normalize_entity_name(text)

            if not original:
                continue

            results[original] = self.normalize_or_resolve(original)

        return results


# =========================================================
# 7. 候选对象规范化
# =========================================================

def normalize_candidate(
    candidate: Any,
    *,
    mention: str = "",
    default_score: float = 0.0,
) -> Dict[str, Any]:
    """
    将不同格式的候选对象统一成 dict。

    支持：
        1. str
        2. dict
    """

    mention = normalize_entity_name(mention)

    if candidate is None:
        return {}

    if isinstance(candidate, str):
        entity_name = normalize_entity_name(candidate)

        if not entity_name:
            return {}

        return {
            "mention": mention,
            "entity_id": "",
            "entity_name": entity_name,
            "score": clip_score(default_score),
            "source": "string",
            "aliases": [],
            "metadata": {},
        }

    if isinstance(candidate, dict):
        entity_name = first_non_empty_str(
            candidate,
            [
                "entity_name",
                "entity",
                "name",
                "title",
                "label",
                "document",
            ],
        )

        entity_name = normalize_entity_name(entity_name)

        if not entity_name:
            return {}

        entity_id = first_non_empty_str(
            candidate,
            [
                "entity_id",
                "id",
                "node_id",
                "qid",
                "key",
            ],
        )

        score = extract_score(candidate, default=default_score)

        aliases = candidate.get("aliases", [])

        if not isinstance(aliases, list):
            aliases = []

        return {
            "mention": mention or normalize_entity_name(candidate.get("mention", "")),
            "entity_id": entity_id,
            "entity_name": entity_name,
            "score": score,
            "source": str(candidate.get("source", "unknown") or "unknown"),
            "aliases": aliases,
            "metadata": {
                key: value
                for key, value in candidate.items()
                if key not in {
                    "mention",
                    "entity_id",
                    "id",
                    "node_id",
                    "qid",
                    "key",
                    "entity_name",
                    "entity",
                    "name",
                    "title",
                    "label",
                    "document",
                    "score",
                    "similarity",
                    "confidence",
                    "distance",
                    "source",
                    "aliases",
                }
            },
        }

    return {}


def normalize_candidates(
    candidates: Iterable[Any],
    *,
    mention: str = "",
    min_score: float = 0.0,
) -> List[Dict[str, Any]]:
    """
    批量规范化候选对象。
    """

    normalized: List[Dict[str, Any]] = []

    for candidate in candidates:
        item = normalize_candidate(
            candidate,
            mention=mention,
            default_score=0.0,
        )

        if not item:
            continue

        score = float(item.get("score", 0.0))

        if score < min_score:
            continue

        normalized.append(item)

    normalized = deduplicate_candidates(normalized)

    return sorted(
        normalized,
        key=lambda item: float(item.get("score", 0.0)),
        reverse=True,
    )


def deduplicate_candidates(
    candidates: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    候选对象去重。

    优先按 entity_id 去重；
    没有 entity_id 时按 entity_name 标准化结果去重。
    """

    best_by_key: Dict[str, Dict[str, Any]] = {}

    for candidate in candidates:
        entity_id = str(candidate.get("entity_id", "") or "").strip()
        entity_name = str(candidate.get("entity_name", "") or "").strip()

        if entity_id:
            key = f"id::{entity_id}"
        else:
            key = f"name::{normalize_node_key(entity_name)}"

        if not key:
            continue

        old = best_by_key.get(key)

        if old is None:
            best_by_key[key] = candidate
            continue

        old_score = float(old.get("score", 0.0))
        new_score = float(candidate.get("score", 0.0))

        if new_score > old_score:
            best_by_key[key] = candidate

    return list(best_by_key.values())


# =========================================================
# 8. 节点索引工具
# =========================================================

def build_node_name_index(
    nodes: Iterable[Any],
    *,
    node_attrs_getter: Optional[Any] = None,
) -> Dict[str, str]:
    """
    构建节点名称索引。

    Args:
        nodes:
            节点 key 列表。

        node_attrs_getter:
            可选函数：
                node_attrs_getter(node) -> dict

    Returns:
        {
            normalized_name: node_key
        }
    """

    index: Dict[str, str] = {}

    for node in nodes:
        node_key = str(node)
        normalized_key = normalize_node_key(node_key)

        if normalized_key:
            index.setdefault(normalized_key, node_key)

        attrs: Dict[str, Any] = {}

        if node_attrs_getter is not None:
            try:
                attrs = node_attrs_getter(node) or {}
            except Exception:
                attrs = {}

        for attr_key in [
            "id",
            "entity_id",
            "name",
            "entity_name",
            "label",
            "title",
            "text",
        ]:
            value = attrs.get(attr_key)

            if value is None:
                continue

            normalized_value = normalize_node_key(value)

            if normalized_value:
                index.setdefault(normalized_value, node_key)

        aliases = attrs.get("aliases") or attrs.get("alias")

        if isinstance(aliases, list):
            for alias in aliases:
                normalized_alias = normalize_node_key(alias)

                if normalized_alias:
                    index.setdefault(normalized_alias, node_key)

        elif isinstance(aliases, str):
            normalized_alias = normalize_node_key(aliases)

            if normalized_alias:
                index.setdefault(normalized_alias, node_key)

    return index


def find_node_key(
    *,
    entity_name: str,
    entity_id: str = "",
    aliases: Optional[List[str]] = None,
    node_index: Dict[str, str],
) -> Tuple[str, str]:
    """
    根据 entity_name / entity_id / aliases 在 node_index 中找 node_key。

    Returns:
        (node_key, match_method)
    """

    aliases = aliases or []

    search_items = []

    if entity_id:
        search_items.append(("entity_id", entity_id))

    if entity_name:
        search_items.append(("entity_name", entity_name))

    for alias in aliases:
        if alias:
            search_items.append(("alias", alias))

    for method, value in search_items:
        normalized = normalize_node_key(value)

        if normalized and normalized in node_index:
            return node_index[normalized], method

    return "", "not_found"


# =========================================================
# 9. 分数与字段工具
# =========================================================

def extract_score(
    item: Dict[str, Any],
    *,
    default: float = 0.0,
) -> float:
    """
    从候选对象中抽取分数。

    支持：
        - score
        - similarity
        - confidence
        - distance
    """

    for key in ["score", "similarity", "confidence"]:
        if key in item:
            return clip_score(safe_float(item.get(key), default=default))

    if "distance" in item:
        distance = safe_float(item.get("distance"), default=1.0)
        return clip_score(1.0 - distance)

    return clip_score(default)


def clip_score(score: float) -> float:
    """
    将分数限制在 [0, 1]。
    """

    if score < 0:
        return 0.0

    if score > 1:
        return 1.0

    return score


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
    从 dict 中按顺序获取第一个非空字符串字段。
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
# 10. 快速测试入口
# =========================================================

if __name__ == "__main__":
    normalizer = EntityNormalizer()

    tests = [
        "  美国  ",
        "United_States",
        "Beyoncé",
        "《三体》",
        "这个",
        "Barack Obama",
    ]

    for item in tests:
        print("=" * 60)
        print("raw:", item)
        print("mention:", normalize_mention(item))
        print("entity:", normalize_entity_name(item))
        print("node_key:", normalize_node_key(item))
        print("clean:", clean_mention_text(item))
        print("valid:", is_valid_entity_text(item))
        print("resolved:", normalizer.resolve(item))