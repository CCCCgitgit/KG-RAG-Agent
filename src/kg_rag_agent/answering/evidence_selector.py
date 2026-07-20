# -*- coding: utf-8 -*-
"""Evidence 评分、筛选与可回答性判断。"""

from __future__ import annotations

import json
import math
import re
from typing import Any, Dict, List, Tuple

from .schemas import (
    AnswerabilityType,
    EvidenceItem,
    EvidenceSelection,
    EvidenceSelectionOptions,
    SemanticScoringResult,
)


class EvidenceSelector:
    """统一 Evidence 选择器；LLM 失败时自动回退到原有规则评分。"""

    def __init__(self, llm_client: Any | None = None) -> None:
        self.llm_client = llm_client

    def select(
        self,
        *,
        query: str,
        evidence: List[EvidenceItem],
        options: EvidenceSelectionOptions | None = None,
    ) -> EvidenceSelection:
        opts = (options or EvidenceSelectionOptions()).normalized()
        items = [EvidenceItem(**dict(item)) for item in (evidence or []) if isinstance(item, dict)]

        if not items:
            result = SemanticScoringResult(
                score=0.0,
                answerability="unanswerable",
                reason="当前可用信息不足，无法可靠回答。",
                selected_evidence_ids=[],
                rejected_evidence_ids=[],
            )
            return EvidenceSelection(result=result, evidence=[], evidence_text="", scoring_type="none")

        if opts.use_llm and self.llm_client is not None:
            result, selected, scoring_type = self._score_with_llm(
                query=str(query or "").strip(),
                evidence=items,
                options=opts,
            )
        else:
            result, selected, scoring_type = _score_with_rules(
                query=str(query or "").strip(),
                evidence=items,
                max_selected_evidence=opts.max_selected_evidence,
                min_evidence_score=opts.min_evidence_score,
                answerable_threshold=opts.answerable_threshold,
                uncertain_threshold=opts.uncertain_threshold,
            )

        return EvidenceSelection(
            result=result,
            evidence=selected,
            evidence_text=_build_evidence_text(selected),
            scoring_type=scoring_type,
        )

    def _score_with_llm(
        self,
        *,
        query: str,
        evidence: List[EvidenceItem],
        options: EvidenceSelectionOptions,
    ) -> Tuple[SemanticScoringResult, List[EvidenceItem], str]:
        try:
            prompt = _build_scoring_prompt(
                query=query,
                compact_evidence=_compact_evidence_for_prompt(evidence),
                max_selected_evidence=options.max_selected_evidence,
            )
            decision = _parse_llm_scoring_output(_call_llm(self.llm_client, prompt))
            selected_ids = decision.get("selected_evidence_ids", [])
            rejected_ids = decision.get("rejected_evidence_ids", [])
            if not isinstance(selected_ids, list):
                selected_ids = []
            if not isinstance(rejected_ids, list):
                rejected_ids = []

            by_id = {
                str(item.get("evidence_id", "")): item
                for item in evidence
                if item.get("evidence_id")
            }
            selected = [
                by_id[str(eid)] for eid in selected_ids if str(eid) in by_id
            ][: options.max_selected_evidence]
            if not selected:
                raise ValueError("LLM did not select valid evidence.")

            score = _clip_score(_safe_float(decision.get("score", 0.0), default=0.0))
            answerability = _normalize_answerability(decision.get("answerability", "uncertain"))
            reason = _sanitize_reason(str(decision.get("reason", "") or "").strip())
            if not reason:
                reason = _build_scoring_reason(
                    answerability=answerability,
                    score=score,
                    selected_evidence=selected,
                )
            selected = _attach_llm_score_to_evidence(
                selected=selected,
                score_map=decision.get("evidence_scores", {}),
            )
            selected_id_set = {
                str(item.get("evidence_id", "")) for item in selected if item.get("evidence_id")
            }
            if not rejected_ids:
                rejected_ids = [
                    str(item.get("evidence_id", ""))
                    for item in evidence
                    if item.get("evidence_id") and str(item.get("evidence_id", "")) not in selected_id_set
                ]
            result = SemanticScoringResult(
                score=score,
                answerability=answerability,
                reason=reason,
                selected_evidence_ids=[
                    str(item.get("evidence_id", "")) for item in selected if item.get("evidence_id")
                ],
                rejected_evidence_ids=[str(eid) for eid in rejected_ids if eid],
            )
            return result, selected, "llm"
        except Exception:
            return _score_with_rules(
                query=query,
                evidence=evidence,
                max_selected_evidence=options.max_selected_evidence,
                min_evidence_score=options.min_evidence_score,
                answerable_threshold=options.answerable_threshold,
                uncertain_threshold=options.uncertain_threshold,
            )


def _score_with_rules(
    *,
    query: str,
    evidence: List[EvidenceItem],
    max_selected_evidence: int,
    min_evidence_score: float,
    answerable_threshold: float,
    uncertain_threshold: float,
) -> Tuple[SemanticScoringResult, List[EvidenceItem], str]:
    """
    使用规则进行 evidence 评分。

    评分维度：
        1. evidence 原始分数
        2. evidence 类型优先级
        3. query 与 evidence 文本重叠
        4. query 中对象与 evidence 中对象重叠
        5. relation / path / neighbor 的直接性
    """

    scored_items: List[Tuple[EvidenceItem, float]] = []

    for item in evidence:
        score = _score_single_evidence(
            query=query,
            evidence=item,
        )

        updated_item = dict(item)
        updated_item["score"] = _clip_score(score)

        metadata = dict(updated_item.get("metadata", {}) or {})
        metadata["semantic_score"] = updated_item["score"]
        metadata["scored_by"] = "rule"
        updated_item["metadata"] = metadata

        scored_items.append(
            (
                EvidenceItem(**updated_item),
                updated_item["score"],
            )
        )

    scored_items = sorted(
        scored_items,
        key=lambda pair: pair[1],
        reverse=True,
    )

    selected: List[EvidenceItem] = [
        item
        for item, score in scored_items
        if score >= min_evidence_score
    ][:max_selected_evidence]

    selected_ids = [
        str(item.get("evidence_id", ""))
        for item in selected
        if item.get("evidence_id")
    ]

    selected_id_set = set(selected_ids)

    rejected_ids = [
        str(item.get("evidence_id", ""))
        for item, _ in scored_items
        if item.get("evidence_id") and str(item.get("evidence_id", "")) not in selected_id_set
    ]

    selected_scores = [
        _safe_float(item.get("score", 0.0), default=0.0)
        for item in selected
    ]

    if selected_scores:
        global_score = _aggregate_scores(selected_scores)
    else:
        global_score = 0.0

    answerability = _decide_answerability(
        score=global_score,
        selected_evidence=selected,
        answerable_threshold=answerable_threshold,
        uncertain_threshold=uncertain_threshold,
    )

    reason = _build_scoring_reason(
        answerability=answerability,
        score=global_score,
        selected_evidence=selected,
    )

    scoring_result = SemanticScoringResult(
        score=global_score,
        answerability=answerability,
        reason=reason,
        selected_evidence_ids=selected_ids,
        rejected_evidence_ids=rejected_ids,
    )

    return scoring_result, selected, "rule"


def _score_single_evidence(
    *,
    query: str,
    evidence: EvidenceItem,
) -> float:
    """
    对单条 evidence 进行规则评分。
    """

    base_score = _safe_float(
        evidence.get("score", 0.0),
        default=0.0,
    )

    evidence_type = str(evidence.get("evidence_type", "")).strip().lower()
    text = str(evidence.get("text", "")).strip()

    type_score = _evidence_type_score(evidence_type)
    text_overlap_score = _text_overlap_score(query, text)
    entity_overlap_score = _entity_overlap_score(query, evidence)
    relation_match_score = _relation_match_score(query, evidence)

    final_score = (
        0.35 * base_score
        + 0.20 * type_score
        + 0.25 * text_overlap_score
        + 0.15 * entity_overlap_score
        + 0.05 * relation_match_score
    )

    return _clip_score(final_score)


def _evidence_type_score(evidence_type: str) -> float:
    """
    evidence 类型分数。

    直接关系通常比路径和邻居更强。
    """

    if evidence_type == "relation":
        return 1.0

    if evidence_type == "path":
        return 0.78

    if evidence_type == "neighbor":
        return 0.58

    if evidence_type == "subgraph":
        return 0.50

    if evidence_type == "text":
        return 0.45

    return 0.40


def _text_overlap_score(query: str, evidence_text: str) -> float:
    """
    query 与 evidence text 的词面重叠分数。
    """

    query_tokens = _simple_tokens(query)
    evidence_tokens = _simple_tokens(evidence_text)

    if not query_tokens or not evidence_tokens:
        return 0.0

    intersection = query_tokens & evidence_tokens
    union = query_tokens | evidence_tokens

    jaccard = len(intersection) / max(len(union), 1)
    recall = len(intersection) / max(len(query_tokens), 1)

    return _clip_score(0.4 * jaccard + 0.6 * recall)


def _entity_overlap_score(
    query: str,
    evidence: EvidenceItem,
) -> float:
    """
    判断 query 是否命中 evidence 中的 source / target / path。
    """

    query_norm = _normalize_text(query)

    candidates: List[str] = []

    for key in ["source_entity", "target_entity", "relation"]:
        value = evidence.get(key)
        if value:
            candidates.append(str(value))

    path = evidence.get("path", []) or []
    if isinstance(path, list):
        candidates.extend(str(x) for x in path)

    triples = evidence.get("triples", []) or []
    if isinstance(triples, list):
        for triple in triples:
            if not isinstance(triple, dict):
                continue

            for key in [
                "head",
                "source",
                "subject",
                "relation",
                "predicate",
                "tail",
                "target",
                "object",
            ]:
                value = triple.get(key)
                if value:
                    candidates.append(str(value))

    hits = 0
    total = 0

    for candidate in candidates:
        candidate_norm = _normalize_text(candidate)

        if not candidate_norm:
            continue

        total += 1

        if candidate_norm in query_norm or query_norm in candidate_norm:
            hits += 1

    if total == 0:
        return 0.0

    return _clip_score(hits / total)


def _relation_match_score(
    query: str,
    evidence: EvidenceItem,
) -> float:
    """
    判断 evidence relation 是否与用户问题中的关系意图接近。

    这里只做轻量关键词判断，不替代 LLM 推理。
    """

    query_lower = query.lower()
    relation = str(evidence.get("relation", "")).lower()
    evidence_type = str(evidence.get("evidence_type", "")).lower()

    if not relation and not evidence_type:
        return 0.0

    relation_intent_keywords = [
        "关系",
        "联系",
        "相关",
        "连接",
        "between",
        "relation",
        "relationship",
        "related",
        "connect",
    ]

    path_intent_keywords = [
        "路径",
        "怎么到",
        "如何连接",
        "path",
        "route",
    ]

    belong_intent_keywords = [
        "属于",
        "类别",
        "类型",
        "class",
        "type",
        "belong",
    ]

    if any(keyword in query_lower for keyword in relation_intent_keywords):
        if evidence_type == "relation":
            return 1.0
        if relation in {"related_to", "relation"}:
            return 0.8

    if any(keyword in query_lower for keyword in path_intent_keywords):
        if evidence_type == "path":
            return 1.0

    if any(keyword in query_lower for keyword in belong_intent_keywords):
        if relation in {"type", "class", "instance_of", "subclass_of", "belongs_to", "属于"}:
            return 1.0

    if relation and relation in query_lower:
        return 1.0

    return 0.4


def _aggregate_scores(scores: List[float]) -> float:
    """
    聚合 evidence 分数为整体语义分。

    采用 top-k 加权：
        top1 权重最大，其余逐步衰减。
    """

    if not scores:
        return 0.0

    scores = sorted(
        [_clip_score(score) for score in scores],
        reverse=True,
    )

    top_scores = scores[:5]

    weights = [
        1.0 / (idx + 1)
        for idx in range(len(top_scores))
    ]

    weighted_sum = sum(score * weight for score, weight in zip(top_scores, weights))
    weight_sum = sum(weights)

    return _clip_score(weighted_sum / max(weight_sum, 1e-8))


def _decide_answerability(
    *,
    score: float,
    selected_evidence: List[EvidenceItem],
    answerable_threshold: float,
    uncertain_threshold: float,
) -> AnswerabilityType:
    """
    根据整体分数和选中 evidence 判断可回答性。
    """

    if not selected_evidence:
        return "unanswerable"

    has_relation = any(
        str(item.get("evidence_type", "")).lower() == "relation"
        for item in selected_evidence
    )

    has_path = any(
        str(item.get("evidence_type", "")).lower() == "path"
        for item in selected_evidence
    )

    if score >= answerable_threshold:
        return "answerable"

    if score >= uncertain_threshold and (has_relation or has_path):
        return "uncertain"

    if score >= uncertain_threshold:
        return "uncertain"

    return "unanswerable"


def _build_scoring_reason(
    *,
    answerability: AnswerabilityType,
    score: float,
    selected_evidence: List[EvidenceItem],
) -> str:
    """
    构造自然语言评分原因。

    注意：
        这里不暴露“知识图谱”“KG-RAG”“实体链接”等内部技术细节。
    """

    if answerability == "answerable":
        return "已有信息与问题匹配度较高，可以据此回答。"

    if answerability == "uncertain":
        return "已有信息与问题有一定关联，但需要谨慎表述。"

    return "当前可用信息不足，无法可靠回答。"


def _call_llm(llm: Any, prompt: str) -> str:
    """
    兼容不同 LLMClient 接口。

    优先级：
        1. generate_with_metadata()
        2. generate()
        3. chat()
    """

    if hasattr(llm, "generate_with_metadata"):
        result = llm.generate_with_metadata(
            prompt=prompt,
            temperature=0.0,
            max_tokens=1024,
        )

        return _extract_llm_text(result)

    if hasattr(llm, "generate"):
        result = llm.generate(
            prompt=prompt,
            temperature=0.0,
            max_tokens=1024,
        )

        return _extract_llm_text(result)

    if hasattr(llm, "chat"):
        try:
            result = llm.chat(
                messages=[
                    {
                        "role": "system",
                        "content": "你是一个证据相关性判断器，只输出 JSON。",
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=0.0,
                max_tokens=1024,
            )
        except TypeError:
            result = llm.chat(prompt)

        return _extract_llm_text(result)

    raise AttributeError("LLMClient has no supported generation method.")


def _extract_llm_text(result: Any) -> str:
    """
    从不同格式的 LLM 返回结果中提取文本。
    """

    if result is None:
        return ""

    if isinstance(result, str):
        return result.strip()

    if isinstance(result, dict):
        for key in ["content", "answer", "text", "output", "response"]:
            value = result.get(key)
            if value:
                return str(value).strip()

        message = result.get("message")
        if isinstance(message, dict) and message.get("content"):
            return str(message["content"]).strip()

        choices = result.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]

            if isinstance(first, dict):
                message = first.get("message")
                if isinstance(message, dict) and message.get("content"):
                    return str(message["content"]).strip()

                if first.get("text"):
                    return str(first["text"]).strip()

    if hasattr(result, "choices"):
        try:
            choice = result.choices[0]
            content = choice.message.content
            return str(content).strip()
        except Exception:
            pass

    if hasattr(result, "content"):
        try:
            return str(result.content).strip()
        except Exception:
            pass

    return str(result).strip()


def _build_scoring_prompt(
    *,
    query: str,
    compact_evidence: List[Dict[str, Any]],
    max_selected_evidence: int,
) -> str:
    """
    构造 LLM scoring prompt。

    注意：
        这是内部 prompt，用户不可见。
        后续补齐 prompt_manager.py 后，可迁移到 prompts/。
    """

    evidence_json = json.dumps(
        compact_evidence,
        ensure_ascii=False,
        indent=2,
    )

    return f"""
你是一个事实材料相关性判断器。

任务：
根据用户问题，从候选材料中选出最相关、最能支撑回答的材料，并判断这些材料是否足以回答问题。

只允许输出 JSON，不允许输出 Markdown，不允许输出解释性文字。

判断标准：
1. 材料是否直接涉及用户问题中的对象。
2. 材料是否直接涉及用户想问的关系、联系、类别、路径或事实。
3. 直接关系材料优先于间接材料。
4. 明显无关材料不要选择。
5. 最多选择 {max_selected_evidence} 条材料。

answerability 只能是：
- answerable：材料足够支撑回答
- uncertain：材料有一定相关性，但回答需要谨慎
- unanswerable：材料不足，不能可靠回答

注意：
reason 必须是自然语言说明，不要出现“知识图谱”“KG”“KG-RAG”“实体链接”“检索流程”等技术词。

用户问题：
{query}

候选材料：
{evidence_json}

输出格式：
{{
  "score": 0.0,
  "answerability": "answerable | uncertain | unanswerable",
  "reason": "一句简短中文原因",
  "selected_evidence_ids": ["ev_xxx"],
  "rejected_evidence_ids": ["ev_yyy"],
  "evidence_scores": {{
    "ev_xxx": 0.92
  }}
}}
""".strip()


def _compact_evidence_for_prompt(
    evidence: List[EvidenceItem],
) -> List[Dict[str, Any]]:
    """
    压缩 evidence，避免 prompt 过长。
    """

    compact: List[Dict[str, Any]] = []

    for item in evidence:
        compact.append(
            {
                "evidence_id": item.get("evidence_id", ""),
                "type": item.get("evidence_type", ""),
                "source": item.get("source_entity", ""),
                "relation": item.get("relation", ""),
                "target": item.get("target_entity", ""),
                "text": item.get("text", ""),
                "score": item.get("score", 0.0),
            }
        )

    return compact


def _parse_llm_scoring_output(raw_output: str) -> Dict[str, Any]:
    """
    解析 LLM scoring 输出。
    """

    json_text = _extract_json_object(raw_output)

    if not json_text:
        return {}

    try:
        data = json.loads(json_text)
    except Exception:
        return {}

    if not isinstance(data, dict):
        return {}

    return data


def _extract_json_object(text: str) -> str:
    """
    从文本中提取 JSON 对象。
    """

    if not text:
        return ""

    match = re.search(r"\{.*\}", text, re.DOTALL)

    if not match:
        return ""

    return match.group()


def _attach_llm_score_to_evidence(
    *,
    selected: List[EvidenceItem],
    score_map: Any,
) -> List[EvidenceItem]:
    """
    将 LLM 给出的 evidence_scores 写入 evidence metadata。
    """

    if not isinstance(score_map, dict):
        return selected

    updated: List[EvidenceItem] = []

    for item in selected:
        updated_item = dict(item)
        evidence_id = str(updated_item.get("evidence_id", ""))

        score = _safe_float(
            score_map.get(evidence_id, updated_item.get("score", 0.0)),
            default=_safe_float(updated_item.get("score", 0.0), default=0.0),
        )

        updated_item["score"] = _clip_score(score)

        metadata = dict(updated_item.get("metadata", {}) or {})
        metadata["semantic_score"] = updated_item["score"]
        metadata["scored_by"] = "llm"
        updated_item["metadata"] = metadata

        updated.append(EvidenceItem(**updated_item))

    return updated


def _build_evidence_text(
    evidence: List[EvidenceItem],
) -> str:
    """
    构造 evidence_text。

    供 reasoning_node 和 generation_node 使用。
    """

    if not evidence:
        return ""

    lines: List[str] = []

    for idx, item in enumerate(evidence, start=1):
        evidence_type = str(item.get("evidence_type", "unknown"))
        score = _safe_float(item.get("score", 0.0), default=0.0)
        text = str(item.get("text", "")).strip()

        if not text:
            continue

        lines.append(
            f"[E{idx}] ({evidence_type}, score={score:.3f}) {text}"
        )

    return "\n".join(lines)


def _simple_tokens(text: str) -> set[str]:
    """
    简单分词。

    中文按 2 字以上连续片段；
    英文按单词。
    """

    text = str(text or "").lower()

    chinese_tokens = set(
        re.findall(r"[\u4e00-\u9fff]{2,}", text)
    )
    english_tokens = set(
        re.findall(r"[a-zA-Z0-9_]{2,}", text)
    )

    stopwords = {
        "什么",
        "怎么",
        "如何",
        "哪些",
        "关系",
        "联系",
        "相关",
        "the",
        "and",
        "for",
        "with",
        "what",
        "who",
        "how",
        "why",
        "which",
    }

    return {
        token for token in chinese_tokens | english_tokens
        if token not in stopwords
    }


def _normalize_text(text: str) -> str:
    """
    文本归一化。
    """

    text = str(text or "").strip().lower()
    text = text.replace("_", " ")
    text = text.replace("-", " ")
    text = re.sub(r"\s+", " ", text)

    return text


def _sanitize_reason(reason: str) -> str:
    """
    清理可能暴露内部实现的 reason。
    """

    if not reason:
        return ""

    forbidden_terms = [
        "知识图谱",
        "KG-RAG",
        "KGRAG",
        "KG",
        "kg_rag",
        "knowledge graph",
        "实体链接",
        "图谱检索",
        "检索流程",
        "检索结果",
    ]

    lowered = reason.lower()

    for term in forbidden_terms:
        if term.lower() in lowered:
            return "已有信息与问题有一定关联，但需要谨慎表述。"

    return reason


def _normalize_answerability(value: Any) -> AnswerabilityType:
    """
    规范化 answerability。
    """

    value = str(value or "uncertain").strip().lower()

    if value == "answerable":
        return "answerable"

    if value == "unanswerable":
        return "unanswerable"

    return "uncertain"


def _safe_int(value: Any, *, default: int) -> int:
    """
    安全转 int。
    """

    try:
        if value is None:
            return default

        return int(value)
    except Exception:
        return default


def _safe_float(value: Any, *, default: float) -> float:
    """
    安全转 float。
    """

    try:
        if value is None:
            return default

        value = float(value)

        if math.isnan(value) or math.isinf(value):
            return default

        return value

    except Exception:
        return default


def _clip_score(score: float) -> float:
    """
    将 score 限制在 [0, 1]。
    """

    if score < 0:
        return 0.0

    if score > 1:
        return 1.0

    return score
