# -*- coding: utf-8 -*-
"""基于已筛选 Evidence 形成结构化中间判断。"""

from __future__ import annotations

import json
import math
import re
from typing import Any, Dict, List, Tuple

from .schemas import (
    AnswerabilityType,
    EvidenceItem,
    ReasoningOptions,
    ReasoningOutput,
    ReasoningResult,
)


class AnswerReasoner:
    """统一推理器；保留原有规则推理并支持可注入 LLM。"""

    def __init__(self, llm_client: Any | None = None) -> None:
        self.llm_client = llm_client

    def reason(
        self,
        *,
        query: str,
        evidence: List[EvidenceItem],
        evidence_text: str = "",
        answerability: AnswerabilityType = "uncertain",
        semantic_score: float = 0.0,
        options: ReasoningOptions | None = None,
    ) -> ReasoningOutput:
        opts = (options or ReasoningOptions()).normalized()
        items = [EvidenceItem(**dict(item)) for item in (evidence or []) if isinstance(item, dict)]
        normalized_answerability = _normalize_answerability(answerability)
        score = _clip_score(_safe_float(semantic_score, default=0.0))

        if not items:
            result = ReasoningResult(
                reasoning_chain=["当前可用信息不足，无法形成可靠判断。"],
                conclusion="当前信息不足，不能可靠回答。",
                used_evidence_ids=[],
                confidence=0.0,
                metadata={"reasoning_type": "none"},
            )
            return ReasoningOutput(
                result=result,
                reasoning_text=_build_reasoning_text(result),
                reasoning_type="none",
            )

        if opts.use_llm and self.llm_client is not None:
            result, reasoning_type = self._reason_with_llm(
                query=str(query or "").strip(),
                evidence=items,
                evidence_text=evidence_text,
                answerability=normalized_answerability,
                semantic_score=score,
                max_reasoning_steps=opts.max_reasoning_steps,
            )
        else:
            result, reasoning_type = _reason_with_rules(
                query=str(query or "").strip(),
                evidence=items,
                answerability=normalized_answerability,
                semantic_score=score,
                max_reasoning_steps=opts.max_reasoning_steps,
            )

        return ReasoningOutput(
            result=result,
            reasoning_text=_build_reasoning_text(result),
            reasoning_type=reasoning_type,
        )

    def _reason_with_llm(
        self,
        *,
        query: str,
        evidence: List[EvidenceItem],
        evidence_text: str,
        answerability: AnswerabilityType,
        semantic_score: float,
        max_reasoning_steps: int,
    ) -> Tuple[ReasoningResult, str]:
        try:
            prompt = _build_reasoning_prompt(
                query=query,
                evidence=evidence,
                evidence_text=evidence_text,
                answerability=answerability,
                semantic_score=semantic_score,
                max_reasoning_steps=max_reasoning_steps,
            )
            result = _parse_llm_reasoning_output(
                raw_output=_call_llm(self.llm_client, prompt),
                evidence=evidence,
                answerability=answerability,
                semantic_score=semantic_score,
                max_reasoning_steps=max_reasoning_steps,
            )
            if not result.get("reasoning_chain") and not result.get("conclusion"):
                raise ValueError("LLM returned empty reasoning.")
            return result, "llm"
        except Exception:
            return _reason_with_rules(
                query=query,
                evidence=evidence,
                answerability=answerability,
                semantic_score=semantic_score,
                max_reasoning_steps=max_reasoning_steps,
            )


def _reason_with_rules(
    *,
    query: str,
    evidence: List[EvidenceItem],
    answerability: AnswerabilityType,
    semantic_score: float,
    max_reasoning_steps: int,
) -> Tuple[ReasoningResult, str]:
    """
    使用规则生成结构化中间判断。

    逻辑：
        1. 优先使用分数高、类型强的材料。
        2. 将材料转换为简短判断步骤。
        3. 根据材料类型和可回答性形成结论。
    """

    selected_evidence = _select_evidence_for_reasoning(
        evidence=evidence,
        max_reasoning_steps=max_reasoning_steps,
    )

    reasoning_chain: List[str] = []

    for item in selected_evidence:
        step = _evidence_to_reasoning_step(item)

        if step:
            reasoning_chain.append(step)

    if not reasoning_chain:
        reasoning_chain.append(
            "已有信息与问题存在一定关联，但未能形成明确的判断步骤。"
        )

    conclusion = _build_rule_conclusion(
        query=query,
        evidence=selected_evidence,
        answerability=answerability,
    )

    confidence = _estimate_reasoning_confidence(
        evidence=selected_evidence,
        answerability=answerability,
        semantic_score=semantic_score,
    )

    used_evidence_ids = [
        str(item.get("evidence_id", ""))
        for item in selected_evidence
        if item.get("evidence_id")
    ]

    reasoning_result = ReasoningResult(
        reasoning_chain=reasoning_chain[:max_reasoning_steps],
        conclusion=conclusion,
        used_evidence_ids=used_evidence_ids,
        confidence=confidence,
        metadata={
            "reasoning_type": "rule",
            "answerability": answerability,
            "semantic_score": semantic_score,
            "num_evidence": len(evidence),
        },
    )

    return reasoning_result, "rule"


def _select_evidence_for_reasoning(
    *,
    evidence: List[EvidenceItem],
    max_reasoning_steps: int,
) -> List[EvidenceItem]:
    """
    选择用于中间推理的材料。
    """

    if not evidence:
        return []

    sorted_evidence = sorted(
        evidence,
        key=lambda item: (
            _evidence_type_priority(str(item.get("evidence_type", ""))),
            _safe_float(item.get("score", 0.0), default=0.0),
        ),
        reverse=True,
    )

    return sorted_evidence[:max_reasoning_steps]


def _evidence_type_priority(evidence_type: str) -> int:
    """
    材料类型优先级。

    直接关系 > 路径 > 邻近关联 > 其他文本。
    """

    evidence_type = evidence_type.lower()

    if evidence_type == "relation":
        return 4

    if evidence_type == "path":
        return 3

    if evidence_type == "neighbor":
        return 2

    if evidence_type == "subgraph":
        return 1

    return 0


def _evidence_to_reasoning_step(item: EvidenceItem) -> str:
    """
    将单条材料转成推理步骤。

    注意：
        这里用自然语言描述，不暴露内部技术名词。
    """

    evidence_type = str(item.get("evidence_type", "")).lower()
    source = str(item.get("source_entity", "") or "").strip()
    target = str(item.get("target_entity", "") or "").strip()
    relation = str(item.get("relation", "") or "").strip()
    text = str(item.get("text", "") or "").strip()

    if evidence_type == "relation":
        if source and target and relation:
            return _sanitize_text(
                f"已有信息显示，{source} 与 {target} 之间存在“{relation}”关系。"
            )

        if text:
            return _sanitize_text(f"已有信息显示：{text}")

    if evidence_type == "path":
        path = item.get("path", []) or []

        if isinstance(path, list) and path:
            path_text = " → ".join(str(node) for node in path)
            return _sanitize_text(
                f"已有信息显示，相关对象之间存在间接联系：{path_text}。"
            )

        if text:
            return _sanitize_text(
                f"已有信息显示，相关对象之间存在间接联系：{text}"
            )

    if evidence_type == "neighbor":
        if source and target:
            return _sanitize_text(
                f"已有信息显示，{source} 与 {target} 存在相关联系。"
            )

        if text:
            return _sanitize_text(f"已有信息显示：{text}")

    if evidence_type == "subgraph":
        if text:
            return _sanitize_text(
                f"已有信息显示，这些对象周围存在相关联系：{text}"
            )

    if text:
        return _sanitize_text(f"已有信息显示：{text}")

    return ""


def _build_rule_conclusion(
    *,
    query: str,
    evidence: List[EvidenceItem],
    answerability: AnswerabilityType,
) -> str:
    """
    基于规则生成中间结论。

    注意：
        这里不是最终回答，只是供 generation_node 使用的中间结论。
    """

    if not evidence:
        return "当前信息不足，不能可靠回答该问题。"

    direct_relations = [
        item for item in evidence
        if str(item.get("evidence_type", "")).lower() == "relation"
    ]

    paths = [
        item for item in evidence
        if str(item.get("evidence_type", "")).lower() == "path"
    ]

    neighbors = [
        item for item in evidence
        if str(item.get("evidence_type", "")).lower() == "neighbor"
    ]

    if answerability == "answerable":
        if direct_relations:
            return "可以根据已有的直接关系信息回答该问题。"

        if paths:
            return "可以根据已有的间接联系信息回答该问题，但需要说明中间联系。"

        if neighbors:
            return "可以根据已有的相关信息回答该问题，但应避免过度推断。"

        return "已有信息可以为回答该问题提供依据。"

    if answerability == "uncertain":
        if direct_relations or paths:
            return "已有信息能够提供一定依据，但回答时需要保留不确定性。"

        return "已有信息与问题相关，但不足以支持强结论。"

    return "当前信息不足，不能可靠回答该问题。"


def _estimate_reasoning_confidence(
    *,
    evidence: List[EvidenceItem],
    answerability: AnswerabilityType,
    semantic_score: float,
) -> float:
    """
    估计 reasoning confidence。
    """

    if not evidence:
        return 0.0

    evidence_scores = [
        _safe_float(item.get("score", 0.0), default=0.0)
        for item in evidence
    ]

    avg_evidence_score = sum(evidence_scores) / max(len(evidence_scores), 1)

    type_bonus = 0.0

    if any(str(item.get("evidence_type", "")).lower() == "relation" for item in evidence):
        type_bonus += 0.12

    if any(str(item.get("evidence_type", "")).lower() == "path" for item in evidence):
        type_bonus += 0.06

    answerability_bonus = {
        "answerable": 0.12,
        "uncertain": 0.02,
        "unanswerable": -0.20,
    }.get(answerability, 0.0)

    confidence = (
        0.45 * avg_evidence_score
        + 0.40 * _clip_score(semantic_score)
        + type_bonus
        + answerability_bonus
    )

    return _clip_score(confidence)


def _build_reasoning_prompt(
    *,
    query: str,
    evidence: List[EvidenceItem],
    evidence_text: str,
    answerability: AnswerabilityType,
    semantic_score: float,
    max_reasoning_steps: int,
) -> str:
    """
    构造 reasoning prompt。

    注意：
        这是内部 prompt，用户不可见。
        后续补齐 prompt_manager.py 后，可迁移到 prompts/reasoning_prompt.txt。
    """

    compact_evidence = _compact_evidence_for_prompt(evidence)

    compact_evidence_json = json.dumps(
        compact_evidence,
        ensure_ascii=False,
        indent=2,
    )

    return f"""
你是一个结构化事实判断助手。

任务：
根据用户问题和已有材料，形成中间判断，用于后续生成最终回答。

要求：
1. 只输出 JSON。
2. 不要输出 Markdown。
3. 不要编造材料中没有的信息。
4. reasoning_chain 最多 {max_reasoning_steps} 条。
5. conclusion 是中间结论，不是面向用户的最终回答。
6. 如果材料不足，要明确说明不能可靠判断。
7. 不要出现“知识图谱”“KG”“KG-RAG”“实体链接”“检索流程”“节点”等技术词。

用户问题：
{query}

可回答性判断：
{answerability}

匹配分数：
{semantic_score:.3f}

已有材料文本：
{evidence_text}

已有材料 JSON：
{compact_evidence_json}

输出格式：
{{
  "reasoning_chain": [
    "第一条中间判断",
    "第二条中间判断"
  ],
  "conclusion": "一句中间结论",
  "used_evidence_ids": ["ev_xxx"],
  "confidence": 0.8
}}
""".strip()


def _compact_evidence_for_prompt(
    evidence: List[EvidenceItem],
) -> List[Dict[str, Any]]:
    """
    压缩材料，避免 prompt 过长。
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
            max_tokens=1200,
        )

        return _extract_llm_text(result)

    if hasattr(llm, "generate"):
        result = llm.generate(
            prompt=prompt,
            temperature=0.0,
            max_tokens=1200,
        )

        return _extract_llm_text(result)

    if hasattr(llm, "chat"):
        try:
            result = llm.chat(
                messages=[
                    {
                        "role": "system",
                        "content": "你是一个结构化事实判断助手，只输出 JSON。",
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=0.0,
                max_tokens=1200,
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


def _parse_llm_reasoning_output(
    *,
    raw_output: str,
    evidence: List[EvidenceItem],
    answerability: AnswerabilityType,
    semantic_score: float,
    max_reasoning_steps: int,
) -> ReasoningResult:
    """
    解析 LLM reasoning 输出。
    """

    json_text = _extract_json_object(raw_output)

    if not json_text:
        return ReasoningResult(
            reasoning_chain=[],
            conclusion="",
            used_evidence_ids=[],
            confidence=0.0,
            metadata={
                "parse_failed": True,
                "raw_output_preview": _preview_text(raw_output),
            },
        )

    try:
        data = json.loads(json_text)
    except Exception:
        return ReasoningResult(
            reasoning_chain=[],
            conclusion="",
            used_evidence_ids=[],
            confidence=0.0,
            metadata={
                "parse_failed": True,
                "raw_output_preview": _preview_text(raw_output),
            },
        )

    if not isinstance(data, dict):
        data = {}

    reasoning_chain = data.get("reasoning_chain", [])

    if not isinstance(reasoning_chain, list):
        reasoning_chain = []

    reasoning_chain = [
        _sanitize_text(str(step).strip())
        for step in reasoning_chain
        if str(step).strip()
    ][:max_reasoning_steps]

    conclusion = _sanitize_text(
        str(data.get("conclusion", "")).strip()
    )

    used_evidence_ids = data.get("used_evidence_ids", [])

    if not isinstance(used_evidence_ids, list):
        used_evidence_ids = []

    valid_evidence_ids = {
        str(item.get("evidence_id", ""))
        for item in evidence
        if item.get("evidence_id")
    }

    used_evidence_ids = [
        str(eid)
        for eid in used_evidence_ids
        if str(eid) in valid_evidence_ids
    ]

    if not used_evidence_ids:
        used_evidence_ids = [
            str(item.get("evidence_id", ""))
            for item in evidence[:max_reasoning_steps]
            if item.get("evidence_id")
        ]

    confidence = _clip_score(
        _safe_float(
            data.get("confidence", semantic_score),
            default=semantic_score,
        )
    )

    if not conclusion:
        conclusion = _fallback_conclusion_by_answerability(answerability)

    return ReasoningResult(
        reasoning_chain=reasoning_chain,
        conclusion=conclusion,
        used_evidence_ids=used_evidence_ids,
        confidence=confidence,
        metadata={
            "reasoning_type": "llm",
            "answerability": answerability,
            "semantic_score": semantic_score,
        },
    )


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


def _normalize_reasoning_result(
    *,
    reasoning_result: ReasoningResult,
    evidence: List[EvidenceItem],
    answerability: AnswerabilityType,
    semantic_score: float,
    max_reasoning_steps: int,
    fallback_type: str,
) -> ReasoningResult:
    """
    规范化 ReasoningResult，避免字段缺失导致 generation_node 使用困难。
    """

    reasoning_chain = reasoning_result.get("reasoning_chain", []) or []

    if not isinstance(reasoning_chain, list):
        reasoning_chain = []

    reasoning_chain = [
        _sanitize_text(str(step).strip())
        for step in reasoning_chain
        if str(step).strip()
    ][:max_reasoning_steps]

    conclusion = _sanitize_text(
        str(reasoning_result.get("conclusion", "") or "").strip()
    )

    if not conclusion:
        conclusion = _fallback_conclusion_by_answerability(answerability)

    used_evidence_ids = reasoning_result.get("used_evidence_ids", []) or []

    if not isinstance(used_evidence_ids, list):
        used_evidence_ids = []

    valid_evidence_ids = {
        str(item.get("evidence_id", ""))
        for item in evidence
        if item.get("evidence_id")
    }

    used_evidence_ids = [
        str(eid)
        for eid in used_evidence_ids
        if str(eid) in valid_evidence_ids
    ]

    if not used_evidence_ids:
        used_evidence_ids = [
            str(item.get("evidence_id", ""))
            for item in evidence[:max_reasoning_steps]
            if item.get("evidence_id")
        ]

    confidence = _clip_score(
        _safe_float(
            reasoning_result.get("confidence", semantic_score),
            default=semantic_score,
        )
    )

    metadata = dict(reasoning_result.get("metadata", {}) or {})
    metadata.setdefault("reasoning_type", fallback_type)
    metadata.setdefault("answerability", answerability)
    metadata.setdefault("semantic_score", semantic_score)

    return ReasoningResult(
        reasoning_chain=reasoning_chain,
        conclusion=conclusion,
        used_evidence_ids=used_evidence_ids,
        confidence=confidence,
        metadata=metadata,
    )


def _build_reasoning_text(reasoning_result: ReasoningResult) -> str:
    """
    将 ReasoningResult 转为文本。

    供 generation_node 使用。
    """

    reasoning_chain = reasoning_result.get("reasoning_chain", []) or []
    conclusion = str(reasoning_result.get("conclusion", "") or "").strip()
    confidence = _safe_float(
        reasoning_result.get("confidence", 0.0),
        default=0.0,
    )

    lines: List[str] = []

    if reasoning_chain:
        lines.append("中间判断：")

        for idx, step in enumerate(reasoning_chain, start=1):
            step = _sanitize_text(str(step).strip())

            if step:
                lines.append(f"{idx}. {step}")

    if conclusion:
        lines.append(f"结论倾向：{_sanitize_text(conclusion)}")

    lines.append(f"置信度：{confidence:.3f}")

    return "\n".join(lines)


def _fallback_conclusion_by_answerability(answerability: AnswerabilityType) -> str:
    """
    根据 answerability 生成兜底中间结论。
    """

    if answerability == "answerable":
        return "已有信息可以支撑回答该问题。"

    if answerability == "uncertain":
        return "已有信息能够提供一定参考，但回答需要谨慎。"

    return "当前信息不足，不能可靠回答该问题。"


def _sanitize_text(text: str) -> str:
    """
    清理可能暴露内部实现的文本。
    """

    if not text:
        return ""

    replacements = {
        "知识图谱": "已有信息",
        "KG-RAG": "已有信息",
        "KGRAG": "已有信息",
        "kg_rag": "已有信息",
        "KG": "已有信息",
        "knowledge graph": "已有信息",
        "实体链接": "对象识别",
        "图谱检索": "信息查找",
        "检索流程": "处理流程",
        "节点": "步骤",
    }

    cleaned = text

    for old, new in replacements.items():
        cleaned = re.sub(
            re.escape(old),
            new,
            cleaned,
            flags=re.IGNORECASE,
        )

    return cleaned.strip()


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


def _preview_text(text: str, max_len: int = 300) -> str:
    """
    截断长文本，避免 metadata 记录过长原始输出。
    """

    text = str(text or "")

    if len(text) <= max_len:
        return text

    return text[:max_len] + "..."
