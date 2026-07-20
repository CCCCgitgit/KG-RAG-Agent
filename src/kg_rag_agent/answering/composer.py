# -*- coding: utf-8 -*-
"""最终答案组合、关系型直接回答、兜底与引用对齐。"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Tuple

from .citation_builder import CitationBuilder
from .schemas import (
    AnswerResult,
    AnswerabilityType,
    EvidenceItem,
    GenerationOptions,
    ReasoningResult,
)

AgentState = Dict[str, Any]


class AnswerComposer:
    """统一答案组合器；保留原项目的关系问答与安全兜底逻辑。"""

    def __init__(
        self,
        llm_client: Any | None = None,
        citation_builder: CitationBuilder | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.citation_builder = citation_builder or CitationBuilder()

    def compose(
        self,
        *,
        query: str,
        evidence: List[EvidenceItem] | None = None,
        reasoning: ReasoningResult | None = None,
        evidence_text: str = "",
        reasoning_text: str = "",
        answerability: AnswerabilityType = "uncertain",
        semantic_score: float = 0.0,
        options: GenerationOptions | None = None,
        clarifying_question: str = "",
        need_clarification: bool = False,
        has_error: bool = False,
        ungrounded_mentions: List[str] | None = None,
        scoring_reason: str = "",
        system_prompt: str = "",
    ) -> AnswerResult:
        opts = (options or GenerationOptions()).normalized()
        items = [EvidenceItem(**dict(item)) for item in (evidence or []) if isinstance(item, dict)]
        reasoning_result = ReasoningResult(**dict(reasoning or {}))
        normalized_answerability = _normalize_answerability(answerability)
        score = _clip_score(_safe_float(semantic_score, default=0.0))
        state: AgentState = {
            "clarifying_question": clarifying_question,
            "ungrounded_mentions": list(ungrounded_mentions or []),
            "scoring_reason": scoring_reason,
        }

        if need_clarification:
            answer = _build_clarification_answer(state)
            return self._result(answer, [], "clarification", normalized_answerability, score)

        if has_error:
            answer = _build_error_answer(state)
            return self._result(answer, [], "error_fallback", normalized_answerability, score)

        if normalized_answerability == "unanswerable" or not items:
            answer = _build_unanswerable_answer(query=query, state=state)
            return self._result(answer, [], "unanswerable", normalized_answerability, score)

        generation_type = "rule"
        if opts.use_llm and self.llm_client is not None:
            try:
                config = {
                    "generation": {
                        "temperature": opts.temperature,
                        "max_tokens": opts.max_tokens,
                    },
                    "prompt": {"generation_system_prompt": system_prompt},
                }
                prompt = _build_generation_prompt(
                    query=query,
                    evidence=items,
                    evidence_text=evidence_text,
                    reasoning=reasoning_result,
                    reasoning_text=reasoning_text,
                    answerability=normalized_answerability,
                    semantic_score=score,
                )
                raw = _call_llm(llm=self.llm_client, prompt=prompt, config=config)
                answer = _extract_answer(raw)
                if not answer:
                    raise ValueError("LLM returned empty answer.")
                generation_type = "llm"
            except Exception:
                answer, generation_type = _generate_with_rules(
                    query=query,
                    evidence=items,
                    reasoning=reasoning_result,
                    answerability=normalized_answerability,
                    semantic_score=score,
                )
        else:
            answer, generation_type = _generate_with_rules(
                query=query,
                evidence=items,
                reasoning=reasoning_result,
                answerability=normalized_answerability,
                semantic_score=score,
            )

        answer = _sanitize_user_answer(answer)
        if not answer:
            answer, generation_type = _generate_with_rules(
                query=query,
                evidence=items,
                reasoning=reasoning_result,
                answerability=normalized_answerability,
                semantic_score=score,
            )
            answer = _sanitize_user_answer(answer)

        citations = self.citation_builder.build(items) if opts.include_citations else []
        return self._result(answer, citations, generation_type, normalized_answerability, score)

    @staticmethod
    def _result(
        answer: str,
        citations: List[Dict[str, Any]],
        generation_type: str,
        answerability: AnswerabilityType,
        semantic_score: float,
    ) -> AnswerResult:
        return AnswerResult(
            answer=answer,
            citations=citations,
            generation_type=generation_type,
            answerability=answerability,
            semantic_score=semantic_score,
            metadata={
                "answer_length": len(answer),
                "num_citations": len(citations),
            },
        )


def _get_generation_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    获取 generation 配置。

    兼容两种来源：
        1. config["generation"]
        2. config["graph"]["generation"]

    如果两边都有，graph.generation 覆盖顶层 generation。
    """

    top_level_config = config.get("generation", {}) or {}
    graph_config = config.get("graph", {}) or {}
    graph_generation_config = graph_config.get("generation", {}) or {}

    merged = dict(top_level_config)
    merged.update(graph_generation_config)

    return merged


def _build_generation_prompt(
    *,
    query: str,
    evidence: List[EvidenceItem],
    evidence_text: str,
    reasoning: ReasoningResult,
    reasoning_text: str,
    answerability: str,
    semantic_score: float,
) -> str:
    """
    构造最终回答 prompt。

    注意：
        这是内部 prompt，用户不可见。
        但是生成内容必须保持自然，不暴露系统实现细节。
    """

    evidence_brief = _build_evidence_brief(evidence)
    confidence = _clip_score(
        _safe_float(
            reasoning.get("confidence", semantic_score),
            default=semantic_score,
        )
    )

    return f"""
请根据已有信息回答用户问题。

用户问题：
{query}

回答可靠性：
{answerability}

匹配分数：
{semantic_score:.3f}

推理置信度：
{confidence:.3f}

已有信息：
{evidence_text}

材料摘要：
{evidence_brief}

中间判断：
{reasoning_text}

写作要求：
1. 用自然、清晰的中文回答。
2. 优先直接给出结论，不要先说空泛模板句。
3. 如果问题是“谁是 A 的妻子/丈夫/职业/出生地”等明确关系问题，请直接回答对应对象。
4. 不要输出 Markdown 表格。
5. 不要编造已有信息中没有的内容。
6. 如果 answerability 是 uncertain，要用“从已有信息看”“可能”“需要谨慎理解”等表达。
7. 如果信息不足，要明确说明目前不能可靠判断。
8. 不要出现以下词语：
   - 知识图谱
   - KG
   - KG-RAG
   - 实体链接
   - 图谱检索
   - 检索节点
   - 路由节点
   - LangGraph
   - StateGraph
   - 内部流程
9. 不要说“我查询了某某系统”，只说“根据已有信息”。
10. 回答要直接面向用户，不要解释系统是如何工作的。

请输出最终回答：
""".strip()


def _call_llm(
    *,
    llm: Any,
    prompt: str,
    config: Dict[str, Any],
) -> Any:
    """
    兼容不同 LLMClient 接口。

    注意：
        本节点只能通过 llm/llm_client.py 的统一接口调用模型，
        不能直接初始化 OpenAI / DeepSeek SDK。
    """

    generation_config = _get_generation_config(config)

    temperature = _safe_float(
        generation_config.get("temperature", 0.2),
        default=0.2,
    )
    max_tokens = _safe_int(
        generation_config.get("max_tokens", 1200),
        default=1200,
    )

    temperature = max(0.0, min(temperature, 2.0))
    max_tokens = max(max_tokens, 1)

    system_prompt = _get_system_prompt(config)

    if hasattr(llm, "chat"):
        try:
            return llm.chat(
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except TypeError:
            pass

        try:
            return llm.chat(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except TypeError:
            pass

        try:
            return llm.chat(prompt)
        except TypeError:
            pass

    if hasattr(llm, "generate_with_metadata"):
        try:
            return llm.generate_with_metadata(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except TypeError:
            try:
                return llm.generate_with_metadata(
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            except TypeError:
                return llm.generate_with_metadata(prompt)

    if hasattr(llm, "generate"):
        try:
            return llm.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except TypeError:
            try:
                return llm.generate(
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            except TypeError:
                return llm.generate(prompt)

    raise AttributeError("LLMClient has no supported generation method.")


def _get_system_prompt(config: Dict[str, Any]) -> str:
    """
    获取 generation system prompt。

    优先级：
        1. config["prompt"]["generation_system_prompt"]
        2. config["prompts"]["generation_system_prompt"] 兼容旧写法
        3. 默认 system prompt
    """

    prompt_config = config.get("prompt", {}) or {}
    prompts_config = config.get("prompts", {}) or {}

    custom_prompt = (
        prompt_config.get("generation_system_prompt")
        or prompts_config.get("generation_system_prompt")
        or ""
    )

    if custom_prompt:
        return str(custom_prompt).strip()

    return (
        "你是一个可靠、自然、专业的中文智能助手。"
        "请根据已有信息回答用户问题。"
        "不要编造没有依据的内容。"
        "如果信息不足，请明确说明不能可靠判断。"
        "不要暴露内部路由、节点、图结构、检索流程、实体链接等系统实现细节。"
    )


def _generate_with_rules(
    *,
    query: str,
    evidence: List[EvidenceItem],
    reasoning: ReasoningResult,
    answerability: str,
    semantic_score: float,
) -> Tuple[str, str]:
    """
    不调用 LLM 的规则生成。

    作用：
        1. LLM 不可用时兜底。
        2. 单元测试时保证系统可跑通。
        3. 对明确关系类问题优先生成直接答案。
    """

    answerability = _normalize_answerability(answerability)

    if not evidence:
        return (
            "目前可用信息不足，不能可靠回答这个问题。你可以补充更具体的对象、关系或背景信息，我再继续帮你分析。",
            "rule",
        )

    if answerability == "unanswerable":
        return (
            "目前可用信息不足，不能可靠回答这个问题。你可以补充更具体的对象、关系或背景信息，我再继续帮你分析。",
            "rule",
        )

    # -------------------------------------------------
    # 1. 优先尝试明确关系问答
    # -------------------------------------------------
    direct_answer = _try_build_direct_fact_answer(
        query=query,
        evidence=evidence,
        answerability=answerability,
    )

    if direct_answer:
        return direct_answer, "rule"

    # -------------------------------------------------
    # 2. 普通证据型回答
    # -------------------------------------------------
    prefix = "根据已有信息，" if answerability == "answerable" else "从已有信息看，可以谨慎判断："

    direct_relation_items = [
        item for item in evidence
        if str(item.get("evidence_type", "")).lower() == "relation"
    ]

    path_items = [
        item for item in evidence
        if str(item.get("evidence_type", "")).lower() == "path"
    ]

    neighbor_items = [
        item for item in evidence
        if str(item.get("evidence_type", "")).lower() == "neighbor"
    ]

    answer_parts: List[str] = []

    if direct_relation_items:
        answer_parts.append(
            prefix + _describe_relation_items(direct_relation_items)
        )
    elif path_items:
        answer_parts.append(
            prefix + _describe_path_items(path_items)
        )
    elif neighbor_items:
        answer_parts.append(
            prefix + _describe_neighbor_items(neighbor_items)
        )
    else:
        first_text = str(evidence[0].get("text", "")).strip()
        if first_text:
            answer_parts.append(prefix + first_text)
        else:
            answer_parts.append(
                "从已有信息看，这个问题有一定相关依据，但还不足以形成非常确定的结论。"
            )

    if answerability == "uncertain":
        answer_parts.append(
            "不过，目前信息并不充分，因此这个结论需要谨慎理解。"
        )

    basis = _build_basis_sentence(evidence)

    if basis:
        answer_parts.append(basis)

    final_answer = "\n\n".join(
        part.strip()
        for part in answer_parts
        if part and part.strip()
    )

    return final_answer, "rule"


def _try_build_direct_fact_answer(
    *,
    query: str,
    evidence: List[EvidenceItem],
    answerability: str,
) -> str:
    """
    对明确关系类问题直接生成答案。

    例：
        Q: Barack Obama 的妻子是谁？
        E: Barack Obama is the spouse of Michelle Obama.
        A: 根据已有信息，Barack Obama 的妻子是 Michelle Obama。

    只基于 evidence，不引入外部事实。
    """

    intent = _detect_relation_intent(query)

    if intent:
        answer = _answer_by_relation_intent(
            query=query,
            evidence=evidence,
            intent=intent,
            answerability=answerability,
        )
        if answer:
            return answer

    # 如果用户问“二者是什么关系”，则尝试关系说明。
    relationship_answer = _answer_relationship_question(
        query=query,
        evidence=evidence,
        answerability=answerability,
    )

    if relationship_answer:
        return relationship_answer

    return ""


def _detect_relation_intent(query: str) -> Dict[str, Any]:
    """
    根据用户问题识别关系意图。

    返回：
        {
            "relation_group": "spouse",
            "answer_label": "妻子",
            "target_as_value": True
        }
    """

    q = _normalize_text(query)

    intent_rules = [
        {
            "relation_group": "spouse",
            "keywords": [
                "妻子", "老婆", "夫人", "太太", "配偶", "spouse", "wife",
            ],
            "answer_label": "妻子",
        },
        {
            "relation_group": "spouse",
            "keywords": [
                "丈夫", "老公", "先生", "husband",
            ],
            "answer_label": "丈夫",
        },
        {
            "relation_group": "occupation",
            "keywords": [
                "职业", "工作", "职务", "做什么", "occupation", "job", "profession",
            ],
            "answer_label": "职业",
        },
        {
            "relation_group": "birth_place",
            "keywords": [
                "出生地", "出生在哪里", "哪里出生", "出生于", "born", "birthplace", "birth place",
            ],
            "answer_label": "出生地",
        },
        {
            "relation_group": "nationality",
            "keywords": [
                "国籍", "哪个国家的人", "是哪国人", "nationality", "citizen",
            ],
            "answer_label": "国籍",
        },
        {
            "relation_group": "position",
            "keywords": [
                "职位", "职称", "担任", "总统", "position", "office", "president",
            ],
            "answer_label": "职位",
        },
        {
            "relation_group": "country",
            "keywords": [
                "哪个国家", "所属国家", "国家", "country",
            ],
            "answer_label": "所属国家",
        },
        {
            "relation_group": "capital",
            "keywords": [
                "首都", "capital",
            ],
            "answer_label": "首都",
        },
    ]

    for rule in intent_rules:
        for keyword in rule["keywords"]:
            if _normalize_text(keyword) in q:
                return {
                    "relation_group": rule["relation_group"],
                    "answer_label": rule["answer_label"],
                }

    return {}


def _answer_by_relation_intent(
    *,
    query: str,
    evidence: List[EvidenceItem],
    intent: Dict[str, Any],
    answerability: str,
) -> str:
    """
    根据关系意图从 evidence 中抽取答案。
    """

    relation_group = str(intent.get("relation_group", "")).strip()
    answer_label = str(intent.get("answer_label", "")).strip()

    matched_items: List[EvidenceItem] = []

    for item in evidence:
        relation = _canonical_relation(item.get("relation", ""))
        if relation == relation_group:
            matched_items.append(item)

    if not matched_items:
        return ""

    best = _choose_best_relation_item_for_query(
        query=query,
        items=matched_items,
    )

    if not best:
        return ""

    source = _display_entity(best.get("source_entity", ""))
    target = _display_entity(best.get("target_entity", ""))
    relation = _canonical_relation(best.get("relation", ""))
    text = str(best.get("text", "") or "").strip()

    if not source or not target:
        extracted_source, extracted_target = _extract_entities_from_evidence_text(text, relation)
        source = source or extracted_source
        target = target or extracted_target

    if not source or not target:
        return ""

    query_norm = _normalize_text(query)
    source_norm = _normalize_text(source)
    target_norm = _normalize_text(target)

    source_in_query = _entity_mentioned_in_query(source, query)
    target_in_query = _entity_mentioned_in_query(target, query)

    # 常见属性关系：用户问 source 的某属性，答案通常是 target。
    # 如果 query 中明确包含 target 而不包含 source，则反向回答。
    if source_in_query and not target_in_query:
        subject = source
        value = target
    elif target_in_query and not source_in_query:
        subject = target
        value = source
    else:
        # 如果无法判断方向，则优先根据 relation 的方向回答。
        subject = source
        value = target

        # spouse 是对称关系。如果用户问“谁是 A 的妻子”，且 evidence 是 B spouse A，
        # 需要让 A 做 subject，B 做 value。
        if relation == "spouse":
            if target_norm and target_norm in query_norm:
                subject = target
                value = source

    if not value:
        return ""

    uncertainty_prefix = "从已有信息看，" if answerability == "uncertain" else "根据已有信息，"

    if relation == "spouse":
        answer_sentence = f"{uncertainty_prefix}{subject} 的{answer_label}是 {value}。"
    elif relation == "occupation":
        answer_sentence = f"{uncertainty_prefix}{subject} 的职业是 {value}。"
    elif relation == "birth_place":
        answer_sentence = f"{uncertainty_prefix}{subject} 的出生地是 {value}。"
    elif relation == "nationality":
        answer_sentence = f"{uncertainty_prefix}{subject} 的国籍相关信息指向 {value}。"
    elif relation == "position":
        answer_sentence = f"{uncertainty_prefix}{subject} 担任或关联的职位是 {value}。"
    elif relation == "country":
        answer_sentence = f"{uncertainty_prefix}{subject} 所属或关联的国家是 {value}。"
    elif relation == "capital":
        answer_sentence = f"{uncertainty_prefix}{subject} 的首都相关信息指向 {value}。"
    else:
        answer_sentence = f"{uncertainty_prefix}{subject} 与 {value} 存在“{best.get('relation', '')}”关系。"

    basis = _build_basis_sentence([best])

    if answerability == "uncertain":
        caution = "不过，目前依据有限，这个结论需要谨慎理解。"
        parts = [answer_sentence, caution]
    else:
        parts = [answer_sentence]

    if basis:
        parts.append(basis)

    return "\n\n".join(parts)


def _answer_relationship_question(
    *,
    query: str,
    evidence: List[EvidenceItem],
    answerability: str,
) -> str:
    """
    回答“二者有什么关系”类问题。
    """

    q = _normalize_text(query)

    relation_question_keywords = [
        "什么关系", "有什么关系", "关系是什么", "relationship", "relation",
    ]

    if not any(keyword in q for keyword in relation_question_keywords):
        return ""

    relation_items = [
        item for item in evidence
        if str(item.get("relation", "")).strip()
    ]

    if not relation_items:
        return ""

    best = _choose_best_relation_item_for_query(
        query=query,
        items=relation_items,
    )

    if not best:
        return ""

    source = _display_entity(best.get("source_entity", ""))
    target = _display_entity(best.get("target_entity", ""))
    relation = str(best.get("relation", "")).strip()
    relation_label = _relation_label_zh(relation)
    text = str(best.get("text", "") or "").strip()

    if not source or not target:
        extracted_source, extracted_target = _extract_entities_from_evidence_text(
            text=text,
            relation=_canonical_relation(relation),
        )
        source = source or extracted_source
        target = target or extracted_target

    if not source or not target or not relation:
        return ""

    prefix = "从已有信息看，" if answerability == "uncertain" else "根据已有信息，"

    if _canonical_relation(relation) == "spouse":
        answer_sentence = f"{prefix}{source} 和 {target} 是配偶关系。"
    else:
        answer_sentence = f"{prefix}{source} 和 {target} 存在“{relation_label}”关系。"

    basis = _build_basis_sentence([best])

    if basis:
        return answer_sentence + "\n\n" + basis

    return answer_sentence


def _choose_best_relation_item_for_query(
    *,
    query: str,
    items: List[EvidenceItem],
) -> Optional[EvidenceItem]:
    """
    从候选 evidence 中选择最适合回答 query 的一条。

    评分依据：
        1. source / target 是否出现在 query 中
        2. evidence 自身 score
        3. text 是否覆盖 query 关键词
    """

    if not items:
        return None

    best_item: Optional[EvidenceItem] = None
    best_score = -1.0

    for item in items:
        source = _display_entity(item.get("source_entity", ""))
        target = _display_entity(item.get("target_entity", ""))
        text = str(item.get("text", "") or "")

        score = _clip_score(
            _safe_float(
                item.get("score", 0.0),
                default=0.0,
            )
        )

        rank_score = score

        if source and _entity_mentioned_in_query(source, query):
            rank_score += 0.5

        if target and _entity_mentioned_in_query(target, query):
            rank_score += 0.5

        overlap = _token_overlap_score(query, text)
        rank_score += 0.2 * overlap

        if rank_score > best_score:
            best_score = rank_score
            best_item = item

    return best_item


def _canonical_relation(relation: Any) -> str:
    """
    关系名归一化。
    """

    raw = str(relation or "").strip().lower()
    raw = raw.replace("-", "_").replace(" ", "_")

    mapping = {
        "spouse": "spouse",
        "wife": "spouse",
        "husband": "spouse",
        "partner": "spouse",
        "配偶": "spouse",
        "妻子": "spouse",
        "丈夫": "spouse",

        "occupation": "occupation",
        "job": "occupation",
        "profession": "occupation",
        "work_as": "occupation",
        "worked_as": "occupation",
        "职业": "occupation",

        "birth_place": "birth_place",
        "birthplace": "birth_place",
        "place_of_birth": "birth_place",
        "born_in": "birth_place",
        "出生地": "birth_place",

        "nationality": "nationality",
        "citizenship": "nationality",
        "citizen_of": "nationality",
        "国籍": "nationality",

        "position": "position",
        "office": "position",
        "held_position": "position",
        "president": "position",
        "职位": "position",

        "country": "country",
        "located_in": "country",
        "part_of": "country",
        "所属国家": "country",

        "capital": "capital",
        "capital_of": "capital",
        "首都": "capital",
    }

    return mapping.get(raw, raw)


def _relation_label_zh(relation: Any) -> str:
    """
    关系名中文展示。
    """

    canonical = _canonical_relation(relation)

    labels = {
        "spouse": "配偶",
        "occupation": "职业",
        "birth_place": "出生地",
        "nationality": "国籍",
        "position": "职位",
        "country": "所属国家",
        "capital": "首都",
    }

    return labels.get(canonical, str(relation or "").strip() or "相关")


def _extract_entities_from_evidence_text(
    text: str,
    relation: str,
) -> Tuple[str, str]:
    """
    从英文 evidence text 中尽量抽取 source / target。

    这是兜底逻辑，只用于 display 字段缺失时。
    """

    text = str(text or "").strip()

    if not text:
        return "", ""

    patterns = [
        r"^(?P<src>.+?)\s+is\s+the\s+spouse\s+of\s+(?P<tgt>.+?)\.$",
        r"^(?P<src>.+?)\s+worked\s+as\s+a\s+(?P<tgt>.+?)\.$",
        r"^(?P<src>.+?)\s+was\s+born\s+in\s+(?P<tgt>.+?)\.$",
        r"^(?P<src>.+?)\s+is\s+located\s+in\s+(?P<tgt>.+?)\.$",
        r"^(?P<src>.+?)\s+is\s+the\s+capital\s+of\s+(?P<tgt>.+?)\.$",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return (
                match.group("src").strip(),
                match.group("tgt").strip(),
            )

    return "", ""


def _entity_mentioned_in_query(entity: str, query: str) -> bool:
    """
    判断实体是否出现在用户问题中。

    同时支持：
        Barack Obama
        barack_obama
        Obama
    """

    entity = str(entity or "").strip()
    query = str(query or "").strip()

    if not entity or not query:
        return False

    entity_norm = _normalize_text(entity)
    query_norm = _normalize_text(query)

    if entity_norm and entity_norm in query_norm:
        return True

    entity_parts = [
        part for part in re.split(r"[\s_·.\-]+", entity_norm)
        if part
    ]

    # 英文人名场景，允许姓氏命中，但避免单字符误判。
    for part in entity_parts:
        if len(part) >= 3 and part in query_norm:
            return True

    return False


def _display_entity(value: Any) -> str:
    """
    将 node_key / entity_id 转成人类可读名称。

    例：
        barack_obama -> Barack Obama
        united_states -> United States
    """

    text = str(value or "").strip()

    if not text:
        return ""

    # 已经像自然名称时，直接返回。
    if " " in text or "·" in text:
        return text

    text = text.replace("-", "_")
    parts = [
        part for part in text.split("_")
        if part
    ]

    if not parts:
        return text

    small_words = {"of", "the", "and", "in", "on", "for", "to"}

    display_parts: List[str] = []
    for idx, part in enumerate(parts):
        lower = part.lower()
        if idx > 0 and lower in small_words:
            display_parts.append(lower)
        else:
            display_parts.append(part[:1].upper() + part[1:])

    return " ".join(display_parts)


def _normalize_text(text: Any) -> str:
    """
    文本归一化，用于规则匹配。
    """

    text = str(text or "").strip().lower()
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    return text


def _token_overlap_score(a: str, b: str) -> float:
    """
    简单 token overlap。
    """

    a_tokens = set(
        token for token in re.split(r"[\s,，。！？?;；:：()（）\"'“”]+", _normalize_text(a))
        if len(token) >= 2
    )
    b_tokens = set(
        token for token in re.split(r"[\s,，。！？?;；:：()（）\"'“”]+", _normalize_text(b))
        if len(token) >= 2
    )

    if not a_tokens or not b_tokens:
        return 0.0

    return len(a_tokens & b_tokens) / max(len(a_tokens), 1)


def _describe_relation_items(
    items: List[EvidenceItem],
) -> str:
    """
    描述直接关系材料。
    """

    descriptions: List[str] = []

    for item in items[:3]:
        source = _display_entity(item.get("source_entity", ""))
        relation = _relation_label_zh(item.get("relation", ""))
        target = _display_entity(item.get("target_entity", ""))
        text = str(item.get("text", "")).strip()

        if source and relation and target:
            descriptions.append(f"{source} 与 {target} 之间存在“{relation}”关系")
        elif text:
            descriptions.append(text)

    if not descriptions:
        return "已有信息可以支撑回答该问题。"

    return "；".join(descriptions) + "。"


def _describe_path_items(
    items: List[EvidenceItem],
) -> str:
    """
    描述路径材料。
    """

    descriptions: List[str] = []

    for item in items[:2]:
        path = item.get("path", []) or []
        text = str(item.get("text", "")).strip()

        if isinstance(path, list) and path:
            descriptions.append(
                " → ".join(_display_entity(node) for node in path)
            )
        elif text:
            descriptions.append(text)

    if not descriptions:
        return "相关对象之间存在一定间接联系。"

    return "相关对象之间存在间接联系：" + "；".join(descriptions) + "。"


def _describe_neighbor_items(
    items: List[EvidenceItem],
) -> str:
    """
    描述邻近材料。
    """

    descriptions: List[str] = []

    for item in items[:3]:
        source = _display_entity(item.get("source_entity", ""))
        target = _display_entity(item.get("target_entity", ""))
        relation = _relation_label_zh(item.get("relation", ""))
        text = str(item.get("text", "")).strip()

        if source and target:
            if relation:
                descriptions.append(f"{source} 与 {target} 存在“{relation}”关系")
            else:
                descriptions.append(f"{source} 与 {target} 存在相关联系")
        elif text:
            descriptions.append(text)

    if not descriptions:
        return "已有信息显示相关对象之间存在一定联系。"

    return "；".join(descriptions) + "。"


def _build_basis_sentence(
    evidence: List[EvidenceItem],
) -> str:
    """
    构造简短依据说明。
    """

    if not evidence:
        return ""

    top_items = evidence[:3]
    basis_parts: List[str] = []

    for idx, item in enumerate(top_items, start=1):
        text = str(item.get("text", "")).strip()

        if not text:
            source = _display_entity(item.get("source_entity", ""))
            relation = _relation_label_zh(item.get("relation", ""))
            target = _display_entity(item.get("target_entity", ""))

            if source and relation and target:
                text = f"{source} 与 {target} 存在“{relation}”关系。"

        if not text:
            continue

        basis_parts.append(f"{idx}. {text}")

    if not basis_parts:
        return ""

    return "依据是：" + " ".join(basis_parts)


def _build_clarification_answer(state: AgentState) -> str:
    """
    构造澄清问题回答。
    """

    clarifying_question = str(
        state.get("clarifying_question", "") or ""
    ).strip()

    if clarifying_question:
        return _sanitize_user_answer(clarifying_question)

    return "请你补充得更具体一些，例如你想问哪个对象、它和谁的关系，或者你想了解哪方面信息。"


def _build_error_answer(state: AgentState) -> str:
    """
    构造异常兜底回答。

    注意：
        不把内部错误细节暴露给用户。
    """

    return "抱歉，我刚刚处理这个问题时遇到了一点问题。你可以换一种问法，或者补充更多上下文后再试。"


def _build_unanswerable_answer(
    *,
    query: str,
    state: AgentState,
) -> str:
    """
    构造不可回答场景的回答。
    """

    ungrounded_mentions = state.get("ungrounded_mentions", []) or []
    scoring_reason = str(state.get("scoring_reason", "") or "").strip()

    if ungrounded_mentions:
        mention_text = "、".join(
            str(item) for item in ungrounded_mentions[:3]
            if str(item).strip()
        )

        if mention_text:
            return (
                f"目前我没有找到足够可靠的信息来回答这个问题，尤其是关于“{mention_text}”的部分还不够明确。"
                "你可以补充更完整的名称、上下文，或者换一种更具体的问法。"
            )

    if scoring_reason:
        return _sanitize_user_answer(
            f"{scoring_reason} 因此目前不能可靠回答。你可以补充更具体的对象、关系或背景信息，我再继续帮你分析。"
        )

    return "目前可用信息不足，不能可靠回答这个问题。你可以补充更具体的对象、关系或背景信息，我再继续帮你分析。"


def _extract_answer(raw_result: Any) -> str:
    """
    从不同格式的 LLM 返回结果中提取文本。
    """

    if raw_result is None:
        return ""

    if isinstance(raw_result, str):
        return raw_result.strip()

    if isinstance(raw_result, dict):
        for key in ["content", "answer", "text", "output", "response"]:
            value = raw_result.get(key)
            if value:
                return str(value).strip()

        message = raw_result.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if content:
                return str(content).strip()

        choices = raw_result.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]

            if isinstance(first, dict):
                message = first.get("message")
                if isinstance(message, dict) and message.get("content"):
                    return str(message["content"]).strip()

                if first.get("text"):
                    return str(first["text"]).strip()

    if hasattr(raw_result, "choices"):
        try:
            choice = raw_result.choices[0]
            content = choice.message.content
            return str(content).strip()
        except Exception:
            pass

    if hasattr(raw_result, "content"):
        try:
            return str(raw_result.content).strip()
        except Exception:
            pass

    return str(raw_result).strip()


def _build_evidence_brief(
    evidence: List[EvidenceItem],
) -> str:
    """
    构造材料摘要。
    """

    if not evidence:
        return "无"

    lines: List[str] = []

    for idx, item in enumerate(evidence[:8], start=1):
        evidence_type = str(item.get("evidence_type", "")).strip()
        source = _display_entity(item.get("source_entity", ""))
        relation = _relation_label_zh(item.get("relation", ""))
        target = _display_entity(item.get("target_entity", ""))
        score = _safe_float(item.get("score", 0.0), default=0.0)

        if source or relation or target:
            lines.append(
                f"{idx}. 类型={evidence_type}；对象={source}；关系={relation}；相关对象={target}；匹配度={score:.3f}"
            )
        else:
            text = str(item.get("text", "")).strip()
            if text:
                lines.append(f"{idx}. {text}")

    return "\n".join(lines) if lines else "无"


def _sanitize_user_answer(answer: str) -> str:
    """
    清理最终用户侧回答。

    目的：
        避免输出内部实现细节。
    """

    answer = str(answer or "").strip()

    if not answer:
        return ""

    replacements = {
        "知识图谱": "已有信息",
        "KG-RAG": "当前系统",
        "KGRAG": "当前系统",
        "kg_rag": "当前系统",
        "knowledge graph": "已有信息",
        "实体链接": "对象识别",
        "实体抽取": "对象识别",
        "图谱检索": "信息查找",
        "检索节点": "处理步骤",
        "路由节点": "处理步骤",
        "LangGraph": "当前系统",
        "StateGraph": "当前系统",
        "内部流程": "处理过程",
        "内部路由": "处理过程",
    }

    cleaned = answer

    for old, new in replacements.items():
        cleaned = re.sub(
            re.escape(old),
            new,
            cleaned,
            flags=re.IGNORECASE,
        )

    # 单独处理 KG，避免误伤普通英文单词中的 kg 片段。
    cleaned = re.sub(
        r"\bKG\b",
        "当前系统",
        cleaned,
        flags=re.IGNORECASE,
    )

    cleaned = _remove_internal_debug_lines(cleaned)

    return cleaned.strip()


def _remove_internal_debug_lines(text: str) -> str:
    """
    移除疑似内部调试行。
    """

    lines = text.splitlines()
    kept_lines: List[str] = []

    forbidden_patterns = [
        r"^route\s*[:：]",
        r"^node\s*[:：]",
        r"^state\s*[:：]",
        r"^metadata\s*[:：]",
        r"^trace\s*[:：]",
        r"^debug\s*[:：]",
        r"^error_stage\s*[:：]",
        r"^answerability\s*[:：]",
        r"^semantic_score\s*[:：]",
    ]

    for line in lines:
        stripped = line.strip()

        if not stripped:
            kept_lines.append(line)
            continue

        should_drop = False

        for pattern in forbidden_patterns:
            if re.search(pattern, stripped, re.IGNORECASE):
                should_drop = True
                break

        if not should_drop:
            kept_lines.append(line)

    return "\n".join(kept_lines)


def _normalize_answerability(value: Any) -> str:
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
