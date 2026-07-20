# -*- coding: utf-8 -*-
"""Mention extraction domain service.

The rule patterns, cleaning, overlap handling, and LLM-response parsing are
migrated from the original mention_extraction_node without changing their
problem-solving behavior. This module has no LangGraph dependency.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Mapping, Optional, Tuple

from .schemas import Mention, MentionExtractionOptions, MentionExtractionResult


class MentionExtractor:
    """Extract mentions without reading or writing AgentState."""

    def __init__(
        self,
        *,
        llm_client: Any = None,
        default_options: Optional[MentionExtractionOptions] = None,
    ) -> None:
        self.llm_client = llm_client
        self.default_options = default_options or MentionExtractionOptions()

    def extract(
        self,
        query: str,
        *,
        options: Optional[MentionExtractionOptions | Mapping[str, Any]] = None,
        config: Optional[Mapping[str, Any]] = None,
    ) -> MentionExtractionResult:
        query_text = str(query or "").strip()
        if not query_text:
            return MentionExtractionResult(
                mentions=[],
                extractor_type="empty_input",
                warnings=["Empty query; no mentions extracted."],
            )

        resolved = self._resolve_options(options=options, config=config)
        if resolved.use_llm:
            mentions, raw_output, extractor_type = _extract_with_llm(
                query_text,
                self.llm_client,
            )
        else:
            mentions, raw_output, extractor_type = _extract_with_rules(query_text)

        merged_config = dict(config or {})
        extraction_config = dict(_get_extraction_config(merged_config))
        extraction_config["min_confidence"] = resolved.min_confidence
        extraction_config["max_mentions"] = resolved.max_mentions
        merged_config["mention_extraction"] = extraction_config

        mentions = _postprocess_mentions(
            query=query_text,
            mentions=mentions,
            config=merged_config,
        )
        mentions = mentions[: resolved.max_mentions]

        warnings: List[str] = []
        if not mentions:
            warnings.append("No mention extracted.")

        return MentionExtractionResult(
            mentions=mentions,
            raw_output=raw_output,
            extractor_type=extractor_type,
            warnings=warnings,
        )

    def _resolve_options(
        self,
        *,
        options: Optional[MentionExtractionOptions | Mapping[str, Any]],
        config: Optional[Mapping[str, Any]],
    ) -> MentionExtractionOptions:
        if isinstance(options, MentionExtractionOptions):
            return options
        if options is not None:
            return MentionExtractionOptions.from_mapping(options)
        configured = _get_extraction_config(dict(config or {}))
        if configured:
            return MentionExtractionOptions.from_mapping(configured)
        return self.default_options


def extract_mentions(
    query: str,
    *,
    llm_client: Any = None,
    options: Optional[MentionExtractionOptions | Mapping[str, Any]] = None,
    config: Optional[Mapping[str, Any]] = None,
) -> MentionExtractionResult:
    return MentionExtractor(llm_client=llm_client).extract(
        query,
        options=options,
        config=config,
    )


def _get_extraction_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    获取 mention extraction 配置。

    兼容两种来源：
        1. config["mention_extraction"]
        2. config["graph"]["mention_extraction"]

    如果两边都有，graph.mention_extraction 覆盖顶层 mention_extraction。
    """

    top_level_config = config.get("mention_extraction", {}) or {}
    graph_config = config.get("graph", {}) or {}
    graph_extraction_config = graph_config.get("mention_extraction", {}) or {}

    merged = dict(top_level_config)
    merged.update(graph_extraction_config)

    return merged



def _extract_with_llm(
    query: str,
    llm_client: Any,
) -> Tuple[List[Mention], str, str]:
    """Use the injected LLM client and fall back to the original rule extractor."""
    if llm_client is None:
        mentions, raw_output, _ = _extract_with_rules(query)
        return mentions, raw_output, "rule_fallback"

    try:
        prompt = _build_mention_prompt(query)
        raw_output = _call_llm(llm_client, prompt)
        mentions = _parse_llm_mentions(query=query, raw_output=raw_output)
        return mentions, raw_output, "llm"
    except Exception:
        mentions, raw_output, _ = _extract_with_rules(query)
        fallback_raw = raw_output or "LLM mention extractor failed. Fallback to rule extractor."
        return mentions, fallback_raw, "rule_fallback"


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
            max_tokens=512,
        )

        return _extract_llm_text(result)

    if hasattr(llm, "generate"):
        result = llm.generate(
            prompt=prompt,
            temperature=0.0,
            max_tokens=512,
        )

        return _extract_llm_text(result)

    if hasattr(llm, "chat"):
        try:
            result = llm.chat(
                messages=[
                    {
                        "role": "system",
                        "content": "你是一个信息抽取助手，只输出 JSON。",
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=0.0,
                max_tokens=512,
            )
        except TypeError:
            result = llm.chat(prompt)

        return _extract_llm_text(result)

    raise AttributeError("LLMClient has no supported generation method.")


def _extract_llm_text(result: Any) -> str:
    """
    从不同格式的 LLM 返回中提取文本。
    """

    if result is None:
        return ""

    if isinstance(result, str):
        return result

    if isinstance(result, dict):
        for key in ["content", "answer", "text", "output", "response"]:
            value = result.get(key)
            if value:
                return str(value)

        message = result.get("message")
        if isinstance(message, dict) and message.get("content"):
            return str(message["content"])

        choices = result.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]

            if isinstance(first, dict):
                message = first.get("message")

                if isinstance(message, dict) and message.get("content"):
                    return str(message["content"])

                if first.get("text"):
                    return str(first["text"])

    if hasattr(result, "choices"):
        try:
            choice = result.choices[0]
            return str(choice.message.content)
        except Exception:
            pass

    if hasattr(result, "content"):
        try:
            return str(result.content)
        except Exception:
            pass

    return str(result)


def _build_mention_prompt(query: str) -> str:
    """
    构造 mention 抽取 prompt。

    注意：
        这是内部 prompt，用户不可见。
    """

    return f"""
你是一个信息抽取模块。

任务：
从用户问题中抽取需要进一步识别的对象名称，也就是用户真正询问的具体人名、地名、机构名、作品名、概念名、疾病名、药物名、事件名、专有名词等。

核心规则：
1. 只抽取对象本身，不要把关系词抽进去。
2. 如果问题是“Barack Obama 的妻子是谁？”，只抽取 “Barack Obama”，不要抽取 “Barack Obama 的妻子”。
3. 如果问题是“谁是 Barack Obama 的妻子？”，只抽取 “Barack Obama”。
4. 如果问题是“A 和 B 有什么关系？”，抽取 “A” 和 “B”。
5. 如果问题是“介绍一下 X”，抽取 “X”。

不要抽取：
- 普通疑问词：谁、什么、哪里、哪个、哪些
- 普通关系词：关系、路径、属于、相关、连接、妻子、丈夫、职业、出生地、国籍
- 泛化动词：介绍、说明、解释、分析
- 没有明确指代的代词：它、他、她、这个、那个

要求：
1. 只输出 JSON。
2. 不要输出 Markdown。
3. 如果没有明确对象，输出空列表。
4. start 和 end 是 mention 在原问题中的字符位置，end 为开区间。
5. confidence 取值 0 到 1。

用户问题：
{query}

输出格式：
[
    {{
        "text": "对象名称",
        "start": 0,
        "end": 4,
        "type": "person | organization | location | concept | event | work | unknown",
        "confidence": 0.95
    }}
]
""".strip()


def _parse_llm_mentions(
    *,
    query: str,
    raw_output: str,
) -> List[Mention]:
    """
    解析 LLM 输出的 mention JSON。
    """

    json_text = _extract_json_array(raw_output)

    if not json_text:
        return []

    try:
        data = json.loads(json_text)
    except Exception:
        return []

    if not isinstance(data, list):
        return []

    mentions: List[Mention] = []

    for item in data:
        if not isinstance(item, dict):
            continue

        text = str(item.get("text", "")).strip()

        if not text:
            continue

        text = _clean_candidate_text(text)

        if not _is_valid_mention_text(text):
            continue

        start = item.get("start")
        end = item.get("end")

        if not isinstance(start, int) or not isinstance(end, int):
            start, end = _find_span(query, text)

        if start < 0 or end <= start:
            continue

        mention = Mention(
            text=text,
            start=start,
            end=end,
            type=str(item.get("type", "unknown") or "unknown"),
            confidence=_safe_float(item.get("confidence", 0.8), default=0.8),
        )

        mentions.append(mention)

    return mentions


# =========================================================
# 4. 规则抽取
# =========================================================

def _extract_with_rules(query: str) -> Tuple[List[Mention], str, str]:
    """
    使用规则抽取 mentions。

    规则优先级：
        1. 引号 / 书名号中的对象
        2. 明确属性问题中的主体对象
        3. 明确关系模板中的 A、B
        4. “关于 X / 介绍 X / X 是什么”类模板
        5. 英文大写专有名词片段
        6. 中英文混合专有名词
    """

    mentions: List[Mention] = []

    mentions.extend(_extract_quoted_mentions(query))
    mentions.extend(_extract_possessive_attribute_mentions(query))
    mentions.extend(_extract_relation_pattern_mentions(query))
    mentions.extend(_extract_about_pattern_mentions(query))
    mentions.extend(_extract_english_proper_mentions(query))
    mentions.extend(_extract_mixed_mentions(query))

    return mentions, "", "rule"


def _extract_quoted_mentions(query: str) -> List[Mention]:
    """
    抽取引号、书名号中的对象。

    示例：
        《三体》的作者是谁？
        "Barack Obama" 的妻子是谁？
    """

    patterns = [
        r"《([^》]{1,80})》",
        r"“([^”]{1,80})”",
        r"\"([^\"]{1,80})\"",
        r"'([^']{1,80})'",
        r"「([^」]{1,80})」",
        r"『([^』]{1,80})』",
    ]

    mentions: List[Mention] = []

    for pattern in patterns:
        for match in re.finditer(pattern, query):
            text = _clean_candidate_text(match.group(1))
            start = match.start(1)
            end = match.end(1)

            if _is_valid_mention_text(text):
                mentions.append(
                    Mention(
                        text=text,
                        start=start,
                        end=end,
                        type="unknown",
                        confidence=0.95,
                    )
                )

    return mentions


def _extract_possessive_attribute_mentions(query: str) -> List[Mention]:
    """
    抽取“主体 + 的 + 属性/关系”类问题中的主体对象。

    示例：
        Barack Obama 的妻子是谁？
        Barack Obama 的职业是什么？
        Michelle Obama 的工作是什么？
        谁是 Barack Obama 的妻子？
        What is Michelle Obama's occupation?
        Who is the wife of Barack Obama?
    """

    patterns = [
        # A 的妻子是谁？
        r"(?P<entity>.+?)的(?:妻子|老婆|夫人|太太|丈夫|老公|先生|配偶)(?:是谁|是什么|叫什么|是哪位|$|[？?])",

        # A 的职业是什么？
        r"(?P<entity>.+?)的(?:职业|工作|职位|职务|身份|国籍|出生地|首都|作者|创始人|所属国家)(?:是什么|是谁|在哪里|是哪|有哪些|$|[？?])",

        # 谁是 A 的妻子？
        r"(?:谁是|哪位是)(?P<entity>.+?)的(?:妻子|老婆|夫人|太太|丈夫|老公|先生|配偶)",

        # A 出生在哪里？
        r"(?P<entity>.+?)(?:出生在哪里|在哪里出生|出生于哪里|出生于哪|是哪国人|来自哪里)",

        # A 是哪个国家的人？
        r"(?P<entity>.+?)(?:是哪个国家的人|是哪国人|属于哪个国家)",

        # What is A's occupation?
        r"(?:what\s+is|what's)\s+(?P<entity>.+?)['’]s\s+(?:occupation|job|profession|nationality|birthplace|birth\s+place)\??",

        # Who is the wife of A?
        r"(?:who\s+is|who's)\s+(?:the\s+)?(?:wife|husband|spouse)\s+of\s+(?P<entity>.+?)\??",

        # Where was A born?
        r"(?:where\s+was)\s+(?P<entity>.+?)\s+born\??",
    ]

    mentions: List[Mention] = []

    for pattern in patterns:
        match = re.search(pattern, query, flags=re.IGNORECASE)

        if not match:
            continue

        text = _clean_candidate_text(match.group("entity"))

        if not _is_valid_mention_text(text):
            continue

        start, end = _find_span(query, text)

        if start < 0:
            continue

        mentions.append(
            Mention(
                text=text,
                start=start,
                end=end,
                type="unknown",
                confidence=0.93,
            )
        )

        return mentions

    return mentions


def _extract_relation_pattern_mentions(query: str) -> List[Mention]:
    """
    抽取关系型问题中的对象。

    示例：
        姚明和 NBA 有什么关系？
        Barack Obama 与 Michelle Obama 的关系是什么？
        A 到 B 是否存在路径？
    """

    patterns = [
        r"(.+?)和(.+?)(?:有什么关系|的关系是什么|之间是什么关系|之间有什么联系|有什么联系|是什么关系)",
        r"(.+?)与(.+?)(?:有什么关系|的关系是什么|之间是什么关系|之间有什么联系|有什么联系|是什么关系)",
        r"(.+?)到(.+?)(?:是否存在路径|有什么路径|的路径是什么|怎么连接)",
        r"(.+?)是否(?:连接|认识|关联|相关于)(.+)",
        r"(.+?)(?:连接到|关联到|相关于)(.+)",
        r"(.+?)属于(.+)",
        r"what\s+is\s+the\s+relationship\s+between\s+(.+?)\s+and\s+(.+?)\??",
        r"how\s+is\s+(.+?)\s+related\s+to\s+(.+?)\??",
    ]

    mentions: List[Mention] = []

    for pattern in patterns:
        match = re.search(pattern, query, flags=re.IGNORECASE)

        if not match:
            continue

        for group_index in range(1, len(match.groups()) + 1):
            raw_text = match.group(group_index)
            text = _clean_candidate_text(raw_text)

            if not _is_valid_mention_text(text):
                continue

            start, end = _find_span(query, text)

            if start < 0:
                continue

            mentions.append(
                Mention(
                    text=text,
                    start=start,
                    end=end,
                    type="unknown",
                    confidence=0.9,
                )
            )

        if mentions:
            return mentions

    return mentions


def _extract_about_pattern_mentions(query: str) -> List[Mention]:
    """
    抽取单对象问题中的对象。

    示例：
        介绍一下姚明
        Barack Obama 是谁？
        阿尔茨海默病是什么？
        和 Tesla 相关的公司有哪些？
    """

    patterns = [
        r"(?:介绍一下|介绍|说明一下|说明|解释一下|解释|分析一下|分析)(.+)",
        r"(?:关于|有关)(.+?)(?:的信息|的内容|的情况|是什么|有哪些|$|[？?])",
        r"(.+?)(?:是什么|是谁|属于什么|有哪些关系|有哪些相关对象|有哪些相关内容|相关的对象有哪些|相关内容有哪些)",
        r"(?:什么是|who is|what is|tell me about|explain|describe)\s+(.+)",
        r"(?:和|与)(.+?)(?:相关的对象有哪些|相关内容有哪些|相关的信息有哪些)",
    ]

    mentions: List[Mention] = []

    for pattern in patterns:
        match = re.search(pattern, query, flags=re.IGNORECASE)

        if not match:
            continue

        text = _clean_candidate_text(match.group(1))

        if not _is_valid_mention_text(text):
            continue

        start, end = _find_span(query, text)

        if start < 0:
            continue

        mentions.append(
            Mention(
                text=text,
                start=start,
                end=end,
                type="unknown",
                confidence=0.82,
            )
        )

        return mentions

    return mentions


def _extract_english_proper_mentions(query: str) -> List[Mention]:
    """
    抽取英文专有名词片段。

    示例：
        Barack Obama
        Michelle Obama
        New York City
        Alzheimer's disease
        National Basketball Association
    """

    patterns = [
        r"\b(?:[A-Z][a-zA-Z0-9_\-']+)(?:\s+(?:[A-Z][a-zA-Z0-9_\-']+|of|and|the|for|in|on|de|van|von)){0,6}\b",
        r"\b[A-Z]{2,}(?:-[A-Z0-9]+)?\b",
    ]

    mentions: List[Mention] = []

    for pattern in patterns:
        for match in re.finditer(pattern, query):
            raw_text = match.group(0)
            text = _clean_candidate_text(raw_text)

            if not _is_valid_mention_text(text):
                continue

            start, end = _find_span(query, text)

            if start < 0:
                start, end = match.start(), match.end()

            mentions.append(
                Mention(
                    text=text,
                    start=start,
                    end=end,
                    type="unknown",
                    confidence=0.78,
                )
            )

    return mentions


def _extract_mixed_mentions(query: str) -> List[Mention]:
    """
    抽取中英文混合专有名词。

    示例：
        ADNI 数据集
        ResNet 模型
        NBA 球员
        阿尔茨海默病 AD
    """

    patterns = [
        r"[\u4e00-\u9fff]{1,20}\s*[A-Za-z][A-Za-z0-9_\-]{1,20}",
        r"[A-Za-z][A-Za-z0-9_\-]{1,20}\s*[\u4e00-\u9fff]{1,20}",
    ]

    mentions: List[Mention] = []

    for pattern in patterns:
        for match in re.finditer(pattern, query):
            raw_text = match.group(0)
            text = _clean_candidate_text(raw_text)

            if not _is_valid_mention_text(text):
                continue

            start, end = _find_span(query, text)

            if start < 0:
                start, end = match.start(), match.end()

            mentions.append(
                Mention(
                    text=text,
                    start=start,
                    end=end,
                    type="unknown",
                    confidence=0.75,
                )
            )

    return mentions


# =========================================================
# 5. 后处理
# =========================================================

def _postprocess_mentions(
    *,
    query: str,
    mentions: List[Mention],
    config: Dict[str, Any],
) -> List[Mention]:
    """
    mention 后处理。

    包括：
        1. 文本清洗
        2. 关系后缀裁剪
        3. span 修正
        4. 停用词过滤
        5. 置信度过滤
        6. 去重
        7. 最大数量限制
    """

    extraction_config = _get_extraction_config(config)

    min_confidence = _safe_float(
        extraction_config.get("min_confidence", 0.3),
        default=0.3,
    )
    max_mentions = _safe_int(
        extraction_config.get("max_mentions", 8),
        default=8,
    )

    processed: List[Mention] = []

    for mention in mentions:
        raw_text = str(mention.get("text", ""))
        text = _clean_candidate_text(raw_text)

        if not _is_valid_mention_text(text):
            continue

        confidence = _safe_float(
            mention.get("confidence", 0.7),
            default=0.7,
        )

        if confidence < min_confidence:
            continue

        start, end = _find_span(query, text)

        if start < 0:
            old_start = mention.get("start")
            old_end = mention.get("end")

            if isinstance(old_start, int) and isinstance(old_end, int) and old_start >= 0 and old_end > old_start:
                start, end = old_start, old_end
            else:
                continue

        processed.append(
            Mention(
                text=text,
                start=start,
                end=end,
                type=str(mention.get("type", "unknown") or "unknown"),
                confidence=confidence,
            )
        )

    processed = _deduplicate_mentions(processed)
    processed = _remove_low_quality_overlaps(processed)
    processed = sorted(
        processed,
        key=lambda item: (
            -float(item.get("confidence", 0.0)),
            item.get("start", 0),
            len(str(item.get("text", ""))),
        ),
    )

    return processed[:max_mentions]


def _deduplicate_mentions(mentions: List[Mention]) -> List[Mention]:
    """
    mention 去重。

    规则：
        1. 文本完全相同，保留置信度高的。
        2. 文本归一化相同，保留更干净的。
    """

    if not mentions:
        return []

    by_key: Dict[str, Mention] = {}

    for mention in mentions:
        text = str(mention.get("text", "")).strip()

        if not text:
            continue

        key = _normalize_for_dedup(text)

        if key not in by_key:
            by_key[key] = mention
            continue

        old = by_key[key]

        if _is_better_mention(new=mention, old=old):
            by_key[key] = mention

    return list(by_key.values())


def _remove_low_quality_overlaps(mentions: List[Mention]) -> List[Mention]:
    """
    移除重叠或包含关系中的低质量 mention。

    核心目标：
        如果同时出现：
            Barack Obama
            Barack Obama 的妻子
        保留：
            Barack Obama
    """

    if not mentions:
        return []

    keep: List[Mention] = []

    for mention in mentions:
        text = str(mention.get("text", "")).strip()
        confidence = float(mention.get("confidence", 0.0))
        start = int(mention.get("start", -1))
        end = int(mention.get("end", -1))

        drop = False

        for other in mentions:
            if mention is other:
                continue

            other_text = str(other.get("text", "")).strip()
            other_conf = float(other.get("confidence", 0.0))
            other_start = int(other.get("start", -1))
            other_end = int(other.get("end", -1))

            if not text or not other_text:
                continue

            text_norm = _normalize_for_dedup(text)
            other_norm = _normalize_for_dedup(other_text)

            if text_norm == other_norm:
                continue

            span_overlap = _spans_overlap(start, end, other_start, other_end)
            text_contains = text_norm in other_norm or other_norm in text_norm

            if not span_overlap and not text_contains:
                continue

            # 当前 mention 是带关系后缀的长短语，且另一个更短更干净，丢弃当前。
            if _has_relation_suffix(text) and len(other_text) < len(text):
                if other_conf >= confidence - 0.2:
                    drop = True
                    break

            # 当前 mention 包含另一个 mention，且另一个更像专名，保留更短专名。
            if other_norm in text_norm and len(other_text) < len(text):
                if _looks_like_entity_name(other_text) and other_conf >= confidence - 0.2:
                    drop = True
                    break

        if not drop:
            keep.append(mention)

    return keep


def _is_better_mention(
    *,
    new: Mention,
    old: Mention,
) -> bool:
    """
    判断 new 是否比 old 更适合保留。
    """

    new_text = str(new.get("text", "")).strip()
    old_text = str(old.get("text", "")).strip()

    new_conf = float(new.get("confidence", 0.0))
    old_conf = float(old.get("confidence", 0.0))

    new_has_suffix = _has_relation_suffix(new_text)
    old_has_suffix = _has_relation_suffix(old_text)

    if old_has_suffix and not new_has_suffix:
        return True

    if new_has_suffix and not old_has_suffix:
        return False

    if new_conf > old_conf + 0.05:
        return True

    if abs(new_conf - old_conf) <= 0.05 and len(new_text) < len(old_text):
        return True

    return False


# =========================================================
# 6. 文本清洗与校验
# =========================================================

def _clean_candidate_text(text: str) -> str:
    """
    清洗候选 mention 文本。
    """

    text = str(text or "").strip()

    if not text:
        return ""

    text = text.replace("\u3000", " ")
    text = re.sub(r"\s+", " ", text).strip()

    # 去掉开头常见虚词
    prefix_patterns = [
        r"^请问",
        r"^麻烦",
        r"^帮我看看",
        r"^帮我查一下",
        r"^帮我分析一下",
        r"^我想知道",
        r"^我想问",
        r"^请介绍",
        r"^介绍一下",
        r"^介绍",
        r"^关于",
        r"^有关",
        r"^the\s+",
        r"^a\s+",
        r"^an\s+",
        r"^who\s+is\s+",
        r"^what\s+is\s+",
        r"^what's\s+",
        r"^tell\s+me\s+about\s+",
    ]

    for pattern in prefix_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()

    # 去掉首尾标点
    text = text.strip(" \t\r\n:：,，.。?？!！;；、()（）[]【】{}<>《》\"'“”‘’")

    # 去掉“实体 + 的 + 属性/关系词”后缀
    text = _strip_relation_suffix(text)

    # 去掉二次产生的标点
    text = text.strip(" \t\r\n:：,，.。?？!！;；、()（）[]【】{}<>《》\"'“”‘’")

    # 去掉末尾常见问题词
    suffix_patterns = [
        r"是什么$",
        r"是谁$",
        r"有哪些$",
        r"有什么$",
        r"怎么样$",
        r"怎么回事$",
        r"的关系$",
        r"相关内容$",
        r"相关对象$",
        r"的信息$",
        r"的情况$",
        r"\?$",
        r"？$",
        r"。$",
        r"\.$",
        r",$",
        r"，$",
        r";$",
        r"；$",
    ]

    for pattern in suffix_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()

    text = text.strip(" \t\r\n:：,，.。?？!！;；、()（）[]【】{}<>《》\"'“”‘’")

    return text


def _strip_relation_suffix(text: str) -> str:
    """
    裁掉 mention 中混入的关系后缀。

    示例：
        Barack Obama 的妻子      -> Barack Obama
        Michelle Obama 的职业    -> Michelle Obama
        United States 的首都     -> United States
    """

    text = str(text or "").strip()

    if not text:
        return ""

    suffix_patterns = [
        r"(?P<entity>.+?)的(?:妻子|老婆|夫人|太太|丈夫|老公|先生|配偶)$",
        r"(?P<entity>.+?)的(?:职业|工作|职位|职务|身份|国籍|出生地|首都|作者|创始人|所属国家)$",
        r"(?P<entity>.+?)(?:出生在哪里|在哪里出生|出生于哪里|来自哪里)$",
        r"(?P<entity>.+?)(?:是哪个国家的人|是哪国人|属于哪个国家)$",
        r"(?P<entity>.+?)(?:相关的对象|相关内容|相关信息)$",
        r"(?P<entity>.+?)['’]s\s+(?:wife|husband|spouse|occupation|job|profession|nationality|birthplace|birth\s+place)$",
        r"(?:the\s+)?(?:wife|husband|spouse)\s+of\s+(?P<entity>.+)$",
    ]

    for pattern in suffix_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)

        if match:
            candidate = match.group("entity").strip()
            candidate = candidate.strip(" \t\r\n:：,，.。?？!！;；、()（）[]【】{}<>《》\"'“”‘’")

            if candidate and candidate != text:
                return candidate

    return text


def _has_relation_suffix(text: str) -> bool:
    """
    判断文本是否明显带有关系后缀。
    """

    text = str(text or "").strip()

    if not text:
        return False

    relation_suffixes = [
        "的妻子",
        "的老婆",
        "的夫人",
        "的太太",
        "的丈夫",
        "的老公",
        "的先生",
        "的配偶",
        "的职业",
        "的工作",
        "的职位",
        "的职务",
        "的身份",
        "的国籍",
        "的出生地",
        "的首都",
        "的作者",
        "的创始人",
        "的所属国家",
    ]

    if any(text.endswith(suffix) for suffix in relation_suffixes):
        return True

    lower = text.lower()

    english_suffixes = [
        "'s wife",
        "'s husband",
        "'s spouse",
        "'s occupation",
        "'s job",
        "'s profession",
        "'s nationality",
        "'s birthplace",
        "'s birth place",
    ]

    return any(lower.endswith(suffix) for suffix in english_suffixes)


def _is_valid_mention_text(text: str) -> bool:
    """
    判断候选 mention 是否有效。
    """

    text = str(text or "").strip()

    if not text:
        return False

    if len(text) > 80:
        return False

    if len(text) <= 1 and not re.match(r"[A-Z]", text):
        return False

    if text.lower() in _STOP_MENTIONS:
        return False

    if text in _STOP_MENTIONS:
        return False

    if re.fullmatch(r"[\W_]+", text):
        return False

    # 纯数字通常不是实体 mention
    if re.fullmatch(r"\d+", text):
        return False

    # 明显的问题短语不是实体
    if _is_question_phrase(text):
        return False

    # 只有关系词，不是实体
    if _is_relation_only_text(text):
        return False

    return True


def _is_question_phrase(text: str) -> bool:
    """
    判断是否是疑问短语。
    """

    text = str(text or "").strip().lower()

    if not text:
        return True

    question_words = [
        "谁",
        "什么",
        "哪里",
        "哪个",
        "哪些",
        "怎么",
        "如何",
        "为什么",
        "who",
        "what",
        "where",
        "which",
        "why",
        "how",
    ]

    return text in question_words


def _is_relation_only_text(text: str) -> bool:
    """
    判断是否只有关系词。
    """

    text = str(text or "").strip().lower()

    relation_words = {
        "妻子",
        "老婆",
        "夫人",
        "太太",
        "丈夫",
        "老公",
        "先生",
        "配偶",
        "职业",
        "工作",
        "职位",
        "职务",
        "身份",
        "国籍",
        "出生地",
        "首都",
        "作者",
        "创始人",
        "所属国家",
        "关系",
        "路径",
        "wife",
        "husband",
        "spouse",
        "occupation",
        "job",
        "profession",
        "nationality",
        "birthplace",
        "capital",
        "relationship",
        "relation",
        "path",
    }

    return text in relation_words


def _looks_like_entity_name(text: str) -> bool:
    """
    判断文本是否像实体名称。
    """

    text = str(text or "").strip()

    if not text:
        return False

    if re.search(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+", text):
        return True

    if re.search(r"[A-Z]{2,}", text):
        return True

    if re.search(r"[\u4e00-\u9fff]{2,}", text):
        return True

    if re.search(r"[A-Za-z].*[\u4e00-\u9fff]|[\u4e00-\u9fff].*[A-Za-z]", text):
        return True

    return False


_STOP_MENTIONS = {
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
    "妻子",
    "丈夫",
    "配偶",
    "职业",
    "出生地",
    "国籍",
    "首都",
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
    "wife",
    "husband",
    "spouse",
    "occupation",
    "job",
    "profession",
}


# =========================================================
# 7. 工具函数
# =========================================================

def _find_span(query: str, text: str) -> Tuple[int, int]:
    """
    在 query 中查找 text 的位置。
    """

    if not query or not text:
        return -1, -1

    start = query.find(text)

    if start >= 0:
        return start, start + len(text)

    # 大小写不敏感再查一次
    lower_start = query.lower().find(text.lower())

    if lower_start >= 0:
        return lower_start, lower_start + len(text)

    # 忽略空格再尝试一次
    compact_query = re.sub(r"\s+", "", query)
    compact_text = re.sub(r"\s+", "", text)

    compact_start = compact_query.lower().find(compact_text.lower())

    if compact_start < 0:
        return -1, -1

    # 将 compact index 映射到原始 index
    original_index = 0
    compact_index = 0
    start_original = -1
    end_original = -1

    while original_index < len(query):
        if not query[original_index].isspace():
            if compact_index == compact_start:
                start_original = original_index

            if compact_index == compact_start + len(compact_text) - 1:
                end_original = original_index + 1
                break

            compact_index += 1

        original_index += 1

    if start_original >= 0 and end_original > start_original:
        return start_original, end_original

    return -1, -1


def _extract_json_array(text: str) -> str:
    """
    从 LLM 输出中提取 JSON 数组。
    """

    if not text:
        return ""

    match = re.search(r"\[.*\]", text, re.DOTALL)

    if not match:
        return ""

    return match.group()


def _normalize_for_dedup(text: str) -> str:
    """
    去重用归一化。
    """

    text = str(text or "").strip().lower()
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" \t\r\n:：,，.。?？!！;；、()（）[]【】{}<>《》\"'“”‘’")

    return text


def _spans_overlap(
    start1: int,
    end1: int,
    start2: int,
    end2: int,
) -> bool:
    """
    判断两个 span 是否重叠。
    """

    if start1 < 0 or end1 <= start1:
        return False

    if start2 < 0 or end2 <= start2:
        return False

    return max(start1, start2) < min(end1, end2)


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

        return float(value)

    except Exception:
        return default


