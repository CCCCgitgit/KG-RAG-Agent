# -*- coding: utf-8 -*-
"""会话窗口与结构化摘要。"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional

from .retriever import estimate_tokens, trim_text_to_token_budget


class ConversationSummarizer:
    """默认使用确定性摘要，避免 Memory 模块依赖在线 LLM。"""

    def __init__(self, *, llm_client: Any = None) -> None:
        self.llm_client = llm_client

    def summarize(
        self,
        messages: Iterable[Mapping[str, Any]],
        *,
        token_budget: int,
        existing_summary: str = "",
        use_llm: bool = False,
    ) -> str:
        normalized = _normalize_messages(messages)
        if not normalized:
            return trim_text_to_token_budget(existing_summary, token_budget)
        if use_llm and self.llm_client is not None:
            summary = self._summarize_with_llm(
                normalized,
                existing_summary=existing_summary,
                token_budget=token_budget,
            )
            if summary:
                return trim_text_to_token_budget(summary, token_budget)
        return _deterministic_summary(
            normalized,
            existing_summary=existing_summary,
            token_budget=token_budget,
        )

    def _summarize_with_llm(
        self,
        messages: list[dict[str, str]],
        *,
        existing_summary: str,
        token_budget: int,
    ) -> str:
        prompt = (
            "请把以下对话压缩为结构化项目记忆，只保留：当前目标、已确认约束、"
            "已完成步骤、未解决事项、关键实体。不要添加未出现的信息。\n\n"
            f"已有摘要：\n{existing_summary or '无'}\n\n"
            f"新消息：\n{_messages_to_text(messages)}"
        )
        try:
            answer = self.llm_client.chat(
                prompt=prompt,
                system_prompt="你是严格的对话摘要器。",
                temperature=0.0,
                max_tokens=max(128, int(token_budget)),
            )
        except Exception:
            return ""
        return str(answer or "").strip()


def _normalize_messages(messages: Iterable[Mapping[str, Any]]) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for item in messages:
        if not isinstance(item, Mapping):
            continue
        role = str(item.get("role") or "unknown").strip()
        content = str(item.get("content") or "").strip()
        if content:
            result.append({"role": role, "content": content})
    return result


def _messages_to_text(messages: list[dict[str, str]]) -> str:
    return "\n".join(f"{item['role']}: {item['content']}" for item in messages)


def _deterministic_summary(
    messages: list[dict[str, str]],
    *,
    existing_summary: str,
    token_budget: int,
) -> str:
    lines: list[str] = []
    if existing_summary.strip():
        lines.append("已有上下文：")
        lines.append(existing_summary.strip())
    lines.append("最近对话要点：")
    for item in messages[-12:]:
        role = "用户" if item["role"] == "user" else "助手" if item["role"] == "assistant" else item["role"]
        content = trim_text_to_token_budget(item["content"], 180)
        lines.append(f"- {role}: {content}")
    return trim_text_to_token_budget("\n".join(lines), token_budget)


__all__ = ["ConversationSummarizer"]
