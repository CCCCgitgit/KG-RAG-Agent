from __future__ import annotations

import pytest
from pydantic import ValidationError

from kg_rag_agent.app.schemas.request import ChatOptions, ChatRequest
from kg_rag_agent.app.settings import APISettings


def test_chat_options_to_service_options_omits_none():
    options = ChatOptions(retrieval_top_k=5, include_citations=False)
    assert options.to_service_options() == {
        "retrieval_top_k": 5,
        "include_citations": False,
    }


def test_chat_request_rejects_unknown_fields_and_empty_query():
    with pytest.raises(ValidationError):
        ChatRequest(query="")
    with pytest.raises(ValidationError):
        ChatRequest(query="hello", unknown="value")


def test_api_settings_rejects_unsafe_cors():
    with pytest.raises(ValueError):
        APISettings(cors_origins=("*",), cors_allow_credentials=True).validate()
