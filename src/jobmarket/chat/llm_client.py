"""Shared OpenAI-compatible client for the chat feature — same construction as
`cv/llm_extraction.py::_client()`, kept in one place so the router's classification
call and the RAG answerer's grounded-generation call don't each reinvent it.
"""

from __future__ import annotations

from openai import OpenAI

from jobmarket.config import get_settings


class ChatLlmError(RuntimeError):
    """Raised when a chat-feature LLM call (routing or grounded generation) fails."""


def build_llm_client() -> OpenAI:
    settings = get_settings()
    if not settings.llm_api_key:
        raise ChatLlmError("LLM_API_KEY is not set (see .env.example)")
    return OpenAI(api_key=settings.llm_api_key, base_url=settings.llm_base_url)


__all__ = ["ChatLlmError", "build_llm_client"]
