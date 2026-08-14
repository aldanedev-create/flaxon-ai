"""Flaxon AI - AI/LLM integration plugin for Flaxon framework."""

from .plugin import FlaxonAIPlugin
from .client import AIClient
from .providers.base import AIProvider
from .decorators import ai_prompt, stream_ai, ai_route
from .router import register_routes
from .types import (
    ProviderType,
    GenerationConfig,
    AIMessage,
    AIResponse,
    StreamChunk,
    ModelInfo,
)

__all__ = [
    "FlaxonAIPlugin",
    "AIClient",
    "AIProvider",
    "ai_prompt",
    "stream_ai",
    "ai_route",
    "register_routes",
    "ProviderType",
    "GenerationConfig",
    "AIMessage",
    "AIResponse",
    "StreamChunk",
    "ModelInfo",
]

__version__ = "0.1.0"