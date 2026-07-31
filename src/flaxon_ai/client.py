"""Async LLM client for Flaxon AI."""

from typing import Optional, List, Dict, Any, AsyncGenerator

from .providers.base import AIProvider


class AIClient:
    """
    Unified async client for all AI providers.
    
    Wraps the provider implementation and provides a consistent interface.
    """
    
    def __init__(self, provider: AIProvider):
        """
        Initialize AI client.
        
        Args:
            provider: AI provider instance
        """
        self._provider = provider
    
    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """Generate text completion."""
        return await self._provider.generate(
            prompt=prompt,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )
    
    async def stream(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream text generation."""
        async for chunk in self._provider.stream(
            prompt=prompt,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        ):
            yield chunk
    
    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """Chat completion."""
        return await self._provider.chat(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )
    
    async def embed(
        self,
        text: str,
        model: Optional[str] = None,
        **kwargs
    ) -> List[float]:
        """Generate embeddings."""
        return await self._provider.embed(
            text=text,
            model=model,
            **kwargs
        )
    
    async def list_models(self) -> List[str]:
        """List available models."""
        return await self._provider.list_models()
    
    async def health_check(self) -> Dict[str, Any]:
        """Check provider health."""
        return await self._provider.health_check()
    
    @property
    def provider(self) -> AIProvider:
        """Get the underlying provider."""
        return self._provider