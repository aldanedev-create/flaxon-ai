"""OpenAI provider for Flaxon AI."""

from typing import Optional, List, Dict, Any, AsyncGenerator

from .base import AIProvider


class OpenAIProvider(AIProvider):
    """
    OpenAI API provider.
    
    Uses openai library with async support.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: str = "gpt-4o-mini",
        cache_models: bool = True,
        **kwargs
    ):
        super().__init__(api_key, default_model, cache_models, **kwargs)
        self._client = None
    
    async def connect(self) -> None:
        """Initialize OpenAI client."""
        if self._client is not None:
            return
        
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError(
                "openai is required for OpenAI provider. "
                "Install with: pip install flaxon-ai[openai]"
            )
        
        self._client = AsyncOpenAI(api_key=self.api_key)
    
    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """Generate text using OpenAI."""
        await self.connect()
        
        model_name = model or self.default_model
        
        response = await self._client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens or 100,
            temperature=temperature if temperature is not None else 0.7,
            **kwargs
        )
        
        return response.choices[0].message.content or ""
    
    async def stream(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream text from OpenAI."""
        await self.connect()
        
        model_name = model or self.default_model
        
        stream_resp = await self._client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens or 100,
            temperature=temperature if temperature is not None else 0.7,
            stream=True,
            **kwargs
        )
        
        async for chunk in stream_resp:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    
    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """Chat completion with OpenAI."""
        await self.connect()
        
        model_name = model or self.default_model
        
        response = await self._client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=max_tokens or 100,
            temperature=temperature if temperature is not None else 0.7,
            **kwargs
        )
        
        return response.choices[0].message.content or ""
    
    async def embed(
        self,
        text: str,
        model: Optional[str] = None,
        **kwargs
    ) -> List[float]:
        """Generate embeddings with OpenAI."""
        await self.connect()
        
        model_name = model or "text-embedding-3-small"
        
        response = await self._client.embeddings.create(
            model=model_name,
            input=text,
            **kwargs
        )
        
        return response.data[0].embedding
    
    async def list_models(self) -> List[str]:
        """List available OpenAI models."""
        await self.connect()
        
        models = await self._client.models.list()
        return [m.id for m in models.data]
    
    async def health_check(self) -> Dict[str, Any]:
        """Check OpenAI API health."""
        try:
            await self.connect()
            await self.list_models()
            return {
                "status": "healthy",
                "provider": "openai",
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "provider": "openai",
                "error": str(e),
            }
    
    async def close(self) -> None:
        """Close provider connections."""
        if self._client:
            await self._client.close()
            self._client = None