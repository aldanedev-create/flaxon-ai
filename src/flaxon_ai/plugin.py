"""AI plugin for Flaxon."""

import os
from typing import Optional, Dict, Any, List

from flaxon import Flaxon
from flaxon.plugins import Plugin
from flaxon.http import Request, Response

from .client import AIClient
from .providers.base import AIProvider
from .providers.gemini import GeminiProvider
from .providers.openai import OpenAIProvider
from .providers.flax import FlaxProvider
from .router import register_routes
from .types import ProviderType, GenerationConfig


class FlaxonAIPlugin(Plugin):
    """
    AI/LLM integration plugin for Flaxon.
    
    Usage:
    
        from flaxon import Flaxon
        from flaxon_ai import FlaxonAIPlugin
        
        app = Flaxon("my-app")
        
        # With Google Gemini
        app.plugins.load_plugin(FlaxonAIPlugin(
            provider="gemini",
            api_key=os.environ.get("GEMINI_API_KEY"),
            default_model="gemini-2.5-flash",
        ))
        
        # With local Flax model
        app.plugins.load_plugin(FlaxonAIPlugin(
            provider="flax",
            default_model="gpt2",
            cache_models=True,
        ))
    """
    
    name = "ai"
    version = "0.1.0"
    description = "AI/LLM integration plugin for Flaxon"
    author = "Aldane Hutchinson"
    requires = []
    
    def __init__(
        self,
        provider: ProviderType = "gemini",
        api_key: Optional[str] = None,
        default_model: str = "gemini-2.5-flash",
        cache_models: bool = True,
        max_tokens: int = 100,
        temperature: float = 0.7,
        streaming_enabled: bool = True,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize AI plugin.
        
        Args:
            provider: AI provider ("gemini", "openai", "flax")
            api_key: API key for cloud providers
            default_model: Default model to use
            cache_models: Cache loaded models in memory
            max_tokens: Default max tokens for generation
            temperature: Default temperature
            streaming_enabled: Enable streaming support
            config: Additional configuration
        """
        self.provider = provider
        self.api_key = api_key or os.environ.get("AI_API_KEY")
        self.default_model = default_model
        self.cache_models = cache_models
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.streaming_enabled = streaming_enabled
        self.config = config or {}
        
        # Initialize provider
        self._provider: Optional[AIProvider] = None
        self.client: Optional[AIClient] = None
        self._app = None
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate plugin configuration."""
        if self.provider in ("gemini", "openai"):
            if not self.api_key:
                raise ValueError(
                    f"API key is required for {self.provider}. "
                    f"Set AI_API_KEY environment variable or pass api_key."
                )
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "FlaxonAIPlugin":
        """Create FlaxonAIPlugin from Flaxon config."""
        return cls(
            provider=config.get("AI_PROVIDER", "gemini"),
            api_key=config.get("AI_API_KEY"),
            default_model=config.get("AI_DEFAULT_MODEL", "gemini-2.5-flash"),
            cache_models=config.get("AI_CACHE_MODELS", True),
            max_tokens=config.get("AI_MAX_TOKENS", 100),
            temperature=config.get("AI_TEMPERATURE", 0.7),
            streaming_enabled=config.get("AI_STREAMING_ENABLED", True),
            config=config,
        )
    
    def setup(self, app: Flaxon) -> None:
        """Setup the plugin with the Flaxon application."""
        self._app = app
        app.state.ai = self
        
        # Initialize provider
        self._init_provider()
        
        # Initialize client
        self.client = AIClient(self._provider)
        
        # Register routes
        register_routes(app, self)
    
    def _init_provider(self) -> None:
        """Initialize the AI provider."""
        providers = {
            "gemini": GeminiProvider,
            "openai": OpenAIProvider,
            "flax": FlaxProvider,
        }
        
        provider_class = providers.get(self.provider)
        if not provider_class:
            raise ValueError(f"Unsupported provider: {self.provider}")
        
        self._provider = provider_class(
            api_key=self.api_key,
            default_model=self.default_model,
            cache_models=self.cache_models,
            **self.config,
        )
    
    def on_load(self) -> None:
        """Called when plugin is loaded."""
        pass
    
    def on_unload(self) -> None:
        """Called when plugin is unloaded."""
        pass
    
    def on_startup(self) -> None:
        """Called on application startup."""
        # Provider connection is lazy
        pass
    
    def on_shutdown(self) -> None:
        """Called on application shutdown."""
        if self._provider:
            import asyncio
            try:
                asyncio.create_task(self._provider.close())
            except RuntimeError:
                pass
    
    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """
        Generate text completion.
        
        Args:
            prompt: Input prompt
            model: Model to use (overrides default)
            max_tokens: Max tokens to generate
            temperature: Temperature for generation
            **kwargs: Additional provider-specific arguments
            
        Returns:
            Generated text
        """
        if not self._provider:
            raise RuntimeError("AI provider not initialized")
        
        return await self._provider.generate(
            prompt=prompt,
            model=model or self.default_model,
            max_tokens=max_tokens or self.max_tokens,
            temperature=temperature or self.temperature,
            **kwargs
        )
    
    async def stream(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ):
        """
        Stream text generation.
        
        Args:
            prompt: Input prompt
            model: Model to use
            max_tokens: Max tokens to generate
            temperature: Temperature for generation
            **kwargs: Additional provider-specific arguments
            
        Yields:
            Text chunks
        """
        if not self._provider:
            raise RuntimeError("AI provider not initialized")
        
        if not self.streaming_enabled:
            result = await self.generate(prompt, model, max_tokens, temperature, **kwargs)
            yield result
            return
        
        async for chunk in self._provider.stream(
            prompt=prompt,
            model=model or self.default_model,
            max_tokens=max_tokens or self.max_tokens,
            temperature=temperature or self.temperature,
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
        """
        Chat completion.
        
        Args:
            messages: List of message dicts with "role" and "content"
            model: Model to use
            max_tokens: Max tokens to generate
            temperature: Temperature for generation
            **kwargs: Additional provider-specific arguments
            
        Returns:
            Assistant response
        """
        if not self._provider:
            raise RuntimeError("AI provider not initialized")
        
        return await self._provider.chat(
            messages=messages,
            model=model or self.default_model,
            max_tokens=max_tokens or self.max_tokens,
            temperature=temperature or self.temperature,
            **kwargs
        )
    
    async def embed(
        self,
        text: str,
        model: Optional[str] = None,
        **kwargs
    ) -> List[float]:
        """
        Generate embeddings.
        
        Args:
            text: Text to embed
            model: Model to use
            **kwargs: Additional provider-specific arguments
            
        Returns:
            Embedding vector
        """
        if not self._provider:
            raise RuntimeError("AI provider not initialized")
        
        return await self._provider.embed(
            text=text,
            model=model or self.default_model,
            **kwargs
        )
    
    async def list_models(self) -> List[str]:
        """
        List available models.
        
        Returns:
            List of model names
        """
        if not self._provider:
            raise RuntimeError("AI provider not initialized")
        
        return await self._provider.list_models()
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Check AI provider health.
        
        Returns:
            Health status
        """
        if not self._provider:
            return {"status": "unavailable", "provider": self.provider}
        
        return await self._provider.health_check()
    
    def get_provider(self) -> Optional[AIProvider]:
        """Get the underlying provider instance."""
        return self._provider