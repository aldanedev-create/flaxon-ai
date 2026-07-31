"""Base AI provider interface."""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, AsyncGenerator


class AIProvider(ABC):
    """
    Abstract base class for all AI providers.
    
    Providers must implement all methods to support the full feature set.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: str = "gemini-2.5-flash",
        cache_models: bool = True,
        **kwargs
    ):
        """
        Initialize provider.
        
        Args:
            api_key: API key for cloud providers
            default_model: Default model to use
            cache_models: Cache models in memory
            **kwargs: Additional provider-specific options
        """
        self.api_key = api_key
        self.default_model = default_model
        self.cache_models = cache_models
        self.kwargs = kwargs
    
    @abstractmethod
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
            model: Model to use
            max_tokens: Max tokens to generate
            temperature: Temperature for generation
            **kwargs: Additional provider-specific arguments
            
        Returns:
            Generated text
        """
        pass
    
    @abstractmethod
    async def stream(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
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
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    async def list_models(self) -> List[str]:
        """
        List available models.
        
        Returns:
            List of model names
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """
        Check provider health.
        
        Returns:
            Health status
        """
        pass
    
    async def connect(self) -> None:
        """Connect to the provider (lazy initialization)."""
        pass
    
    async def close(self) -> None:
        """Close provider connections."""
        pass