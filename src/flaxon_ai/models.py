"""Model management and caching for Flaxon AI."""

import json
import os
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ModelInfo:
    """Information about an AI model."""
    
    name: str
    provider: str
    version: Optional[str] = None
    description: Optional[str] = None
    max_tokens: Optional[int] = None
    is_cached: bool = False
    cache_size: Optional[int] = None
    last_used: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ModelManager:
    """
    Manage AI models: loading, caching, and listing.
    
    Features:
        - In-memory model cache
        - Optional disk cache
        - Model metadata tracking
        - LRU eviction (planned)
    """
    
    def __init__(self, cache_dir: Optional[str] = None, max_cache_size: int = 10):
        """
        Initialize model manager.
        
        Args:
            cache_dir: Directory for disk cache
            max_cache_size: Maximum number of models to cache in memory
        """
        self.cache_dir = cache_dir or os.path.join(os.getcwd(), "models_cache")
        self.max_cache_size = max_cache_size
        self._memory_cache: Dict[str, Any] = {}
        self._metadata: Dict[str, ModelInfo] = {}
        self._usage_order: List[str] = []
        
        # Create cache directory if needed
        if cache_dir and not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
    
    def load_model(self, name: str, loader_func, force_reload: bool = False) -> Any:
        """
        Load a model, using cache if available.
        
        Args:
            name: Model name
            loader_func: Function to load the model
            force_reload: Force reload from source
            
        Returns:
            Loaded model instance
        """
        # Check cache
        if not force_reload and name in self._memory_cache:
            self._update_usage(name)
            return self._memory_cache[name]
        
        # Load from disk cache
        if not force_reload and self._load_from_disk(name):
            model = self._memory_cache.get(name)
            if model:
                self._update_usage(name)
                return model
        
        # Load from source
        model = loader_func()
        
        # Cache it
        self._cache_model(name, model)
        
        return model
    
    def _cache_model(self, name: str, model: Any) -> None:
        """Cache a model in memory."""
        # Evict if at limit
        if len(self._memory_cache) >= self.max_cache_size:
            self._evict_oldest()
        
        self._memory_cache[name] = model
        self._update_usage(name)
        
        # Update metadata
        if name in self._metadata:
            self._metadata[name].is_cached = True
            self._metadata[name].last_used = datetime.now()
        
        # Save to disk
        self._save_to_disk(name, model)
    
    def _update_usage(self, name: str) -> None:
        """Update usage order for LRU eviction."""
        if name in self._usage_order:
            self._usage_order.remove(name)
        self._usage_order.append(name)
    
    def _evict_oldest(self) -> None:
        """Evict the oldest cached model."""
        if self._usage_order:
            oldest = self._usage_order.pop(0)
            if oldest in self._memory_cache:
                del self._memory_cache[oldest]
                if oldest in self._metadata:
                    self._metadata[oldest].is_cached = False
    
    def _save_to_disk(self, name: str, model: Any) -> None:
        """Save model to disk (placeholder - implement for specific model types)."""
        # This would be implemented for specific model types
        pass
    
    def _load_from_disk(self, name: str) -> bool:
        """Load model from disk (placeholder)."""
        return False
    
    def get_cached(self, name: str) -> Optional[Any]:
        """Get a cached model by name."""
        if name in self._memory_cache:
            self._update_usage(name)
            return self._memory_cache[name]
        return None
    
    def get_metadata(self, name: str) -> Optional[ModelInfo]:
        """Get metadata for a model."""
        return self._metadata.get(name)
    
    def list_cached(self) -> List[str]:
        """List cached models."""
        return list(self._memory_cache.keys())
    
    def list_metadata(self) -> Dict[str, ModelInfo]:
        """Get all model metadata."""
        return self._metadata
    
    def register_model(
        self,
        name: str,
        provider: str,
        version: Optional[str] = None,
        description: Optional[str] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> None:
        """Register model metadata."""
        self._metadata[name] = ModelInfo(
            name=name,
            provider=provider,
            version=version,
            description=description,
            max_tokens=max_tokens,
            is_cached=name in self._memory_cache,
            metadata=kwargs,
        )
    
    def clear_cache(self, name: Optional[str] = None) -> None:
        """Clear model cache."""
        if name:
            if name in self._memory_cache:
                del self._memory_cache[name]
                if name in self._metadata:
                    self._metadata[name].is_cached = False
        else:
            self._memory_cache.clear()
            for info in self._metadata.values():
                info.is_cached = False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "cached_models": len(self._memory_cache),
            "max_cache_size": self.max_cache_size,
            "cache_dir": self.cache_dir,
            "total_metadata": len(self._metadata),
            "usage_order": self._usage_order[:10],  # Show last 10
        }