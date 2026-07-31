"""Flax/JAX local provider for Flaxon AI."""

import os
from typing import Optional, List, Dict, Any, AsyncGenerator

from .base import AIProvider


class FlaxProvider(AIProvider):
    """
    Local Flax/JAX model provider.
    
    Uses Hugging Face transformers with Flax weights for local inference.
    Supports GPU/TPU acceleration via JAX.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: str = "gpt2",
        cache_models: bool = True,
        model_dir: Optional[str] = None,
        **kwargs
    ):
        super().__init__(api_key, default_model, cache_models, **kwargs)
        self.model_dir = model_dir or os.path.join(os.getcwd(), "models")
        self._model = None
        self._tokenizer = None
        self._params = None
    
    async def connect(self) -> None:
        """Load Flax model into memory."""
        if self._model is not None:
            return
        
        try:
            import jax
            from transformers import (
                FlaxAutoModelForCausalLM,
                FlaxAutoModel,
                AutoTokenizer,
            )
        except ImportError:
            raise ImportError(
                "jax, flax, and transformers are required for Flax provider. "
                "Install with: pip install flaxon-ai[flax]"
            )
        
        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.default_model,
            cache_dir=self.model_dir,
        )
        
        # Add padding token if missing
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        
        # Load model with Flax weights
        try:
            self._model = FlaxAutoModelForCausalLM.from_pretrained(
                self.default_model,
                cache_dir=self.model_dir,
            )
            self._params = self._model.params
        except Exception:
            self._model = FlaxAutoModel.from_pretrained(
                self.default_model,
                cache_dir=self.model_dir,
            )
            self._params = self._model.params
    
    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """Generate text using local Flax model."""
        await self.connect()
        
        inputs = self._tokenizer(
            prompt,
            return_tensors="jax",
            padding=True,
            truncation=True,
            max_length=512,
        )
        
        max_new_tokens = max_tokens or 50
        
        if hasattr(self._model, "generate"):
            output_ids = self._model.generate(
                inputs["input_ids"],
                params=self._params,
                max_new_tokens=max_new_tokens,
                do_sample=temperature is not None and temperature > 0,
                temperature=temperature or 0.7,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )
            
            if hasattr(output_ids, "sequences"):
                output_ids = output_ids.sequences
            elif isinstance(output_ids, tuple):
                output_ids = output_ids[0]
            
            return self._tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        return "Flax model does not support text generation for this model type."
    
    async def stream(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream text from Flax model (simulated)."""
        result = await self.generate(prompt, model, max_tokens, temperature, **kwargs)
        
        words = result.split()
        for i, word in enumerate(words):
            yield word
            if i < len(words) - 1:
                yield " "
    
    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """Chat completion with Flax model."""
        formatted_prompt = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            formatted_prompt += f"{role}: {content}\n"
        formatted_prompt += "assistant:"
        
        return await self.generate(formatted_prompt, model, max_tokens, temperature, **kwargs)
    
    async def embed(
        self,
        text: str,
        model: Optional[str] = None,
        **kwargs
    ) -> List[float]:
        """Generate embeddings with Flax model."""
        await self.connect()
        
        import jax.numpy as jnp
        
        inputs = self._tokenizer(
            text,
            return_tensors="jax",
            padding=True,
            truncation=True,
            max_length=512,
        )
        
        outputs = self._model(
            inputs["input_ids"],
            params=self._params,
            output_hidden_states=True,
        )
        
        if hasattr(outputs, "last_hidden_state"):
            embeddings = jnp.mean(outputs.last_hidden_state, axis=1)
            return [float(x) for x in embeddings[0]]
        
        return [0.0] * 768
    
    async def list_models(self) -> List[str]:
        """List available local models."""
        models = []
        if os.path.exists(self.model_dir):
            for item in os.listdir(self.model_dir):
                if os.path.isdir(os.path.join(self.model_dir, item)):
                    if any(f.endswith(".json") for f in os.listdir(os.path.join(self.model_dir, item))):
                        models.append(item)
        
        if self.default_model not in models:
            models.insert(0, self.default_model)
        
        return models
    
    async def health_check(self) -> Dict[str, Any]:
        """Check Flax model health."""
        try:
            await self.connect()
            return {
                "status": "healthy",
                "provider": "flax",
                "model": self.default_model,
                "model_dir": self.model_dir,
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "provider": "flax",
                "error": str(e),
            }
    
    async def close(self) -> None:
        """Close provider connections (clear memory)."""
        self._model = None
        self._params = None
        self._tokenizer = None