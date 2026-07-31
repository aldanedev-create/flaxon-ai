"""Google Gemini provider for Flaxon AI."""

import json
from typing import Optional, List, Dict, Any, AsyncGenerator

from .base import AIProvider


class GeminiProvider(AIProvider):
    """
    Google Gemini API provider.
    
    Uses google-generativeai library for async operations.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: str = "gemini-2.5-flash",
        cache_models: bool = True,
        **kwargs
    ):
        super().__init__(api_key, default_model, cache_models, **kwargs)
        self._client = None
    
    async def connect(self) -> None:
        """Initialize Gemini client."""
        if self._client is not None:
            return
        
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError(
                "google-generativeai is required for Gemini provider. "
                "Install with: pip install flaxon-ai[gemini]"
            )
        
        genai.configure(api_key=self.api_key)
        self._client = genai
    
    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """Generate text using Gemini."""
        await self.connect()
        
        model_name = model or self.default_model
        
        gen_config = {}
        if max_tokens:
            gen_config["max_output_tokens"] = max_tokens
        if temperature is not None:
            gen_config["temperature"] = temperature
        
        model_inst = self._client.GenerativeModel(
            model_name=model_name,
            generation_config=gen_config,
        )
        
        response = await model_inst.generate_content_async(prompt)
        return response.text
    
    async def stream(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream text from Gemini."""
        await self.connect()
        
        model_name = model or self.default_model
        
        gen_config = {}
        if max_tokens:
            gen_config["max_output_tokens"] = max_tokens
        if temperature is not None:
            gen_config["temperature"] = temperature
        
        model_inst = self._client.GenerativeModel(
            model_name=model_name,
            generation_config=gen_config,
        )
        
        response = await model_inst.generate_content_async(prompt, stream=True)
        
        async for chunk in response:
            if chunk.text:
                yield chunk.text
    
    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """Chat completion with Gemini."""
        await self.connect()
        
        model_name = model or self.default_model
        
        system_instruction = None
        chat_messages = []
        
        # Process roles and extract system instructions
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                system_instruction = content
            else:
                gemini_role = "model" if role == "assistant" else "user"
                chat_messages.append({
                    "role": gemini_role,
                    "parts": [content]
                })
        
        gen_config = {}
        if max_tokens:
            gen_config["max_output_tokens"] = max_tokens
        if temperature is not None:
            gen_config["temperature"] = temperature
        
        model_inst = self._client.GenerativeModel(
            model_name=model_name,
            system_instruction=system_instruction,
            generation_config=gen_config,
        )
        
        if not chat_messages:
            return ""
        
        history = chat_messages[:-1]
        last_message = chat_messages[-1]["parts"][0]
        
        chat_session = model_inst.start_chat(history=history)
        response = await chat_session.send_message_async(last_message)
        
        return response.text
    
    async def embed(
        self,
        text: str,
        model: Optional[str] = None,
        **kwargs
    ) -> List[float]:
        """Generate embeddings with Gemini."""
        await self.connect()
        
        model_name = model or "models/embedding-001"
        
        result = await self._client.embed_content_async(
            model=model_name,
            content=text,
        )
        
        return result["embedding"]
    
    async def list_models(self) -> List[str]:
        """List available Gemini models."""
        await self.connect()
        
        models = self._client.list_models()
        return [m.name for m in models if "gemini" in m.name]
    
    async def health_check(self) -> Dict[str, Any]:
        """Check Gemini API health."""
        try:
            await self.connect()
            models = await self.list_models()
            return {
                "status": "healthy",
                "provider": "gemini",
                "models_count": len(models),
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "provider": "gemini",
                "error": str(e),
            }
    
    async def close(self) -> None:
        """Close provider connections."""
        self._client = None