"""Pre-built AI endpoints for Flaxon AI."""

import json
from typing import Dict, Any

from flaxon import Flaxon
from flaxon.http import Request, Response, JSONResponse
from flaxon.exceptions import HTTPException

from .streaming import StreamResponse
from .types import GenerationConfig


def register_routes(app: Flaxon, plugin) -> None:
    """
    Register pre-built AI routes with the Flaxon application.
    
    Routes:
        - POST /ai/generate - Generate text
        - POST /ai/stream - Stream text via SSE
        - POST /ai/chat - Chat completion
        - POST /ai/embed - Generate embeddings
        - GET /ai/models - List available models
        - GET /ai/health - Health check
    """
    
    @app.post("/ai/generate", name="ai_generate")
    async def generate_endpoint(request: Request) -> Response:
        """Generate text completion."""
        try:
            data = await request.json()
        except:
            data = {}
        
        prompt = data.get("prompt")
        if not prompt:
            return JSONResponse(
                {"error": "prompt is required"},
                status=400
            )
        
        config = GenerationConfig(
            model=data.get("model"),
            max_tokens=data.get("max_tokens"),
            temperature=data.get("temperature"),
        )
        
        try:
            result = await plugin.generate(
                prompt=prompt,
                model=config.model,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
            )
            
            return JSONResponse({
                "success": True,
                "text": result,
                "config": config.to_dict(),
            })
        except Exception as e:
            return JSONResponse({
                "success": False,
                "error": str(e),
            }, status=500)
    
    @app.post("/ai/stream", name="ai_stream")
    async def stream_endpoint(request: Request) -> Response:
        """Stream text via Server-Sent Events."""
        try:
            data = await request.json()
        except:
            data = {}
        
        prompt = data.get("prompt")
        if not prompt:
            return JSONResponse(
                {"error": "prompt is required"},
                status=400
            )
        
        config = GenerationConfig(
            model=data.get("model"),
            max_tokens=data.get("max_tokens"),
            temperature=data.get("temperature"),
        )
        
        # Return streaming response
        return StreamResponse(
            plugin.stream(
                prompt=prompt,
                model=config.model,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
            ),
            metadata={
                "prompt": prompt,
                "config": config.to_dict(),
            }
        )
    
    @app.post("/ai/chat", name="ai_chat")
    async def chat_endpoint(request: Request) -> Response:
        """Chat completion."""
        try:
            data = await request.json()
        except:
            data = {}
        
        messages = data.get("messages")
        if not messages:
            return JSONResponse(
                {"error": "messages is required"},
                status=400
            )
        
        config = GenerationConfig(
            model=data.get("model"),
            max_tokens=data.get("max_tokens"),
            temperature=data.get("temperature"),
        )
        
        try:
            result = await plugin.chat(
                messages=messages,
                model=config.model,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
            )
            
            return JSONResponse({
                "success": True,
                "text": result,
                "config": config.to_dict(),
            })
        except Exception as e:
            return JSONResponse({
                "success": False,
                "error": str(e),
            }, status=500)
    
    @app.post("/ai/embed", name="ai_embed")
    async def embed_endpoint(request: Request) -> Response:
        """Generate embeddings."""
        try:
            data = await request.json()
        except:
            data = {}
        
        text = data.get("text")
        if not text:
            return JSONResponse(
                {"error": "text is required"},
                status=400
            )
        
        model = data.get("model")
        
        try:
            result = await plugin.embed(
                text=text,
                model=model,
            )
            
            return JSONResponse({
                "success": True,
                "embedding": result,
                "dimensions": len(result),
            })
        except Exception as e:
            return JSONResponse({
                "success": False,
                "error": str(e),
            }, status=500)
    
    @app.get("/ai/models", name="ai_models")
    async def models_endpoint(request: Request) -> Response:
        """List available models."""
        try:
            models = await plugin.list_models()
            
            return JSONResponse({
                "success": True,
                "models": models,
                "count": len(models),
                "default": plugin.default_model,
                "provider": plugin.provider,
            })
        except Exception as e:
            return JSONResponse({
                "success": False,
                "error": str(e),
            }, status=500)
    
    @app.get("/ai/health", name="ai_health")
    async def health_endpoint(request: Request) -> Response:
        """AI service health check."""
        status = await plugin.health_check()
        
        return JSONResponse({
            "success": True,
            **status,
            "version": plugin.version,
            "streaming_enabled": plugin.streaming_enabled,
        })