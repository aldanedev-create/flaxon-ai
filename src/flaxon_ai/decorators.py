"""Route and GraphQL AI helpers for Flaxon AI."""

import json
from functools import wraps
from typing import Any, Callable, Optional

from flaxon.http import Request, Response


def ai_prompt(template: str, model: Optional[str] = None):
    """
    Decorator to automatically process route outputs through an LLM prompt.
    
    Usage:
    
        @app.get("/summarize")
        @ai_prompt("Summarize this text: {data}")
        async def get_data(request):
            return {"data": await fetch_article()}
    
    Args:
        template: Prompt template with {key} placeholders
        model: Model to use (optional)
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Find request in args
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break
            
            # Execute original route to get data
            result = await func(*args, **kwargs)
            
            # Get AI instance from app state
            if request and hasattr(request, "app") and hasattr(request.app.state, "ai"):
                ai = request.app.state.ai
            else:
                # Fallback: try to find in kwargs
                ai = kwargs.get("ai")
            
            if not ai:
                raise RuntimeError("FlaxonAI instance not found in execution context.")
            
            # Format prompt with data
            if isinstance(result, dict):
                prompt = template.format(**result)
            else:
                prompt = template.format(data=result)
            
            # Generate AI response
            response_text = await ai.generate(prompt, model=model)
            
            # Return combined result
            if isinstance(result, dict):
                result["ai_response"] = response_text
                return result
            else:
                return {
                    "original": result,
                    "ai_response": response_text,
                }
        return wrapper
    return decorator


def stream_ai(model: Optional[str] = None, max_tokens: Optional[int] = None):
    """
    Decorator to enable streaming for an AI route.
    
    Usage:
    
        @app.get("/stream")
        @stream_ai()
        async def stream_response(request):
            return await ai.generate("Write a story...")
    
    Args:
        model: Model to use
        max_tokens: Max tokens to generate
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get AI instance
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break
            
            if request and hasattr(request, "app") and hasattr(request.app.state, "ai"):
                ai = request.app.state.ai
            else:
                ai = kwargs.get("ai")
            
            if not ai:
                raise RuntimeError("FlaxonAI instance not found.")
            
            # Get prompt from function result
            result = await func(*args, **kwargs)
            
            # Extract prompt
            if isinstance(result, dict):
                prompt = result.get("prompt", "")
            else:
                prompt = str(result)
            
            if not prompt:
                return Response.json({"error": "No prompt provided"}, status=400)
            
            # Return streaming response
            from .streaming import StreamResponse
            return StreamResponse(
                ai.stream(prompt, model=model, max_tokens=max_tokens)
            )
        return wrapper
    return decorator


def ai_route(path: str, method: str = "POST", model: Optional[str] = None):
    """
    Decorator to register an AI route with defaults.
    
    Usage:
    
        @ai_route("/ai/custom", method="GET")
        async def custom_ai(request):
            return await ai.generate("Custom prompt")
    
    Args:
        path: Route path
        method: HTTP method
        model: Default model for this route
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(request: Request):
            return await func(request)
        
        # Store route metadata
        wrapper._ai_route = {
            "path": path,
            "method": method,
            "model": model,
        }
        return wrapper
    return decorator


def graphql_ai(field_name: Optional[str] = None, model: Optional[str] = None):
    """
    Decorator for GraphQL AI field resolvers.
    
    Usage:
    
        @graphql_ai("bio")
        async def resolve_bio(parent, args, context, info):
            return await context["app"].state.ai.generate(
                f"Write a bio for {parent.get('name')}"
            )
    
    Args:
        field_name: GraphQL field name
        model: Model to use
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(parent, args, context, info):
            # Get AI instance from context
            ai = context.get("app", {}).get("state", {}).get("ai")
            if not ai:
                raise RuntimeError("FlaxonAI instance not found in GraphQL context.")
            
            # Add AI to kwargs
            kwargs = {"ai": ai, "model": model}
            
            # Execute resolver with AI available
            result = await func(parent, args, context, info)
            
            # If result is a string, treat it as a prompt
            if isinstance(result, str):
                return await ai.generate(result, model=model)
            
            return result
        return wrapper
    return decorator