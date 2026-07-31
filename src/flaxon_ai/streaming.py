"""SSE streaming helpers for Flaxon AI."""

import json
import asyncio
from typing import AsyncGenerator, Optional, Dict, Any

from flaxon.http import Response


class StreamResponse:
    """
    Server-Sent Events streaming response.
    
    Usage:
    
        return StreamResponse(
            ai.stream("Write a story..."),
            metadata={"prompt": "Write a story..."}
        )
    """
    
    def __init__(
        self,
        generator: AsyncGenerator[str, None],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize streaming response.
        
        Args:
            generator: Async generator yielding text chunks
            metadata: Metadata to send as first event
        """
        self.generator = generator
        self.metadata = metadata or {}
    
    async def __call__(self, scope, receive, send):
        """ASGI callable for streaming response."""
        # Send response headers
        headers = [
            (b"content-type", b"text/event-stream"),
            (b"cache-control", b"no-cache"),
            (b"connection", b"keep-alive"),
            (b"x-accel-buffering", b"no"),
        ]
        
        await send({
            "type": "http.response.start",
            "status": 200,
            "headers": headers,
        })
        
        # Send metadata if provided
        if self.metadata:
            await send({
                "type": "http.response.body",
                "body": f"event: metadata\ndata: {json.dumps(self.metadata)}\n\n".encode(),
                "more_body": True,
            })
        
        # Stream chunks
        try:
            chunk_count = 0
            async for chunk in self.generator:
                # Format as SSE
                event = f"event: token\ndata: {json.dumps({'text': chunk, 'index': chunk_count})}\n\n"
                await send({
                    "type": "http.response.body",
                    "body": event.encode(),
                    "more_body": True,
                })
                chunk_count += 1
                
                # Yield control to event loop
                await asyncio.sleep(0.001)
            
            # Send completion event
            complete_event = f"event: done\ndata: {json.dumps({'total_chunks': chunk_count})}\n\n"
            await send({
                "type": "http.response.body",
                "body": complete_event.encode(),
                "more_body": False,
            })
            
        except Exception as e:
            # Send error event
            error_event = f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"
            await send({
                "type": "http.response.body",
                "body": error_event.encode(),
                "more_body": False,
            })


def create_stream_response(
    generator: AsyncGenerator[str, None],
    metadata: Optional[Dict[str, Any]] = None,
) -> StreamResponse:
    """
    Create a streaming response.
    
    Args:
        generator: Async generator yielding text chunks
        metadata: Metadata to include
        
    Returns:
        StreamResponse instance
    """
    return StreamResponse(generator, metadata)


async def stream_generator_from_text(text: str, chunk_size: int = 5) -> AsyncGenerator[str, None]:
    """
    Convert text to a streaming generator.
    
    Args:
        text: Text to stream
        chunk_size: Number of words per chunk
        
    Yields:
        Text chunks
    """
    words = text.split()
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        yield chunk
        await asyncio.sleep(0.01)