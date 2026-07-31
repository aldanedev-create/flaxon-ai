"""Tests for AIClient."""

import pytest
from unittest.mock import Mock, AsyncMock

from flaxon_ai.client import AIClient


class TestAIClient:
    """Test AIClient class."""

    def test_client_initialization(self):
        """Test client initialization."""
        provider = Mock()
        client = AIClient(provider)

        assert client._provider is provider

    @pytest.mark.asyncio
    async def test_client_generate(self):
        """Test generate method."""
        provider = Mock()
        provider.generate = AsyncMock(return_value="Generated text")
        
        client = AIClient(provider)
        result = await client.generate("Test prompt", max_tokens=100)

        assert result == "Generated text"
        provider.generate.assert_called_once_with(
            prompt="Test prompt",
            model=None,
            max_tokens=100,
            temperature=None,
        )

    @pytest.mark.asyncio
    async def test_client_stream(self):
        """Test stream method."""
        provider = Mock()
        provider.stream = AsyncMock(return_value=["chunk1", "chunk2"])
        
        client = AIClient(provider)
        chunks = []
        async for chunk in client.stream("Test prompt"):
            chunks.append(chunk)

        assert chunks == ["chunk1", "chunk2"]

    @pytest.mark.asyncio
    async def test_client_chat(self):
        """Test chat method."""
        provider = Mock()
        provider.chat = AsyncMock(return_value="Chat response")
        
        client = AIClient(provider)
        messages = [{"role": "user", "content": "Hello"}]
        result = await client.chat(messages)

        assert result == "Chat response"
        provider.chat.assert_called_once_with(
            messages=messages,
            model=None,
            max_tokens=None,
            temperature=None,
        )

    @pytest.mark.asyncio
    async def test_client_embed(self):
        """Test embed method."""
        provider = Mock()
        provider.embed = AsyncMock(return_value=[0.1, 0.2])
        
        client = AIClient(provider)
        result = await client.embed("Test text")

        assert result == [0.1, 0.2]
        provider.embed.assert_called_once_with(
            text="Test text",
            model=None,
        )

    @pytest.mark.asyncio
    async def test_client_list_models(self):
        """Test list_models method."""
        provider = Mock()
        provider.list_models = AsyncMock(return_value=["model1", "model2"])
        
        client = AIClient(provider)
        result = await client.list_models()

        assert result == ["model1", "model2"]

    @pytest.mark.asyncio
    async def test_client_health_check(self):
        """Test health_check method."""
        provider = Mock()
        provider.health_check = AsyncMock(return_value={"status": "healthy"})
        
        client = AIClient(provider)
        result = await client.health_check()

        assert result == {"status": "healthy"}

    @pytest.mark.asyncio
    async def test_client_provider_property(self):
        """Test provider property."""
        provider = Mock()
        client = AIClient(provider)

        assert client.provider is provider