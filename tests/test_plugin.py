"""Tests for FlaxonAIPlugin."""

import os
import pytest
from unittest.mock import Mock, AsyncMock, patch

from flaxon import Flaxon
from flaxon_ai import FlaxonAIPlugin
from flaxon_ai.plugin import FlaxonAIPlugin


class TestFlaxonAIPlugin:
    """Test FlaxonAIPlugin class."""

    def test_plugin_initialization(self):
        """Test basic plugin initialization."""
        plugin = FlaxonAIPlugin(
            provider="gemini",
            api_key="test-key",
            default_model="gemini-2.5-flash",
        )

        assert plugin.provider == "gemini"
        assert plugin.api_key == "test-key"
        assert plugin.default_model == "gemini-2.5-flash"
        assert plugin.max_tokens == 100
        assert plugin.temperature == 0.7
        assert plugin.streaming_enabled is True
        assert plugin.name == "ai"
        assert plugin.version == "0.1.0"

    def test_plugin_from_env(self, monkeypatch):
        """Test plugin loads from environment variables."""
        monkeypatch.setenv("AI_API_KEY", "env-api-key")
        
        plugin = FlaxonAIPlugin(
            provider="openai",
            default_model="gpt-4o-mini",
        )

        assert plugin.api_key == "env-api-key"
        assert plugin.provider == "openai"
        assert plugin.default_model == "gpt-4o-mini"

    def test_plugin_from_config(self):
        """Test plugin from Flaxon config."""
        config = {
            "AI_PROVIDER": "openai",
            "AI_API_KEY": "config-key",
            "AI_DEFAULT_MODEL": "gpt-4",
            "AI_MAX_TOKENS": 200,
            "AI_TEMPERATURE": 0.9,
            "AI_STREAMING_ENABLED": False,
            "AI_CACHE_MODELS": False,
        }

        plugin = FlaxonAIPlugin.from_config(config)

        assert plugin.provider == "openai"
        assert plugin.api_key == "config-key"
        assert plugin.default_model == "gpt-4"
        assert plugin.max_tokens == 200
        assert plugin.temperature == 0.9
        assert plugin.streaming_enabled is False
        assert plugin.cache_models is False

    def test_plugin_defaults(self):
        """Test plugin default values."""
        plugin = FlaxonAIPlugin()

        assert plugin.provider == "gemini"
        assert plugin.api_key is None
        assert plugin.default_model == "gemini-2.5-flash"
        assert plugin.max_tokens == 100
        assert plugin.temperature == 0.7
        assert plugin.streaming_enabled is True

    def test_plugin_missing_api_key(self):
        """Test plugin raises error when API key is missing."""
        with pytest.raises(ValueError) as excinfo:
            FlaxonAIPlugin(
                provider="gemini",
                api_key=None,
            )
        
        assert "API key is required" in str(excinfo.value)

    def test_plugin_has_name_and_version(self):
        """Test plugin has name and version attributes."""
        plugin = FlaxonAIPlugin(api_key="test-key")

        assert plugin.name == "ai"
        assert plugin.version == "0.1.0"

    @pytest.mark.asyncio
    async def test_plugin_setup(self):
        """Test plugin setup."""
        app = Flaxon("test-app")
        plugin = FlaxonAIPlugin(api_key="test-key")

        plugin.setup(app)

        assert hasattr(app.state, "ai")
        assert app.state.ai is plugin
        assert plugin._app is app

    @pytest.mark.asyncio
    async def test_plugin_generate(self):
        """Test generate method."""
        app = Flaxon("test-app")
        plugin = FlaxonAIPlugin(api_key="test-key")
        plugin.setup(app)
        
        # Mock the provider
        plugin._provider = Mock()
        plugin._provider.generate = AsyncMock(return_value="Generated text")

        result = await plugin.generate("Test prompt")

        assert result == "Generated text"
        plugin._provider.generate.assert_called_once_with(
            prompt="Test prompt",
            model="gemini-2.5-flash",
            max_tokens=100,
            temperature=0.7,
        )

    @pytest.mark.asyncio
    async def test_plugin_generate_with_custom_params(self):
        """Test generate with custom parameters."""
        app = Flaxon("test-app")
        plugin = FlaxonAIPlugin(api_key="test-key")
        plugin.setup(app)
        
        # Mock the provider
        plugin._provider = Mock()
        plugin._provider.generate = AsyncMock(return_value="Generated text")

        result = await plugin.generate(
            prompt="Test prompt",
            model="custom-model",
            max_tokens=200,
            temperature=0.5,
        )

        assert result == "Generated text"
        plugin._provider.generate.assert_called_once_with(
            prompt="Test prompt",
            model="custom-model",
            max_tokens=200,
            temperature=0.5,
        )

    @pytest.mark.asyncio
    async def test_plugin_stream(self):
        """Test stream method."""
        app = Flaxon("test-app")
        plugin = FlaxonAIPlugin(api_key="test-key")
        plugin.setup(app)
        
        # Mock the provider
        plugin._provider = Mock()
        plugin._provider.stream = AsyncMock(return_value=["chunk1", "chunk2"])

        chunks = []
        async for chunk in plugin.stream("Test prompt"):
            chunks.append(chunk)

        assert chunks == ["chunk1", "chunk2"]

    @pytest.mark.asyncio
    async def test_plugin_stream_disabled(self):
        """Test stream when disabled falls back to generate."""
        app = Flaxon("test-app")
        plugin = FlaxonAIPlugin(
            api_key="test-key",
            streaming_enabled=False,
        )
        plugin.setup(app)
        
        # Mock the provider
        plugin._provider = Mock()
        plugin._provider.generate = AsyncMock(return_value="Full response")

        chunks = []
        async for chunk in plugin.stream("Test prompt"):
            chunks.append(chunk)

        assert chunks == ["Full response"]

    @pytest.mark.asyncio
    async def test_plugin_chat(self):
        """Test chat method."""
        app = Flaxon("test-app")
        plugin = FlaxonAIPlugin(api_key="test-key")
        plugin.setup(app)
        
        # Mock the provider
        plugin._provider = Mock()
        plugin._provider.chat = AsyncMock(return_value="Chat response")

        messages = [{"role": "user", "content": "Hello"}]
        result = await plugin.chat(messages)

        assert result == "Chat response"
        plugin._provider.chat.assert_called_once_with(
            messages=messages,
            model="gemini-2.5-flash",
            max_tokens=100,
            temperature=0.7,
        )

    @pytest.mark.asyncio
    async def test_plugin_embed(self):
        """Test embed method."""
        app = Flaxon("test-app")
        plugin = FlaxonAIPlugin(api_key="test-key")
        plugin.setup(app)
        
        # Mock the provider
        plugin._provider = Mock()
        plugin._provider.embed = AsyncMock(return_value=[0.1, 0.2, 0.3])

        result = await plugin.embed("Test text")

        assert result == [0.1, 0.2, 0.3]
        plugin._provider.embed.assert_called_once_with(
            text="Test text",
            model="gemini-2.5-flash",
        )

    @pytest.mark.asyncio
    async def test_plugin_list_models(self):
        """Test list_models method."""
        app = Flaxon("test-app")
        plugin = FlaxonAIPlugin(api_key="test-key")
        plugin.setup(app)
        
        # Mock the provider
        plugin._provider = Mock()
        plugin._provider.list_models = AsyncMock(return_value=["model1", "model2"])

        result = await plugin.list_models()

        assert result == ["model1", "model2"]

    @pytest.mark.asyncio
    async def test_plugin_health_check(self):
        """Test health_check method."""
        app = Flaxon("test-app")
        plugin = FlaxonAIPlugin(api_key="test-key")
        plugin.setup(app)
        
        # Mock the provider
        plugin._provider = Mock()
        plugin._provider.health_check = AsyncMock(return_value={"status": "healthy"})

        result = await plugin.health_check()

        assert result == {"status": "healthy"}