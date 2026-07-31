"""Integration tests for Flaxon AI plugin."""

import pytest
import json
from unittest.mock import Mock, AsyncMock, patch

from flaxon import Flaxon
from flaxon.testing import TestClient

from flaxon_ai import FlaxonAIPlugin
from flaxon_ai.router import register_routes


class TestFlaxonAIIntegration:
    """Integration tests for Flaxon AI plugin."""

    def setup_method(self):
        """Setup test app."""
        self.app = Flaxon("test-app")
        self.plugin = FlaxonAIPlugin(
            api_key="test-key",
            default_model="gemini-2.5-flash",
        )
        self.plugin.setup(self.app)
        self.client = TestClient(self.app)

    @pytest.mark.asyncio
    async def test_plugin_loads_correctly(self):
        """Test plugin loads correctly with app."""
        assert hasattr(self.app.state, "ai")
        assert self.app.state.ai is self.plugin

    @pytest.mark.asyncio
    async def test_generate_endpoint_success(self):
        """Test successful generate endpoint."""
        # Mock the generate method
        self.plugin.generate = AsyncMock(return_value="Generated response")

        response = self.client.post(
            "/ai/generate",
            json={"prompt": "Test prompt", "max_tokens": 50}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["text"] == "Generated response"
        assert data["config"]["max_tokens"] == 50

    @pytest.mark.asyncio
    async def test_generate_endpoint_missing_prompt(self):
        """Test generate endpoint with missing prompt."""
        response = self.client.post(
            "/ai/generate",
            json={}
        )

        assert response.status_code == 400
        data = response.json()
        assert data["error"] == "prompt is required"

    @pytest.mark.asyncio
    async def test_generate_endpoint_error(self):
        """Test generate endpoint with error."""
        # Mock the generate method to raise an error
        self.plugin.generate = AsyncMock(side_effect=Exception("Test error"))

        response = self.client.post(
            "/ai/generate",
            json={"prompt": "Test prompt"}
        )

        assert response.status_code == 500
        data = response.json()
        assert data["success"] is False
        assert data["error"] == "Test error"

    @pytest.mark.asyncio
    async def test_chat_endpoint_success(self):
        """Test successful chat endpoint."""
        # Mock the chat method
        self.plugin.chat = AsyncMock(return_value="Chat response")

        messages = [{"role": "user", "content": "Hello"}]
        response = self.client.post(
            "/ai/chat",
            json={"messages": messages}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["text"] == "Chat response"

    @pytest.mark.asyncio
    async def test_chat_endpoint_missing_messages(self):
        """Test chat endpoint with missing messages."""
        response = self.client.post(
            "/ai/chat",
            json={}
        )

        assert response.status_code == 400
        data = response.json()
        assert data["error"] == "messages is required"

    @pytest.mark.asyncio
    async def test_embed_endpoint_success(self):
        """Test successful embed endpoint."""
        # Mock the embed method
        self.plugin.embed = AsyncMock(return_value=[0.1, 0.2, 0.3])

        response = self.client.post(
            "/ai/embed",
            json={"text": "Test text"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["embedding"] == [0.1, 0.2, 0.3]
        assert data["dimensions"] == 3

    @pytest.mark.asyncio
    async def test_embed_endpoint_missing_text(self):
        """Test embed endpoint with missing text."""
        response = self.client.post(
            "/ai/embed",
            json={}
        )

        assert response.status_code == 400
        data = response.json()
        assert data["error"] == "text is required"

    @pytest.mark.asyncio
    async def test_models_endpoint(self):
        """Test models endpoint."""
        # Mock the list_models method
        self.plugin.list_models = AsyncMock(return_value=["model1", "model2"])

        response = self.client.get("/ai/models")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["models"] == ["model1", "model2"]
        assert data["count"] == 2
        assert data["default"] == "gemini-2.5-flash"
        assert data["provider"] == "gemini"

    @pytest.mark.asyncio
    async def test_health_endpoint(self):
        """Test health endpoint."""
        # Mock the health_check method
        self.plugin.health_check = AsyncMock(return_value={"status": "healthy"})

        response = self.client.get("/ai/health")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["status"] == "healthy"
        assert data["version"] == "0.1.0"
        assert data["streaming_enabled"] is True

    @pytest.mark.asyncio
    async def test_plugin_with_custom_config(self):
        """Test plugin with custom configuration."""
        app = Flaxon("test-app")
        plugin = FlaxonAIPlugin(
            provider="openai",
            api_key="test-key",
            default_model="gpt-4",
            max_tokens=200,
            temperature=0.5,
        )
        plugin.setup(app)
        client = TestClient(app)

        # Mock the generate method
        plugin.generate = AsyncMock(return_value="Custom response")

        response = client.post(
            "/ai/generate",
            json={"prompt": "Test prompt"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["text"] == "Custom response"

        # Verify config was passed correctly
        plugin.generate.assert_called_once_with(
            prompt="Test prompt",
            model=None,
            max_tokens=200,
            temperature=0.5,
        )

    @pytest.mark.asyncio
    async def test_stream_endpoint(self):
        """Test stream endpoint."""
        # Mock the stream method
        async def mock_stream(*args, **kwargs):
            yield "chunk1"
            yield "chunk2"
        
        self.plugin.stream = mock_stream

        response = self.client.post(
            "/ai/stream",
            json={"prompt": "Test prompt"}
        )

        assert response.status_code == 200
        assert response.headers.get("content-type") == "text/event-stream"

    @pytest.mark.asyncio
    async def test_stream_endpoint_missing_prompt(self):
        """Test stream endpoint with missing prompt."""
        response = self.client.post(
            "/ai/stream",
            json={}
        )

        assert response.status_code == 400
        data = response.json()
        assert data["error"] == "prompt is required"

    @pytest.mark.asyncio
    async def test_ai_prompt_decorator(self):
        """Test ai_prompt decorator."""
        from flaxon_ai.decorators import ai_prompt

        @self.app.get("/test-ai")
        @ai_prompt("Summarize: {data}")
        async def test_route(request):
            return {"data": "Long text to summarize"}

        # Mock the generate method
        self.plugin.generate = AsyncMock(return_value="Summary")

        response = self.client.get("/test-ai")

        assert response.status_code == 200
        data = response.json()
        assert data["ai_response"] == "Summary"
        assert data["data"] == "Long text to summarize"

    @pytest.mark.asyncio
    async def test_plugin_without_api_key(self):
        """Test plugin without API key raises error."""
        with pytest.raises(ValueError):
            FlaxonAIPlugin(
                provider="gemini",
                api_key=None,
            )

    @pytest.mark.asyncio
    async def test_plugin_unsupported_provider(self):
        """Test plugin with unsupported provider."""
        with pytest.raises(ValueError) as excinfo:
            app = Flaxon("test-app")
            plugin = FlaxonAIPlugin(
                provider="unsupported",
                api_key="test-key",
            )
            plugin.setup(app)

        assert "Unsupported provider" in str(excinfo.value)