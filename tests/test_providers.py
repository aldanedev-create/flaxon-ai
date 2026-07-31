"""Tests for AI providers."""

import pytest
from unittest.mock import Mock, AsyncMock, patch

from flaxon_ai.providers.base import AIProvider
from flaxon_ai.providers.gemini import GeminiProvider
from flaxon_ai.providers.openai import OpenAIProvider
from flaxon_ai.providers.flax import FlaxProvider


class TestAIProviderBase:
    """Test AIProvider base class."""

    def test_provider_initialization(self):
        """Test provider initialization."""
        class TestProvider(AIProvider):
            async def generate(self, *args, **kwargs):
                return "test"
            
            async def stream(self, *args, **kwargs):
                yield "test"
            
            async def chat(self, *args, **kwargs):
                return "test"
            
            async def embed(self, *args, **kwargs):
                return [0.1]
            
            async def list_models(self):
                return ["test"]
            
            async def health_check(self):
                return {"status": "healthy"}

        provider = TestProvider(
            api_key="test-key",
            default_model="test-model",
            cache_models=True,
        )

        assert provider.api_key == "test-key"
        assert provider.default_model == "test-model"
        assert provider.cache_models is True


class TestGeminiProvider:
    """Test GeminiProvider class."""

    def test_provider_initialization(self):
        """Test provider initialization."""
        provider = GeminiProvider(
            api_key="test-key",
            default_model="gemini-2.5-flash",
        )

        assert provider.api_key == "test-key"
        assert provider.default_model == "gemini-2.5-flash"
        assert provider._client is None

    @pytest.mark.asyncio
    @patch("flaxon_ai.providers.gemini.genai")
    async def test_provider_connect(self, mock_genai):
        """Test provider connection."""
        provider = GeminiProvider(api_key="test-key")
        await provider.connect()

        mock_genai.configure.assert_called_once_with(api_key="test-key")
        assert provider._client is not None

    @pytest.mark.asyncio
    @patch("flaxon_ai.providers.gemini.genai")
    async def test_provider_generate(self, mock_genai):
        """Test generate method."""
        # Mock the client
        mock_model = AsyncMock()
        mock_model.generate_content_async = AsyncMock(return_value=Mock(text="Generated text"))
        mock_genai.GenerativeModel.return_value = mock_model

        provider = GeminiProvider(api_key="test-key")
        result = await provider.generate("Test prompt")

        assert result == "Generated text"

    @pytest.mark.asyncio
    @patch("flaxon_ai.providers.gemini.genai")
    async def test_provider_stream(self, mock_genai):
        """Test stream method."""
        # Mock the client
        mock_chunk1 = Mock(text="chunk1")
        mock_chunk2 = Mock(text="chunk2")
        mock_response = AsyncMock()
        mock_response.__aiter__ = AsyncMock(return_value=iter([mock_chunk1, mock_chunk2]))
        
        mock_model = AsyncMock()
        mock_model.generate_content_async.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        provider = GeminiProvider(api_key="test-key")
        chunks = []
        async for chunk in provider.stream("Test prompt"):
            chunks.append(chunk)

        assert chunks == ["chunk1", "chunk2"]

    @pytest.mark.asyncio
    @patch("flaxon_ai.providers.gemini.genai")
    async def test_provider_health_check(self, mock_genai):
        """Test health_check method."""
        mock_genai.list_models.return_value = ["model1", "model2"]

        provider = GeminiProvider(api_key="test-key")
        await provider.connect()
        result = await provider.health_check()

        assert result["status"] == "healthy"
        assert result["provider"] == "gemini"


class TestOpenAIProvider:
    """Test OpenAIProvider class."""

    def test_provider_initialization(self):
        """Test provider initialization."""
        provider = OpenAIProvider(
            api_key="test-key",
            default_model="gpt-4o-mini",
        )

        assert provider.api_key == "test-key"
        assert provider.default_model == "gpt-4o-mini"
        assert provider._client is None

    @pytest.mark.asyncio
    @patch("flaxon_ai.providers.openai.AsyncOpenAI")
    async def test_provider_connect(self, mock_async_openai):
        """Test provider connection."""
        provider = OpenAIProvider(api_key="test-key")
        await provider.connect()

        mock_async_openai.assert_called_once_with(api_key="test-key")
        assert provider._client is not None

    @pytest.mark.asyncio
    @patch("flaxon_ai.providers.openai.AsyncOpenAI")
    async def test_provider_generate(self, mock_async_openai):
        """Test generate method."""
        # Mock the client
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Generated text"))]
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_async_openai.return_value = mock_client

        provider = OpenAIProvider(api_key="test-key")
        await provider.connect()
        result = await provider.generate("Test prompt")

        assert result == "Generated text"

    @pytest.mark.asyncio
    @patch("flaxon_ai.providers.openai.AsyncOpenAI")
    async def test_provider_health_check(self, mock_async_openai):
        """Test health_check method."""
        mock_client = AsyncMock()
        mock_client.models.list = AsyncMock(return_value=Mock(data=[]))
        mock_async_openai.return_value = mock_client

        provider = OpenAIProvider(api_key="test-key")
        await provider.connect()
        result = await provider.health_check()

        assert result["status"] == "healthy"
        assert result["provider"] == "openai"


class TestFlaxProvider:
    """Test FlaxProvider class."""

    def test_provider_initialization(self):
        """Test provider initialization."""
        provider = FlaxProvider(
            default_model="gpt2",
            cache_models=True,
        )

        assert provider.default_model == "gpt2"
        assert provider.cache_models is True
        assert provider._model is None
        assert provider._tokenizer is None
        assert provider._params is None

    @pytest.mark.asyncio
    @patch("flaxon_ai.providers.flax.FlaxAutoModelForCausalLM")
    @patch("flaxon_ai.providers.flax.AutoTokenizer")
    async def test_provider_connect(self, mock_tokenizer, mock_model):
        """Test provider connection."""
        mock_tokenizer.from_pretrained.return_value = Mock()
        mock_model.from_pretrained.return_value = Mock(params={})

        provider = FlaxProvider(default_model="gpt2")
        await provider.connect()

        assert provider._model is not None
        assert provider._tokenizer is not None
        assert provider._params is not None

    @pytest.mark.asyncio
    @patch("flaxon_ai.providers.flax.FlaxAutoModelForCausalLM")
    @patch("flaxon_ai.providers.flax.AutoTokenizer")
    async def test_provider_generate(self, mock_tokenizer, mock_model):
        """Test generate method."""
        # Mock tokenizer
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.return_value = {"input_ids": [[1, 2, 3]]}
        mock_tokenizer_instance.decode.return_value = "Generated text"
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        # Mock model
        mock_model_instance = Mock()
        mock_model_instance.generate.return_value = Mock(sequences=[[1, 2, 3, 4, 5]])
        mock_model.from_pretrained.return_value = mock_model_instance

        provider = FlaxProvider(default_model="gpt2")
        await provider.connect()
        result = await provider.generate("Test prompt")

        assert result == "Generated text"

    @pytest.mark.asyncio
    @patch("flaxon_ai.providers.flax.FlaxAutoModelForCausalLM")
    @patch("flaxon_ai.providers.flax.AutoTokenizer")
    async def test_provider_health_check(self, mock_tokenizer, mock_model):
        """Test health_check method."""
        mock_tokenizer.from_pretrained.return_value = Mock()
        mock_model.from_pretrained.return_value = Mock(params={})

        provider = FlaxProvider(default_model="gpt2")
        await provider.connect()
        result = await provider.health_check()

        assert result["status"] == "healthy"
        assert result["provider"] == "flax"
        assert result["model"] == "gpt2"