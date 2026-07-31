"""Type definitions for Flaxon AI."""

from typing import Optional, List, Dict, Any, Union, Literal
from dataclasses import dataclass, field


# Provider types
ProviderType = Literal["gemini", "openai", "flax"]


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    
    model: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    stop: Optional[List[str]] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in self.__dict__.items() if v is not None}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GenerationConfig":
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class AIMessage:
    """Chat message format."""
    
    role: str  # "system", "user", "assistant"
    content: str
    name: Optional[str] = None
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary."""
        result = {"role": self.role, "content": self.content}
        if self.name:
            result["name"] = self.name
        return result


@dataclass
class AIResponse:
    """AI response format."""
    
    text: str
    model: Optional[str] = None
    usage: Optional[Dict[str, int]] = None
    finish_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {"text": self.text}
        if self.model:
            result["model"] = self.model
        if self.usage:
            result["usage"] = self.usage
        if self.finish_reason:
            result["finish_reason"] = self.finish_reason
        if self.metadata:
            result["metadata"] = self.metadata
        return result


@dataclass
class StreamChunk:
    """Streaming response chunk."""
    
    text: str
    index: int = 0
    is_final: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "text": self.text,
            "index": self.index,
            "is_final": self.is_final,
        }
        if self.metadata:
            result["metadata"] = self.metadata
        return result


@dataclass
class ModelInfo:
    """Model information."""
    
    name: str
    provider: str
    version: Optional[str] = None
    description: Optional[str] = None
    max_tokens: Optional[int] = None
    is_cached: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


# Type aliases
Prompt = str
Messages = List[Dict[str, str]]
Embedding = List[float]
ModelName = str
ProviderName = str