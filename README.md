# Flaxon AI


<p align="center">
  <img src="https://raw.githubusercontent.com/aldanedev-create/Flaxon-Backend-Framework/main/assets/flaxon.png" alt="flaxon Logo"
   width="200"/>
</p>


  
  <p align="center">
  <a href="https://pypi.org/project/flaxon/"><img src="https://img.shields.io/pypi/v/flaxon.svg" alt="PyPI version"></a>
  <a href="https://github.com/aldanedev-create/Flaxon-Backend-Framework/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/badge/code%20style-ruff-000000.svg" alt="Code style: ruff"></a>
</p>



**AI/LLM integration plugin for Flaxon framework** with support for Google Gemini, OpenAI, and local Flax/JAX models.

## Features

- 🤖 **Multiple AI Providers** — Google Gemini, OpenAI, Flax/JAX local models
- ⚡ **Async Generation** — Non-blocking AI completions
- 📡 **Streaming Support** — Server-Sent Events (SSE) for real-time responses
- 🧠 **Flax/JAX Integration** — Local model inference with GPU/TPU acceleration
- 🎯 **Route Decorators** — Easy AI integration into Flaxon routes
- 📊 **GraphQL Helpers** — AI field resolvers for GraphQL
- 🚀 **Pre-built Endpoints** — `/ai/generate`, `/ai/stream`, `/ai/chat`
- 💾 **Model Management** — Load and cache local models



## Installation

```bash
# Basic installation
pip install flaxon-ai

# With Google Gemini support
pip install flaxon-ai[gemini]

# With OpenAI support
pip install flaxon-ai[openai]

# With local Flax/JAX support
pip install flaxon-ai[flax]

# With all providers
pip install flaxon-ai[all]


Quick Start
python
from flaxon import Flaxon
from flaxon_ai import FlaxonAIPlugin

app = Flaxon("my-app")

# Load AI plugin with Google Gemini
app.plugins.load_plugin(FlaxonAIPlugin(
    provider="gemini",
    api_key=os.environ.get("GEMINI_API_KEY"),
    default_model="gemini-2.5-flash",
))

# Use in a route
@app.post("/api/generate")
async def generate(request):
    data = await request.json()
    result = await app.state.ai.generate(data["prompt"])
    return {"result": result}
Configuration
Environment Variables
bash
# Google Gemini
GEMINI_API_KEY=your-api-key

# OpenAI
OPENAI_API_KEY=your-api-key

# Flax/JAX (no API key required)
With Flaxon Config
python
app = Flaxon("my-app", config={
    "AI_PROVIDER": "gemini",
    "AI_API_KEY": os.environ.get("GEMINI_API_KEY"),
    "AI_DEFAULT_MODEL": "gemini-2.5-flash",
    "AI_MAX_TOKENS": 100,
    "AI_TEMPERATURE": 0.7,
})

plugin = FlaxonAIPlugin.from_config(app.config)
app.plugins.load_plugin(plugin)
Usage Examples
Basic Generation
python
@app.post("/api/summarize")
async def summarize(request):
    data = await request.json()
    text = data.get("text", "")
    
    summary = await app.state.ai.generate(
        f"Summarize this text in 2 sentences:\n{text}",
        max_tokens=100
    )
    return {"summary": summary}
Streaming Response
python
@app.get("/api/stream")
async def stream_response(request):
    from flaxon_ai import stream_response
    
    return await stream_response(
        app.state.ai,
        "Write a short story about a robot",
        model="gemini-2.5-flash"
    )
Using Decorators
python
from flaxon_ai import ai_prompt, stream_ai

@app.get("/api/smart-bio")
@ai_prompt("Write a professional bio based on: {data}")
async def get_user_data(request):
    user = await get_user(request.session.get("user_id"))
    return {"data": f"Name: {user.name}, Skills: {user.skills}"}
Local Flax Model
python
# Load local Flax model
app.plugins.load_plugin(FlaxonAIPlugin(
    provider="flax",
    default_model="gpt2",
    cache_models=True,
))

# Use it just like a cloud provider
result = await app.state.ai.generate(
    "Write a poem about Python",
    max_tokens=60
)
Chat Completion
python
@app.post("/api/chat")
async def chat(request):
    data = await request.json()
    messages = data.get("messages", [])
    
    response = await app.state.ai.chat(
        messages=messages,
        model="gemini-2.5-flash"
    )
    return {"response": response}
Embeddings
python
@app.post("/api/embed")
async def embed(request):
    data = await request.json()
    text = data.get("text", "")
    
    embedding = await app.state.ai.embed(text)
    return {"embedding": embedding}

Pre-built Routes
Route	Method	Description
/ai/generate	POST	Generate text completion
/ai/stream	POST	Stream text via SSE
/ai/chat	POST	Chat completion
/ai/embed	POST	Generate embeddings
/ai/models	GET	List available models
/ai/health	GET	Health check



Security Best Practices
✅ Never hardcode API keys

✅ Use environment variables or secrets manager

✅ Validate and sanitize user prompts

✅ Implement rate limiting for AI endpoints

✅ Limit streaming duration and token count

✅ Use HTTPS in production

✅ Verify local model integrity before loading

Roadmap
Version	Features
0.1.0	Basic AI plugin, Gemini provider
0.2.0	OpenAI provider, streaming support
0.3.0	Flax/JAX local models
0.4.0	Embeddings, model management
0.5.0	Function calling, tool use
0.6.0	Fine-tuning support

License
MIT License - See LICENSE file for details.