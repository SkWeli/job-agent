"""Model factory. All supported providers are free to use:

  ollama        local, unlimited, private   (e.g. qwen3:8b, qwen3.6:27b, gemma4:e4b)
  groq          free cloud tier, rate-limited, no card   (e.g. llama-3.3-70b-versatile)
  google_genai  Gemini free tier, tighter limits; free-tier data may be used by Google

Any other LangChain provider works too, through init_chat_model.
"""
from __future__ import annotations

from langchain.chat_models import init_chat_model
from langchain_core.rate_limiters import InMemoryRateLimiter

KEY_ENV = {"groq": "GROQ_API_KEY", "google_genai": "GOOGLE_API_KEY"}


def make_model(llm_cfg: dict, override_model: str | None = None):
    provider = llm_cfg["provider"]
    rpm = llm_cfg.get("requests_per_minute")
    limiter = InMemoryRateLimiter(requests_per_second=rpm / 60, max_bucket_size=1) if rpm else None
    kwargs = {"temperature": llm_cfg.get("temperature", 0), "rate_limiter": limiter, "max_retries": 6}
    if provider == "ollama":
        kwargs.pop("max_retries")
        kwargs["num_ctx"] = llm_cfg.get("num_ctx", 32768)  # Ollama's default context is small
        if llm_cfg.get("base_url"):
            kwargs["base_url"] = llm_cfg["base_url"]
    return init_chat_model(override_model or llm_cfg["model"], model_provider=provider, **kwargs)
