from __future__ import annotations

from langchain.chat_models import init_chat_model
from langchain_openai import AzureChatOpenAI
from langchain_core.language_models import BaseChatModel

from trading_agent.core.config import Settings


def _api_key_env_for_provider(settings: Settings, provider: str) -> str:
    if provider == settings.model_provider:
        return settings.model_api_key_env()
    return {
        "openai": "OPENAI_API_KEY",
        "azure_openai": "AZURE_OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google_genai": "GOOGLE_API_KEY",
    }.get(provider, "MODEL_API_KEY")


def _api_key_for_provider(settings: Settings, provider: str) -> str | None:
    if provider == "azure_openai":
        if settings.azure_openai_api_key is not None:
            return settings.azure_openai_api_key.get_secret_value()
        if settings.model_api_key is not None:
            return settings.model_api_key.get_secret_value()
        return None
    if settings.model_api_key is not None:
        return settings.model_api_key.get_secret_value()
    provider_key = {
        "openai": settings.openai_api_key,
        "anthropic": settings.anthropic_api_key,
        "google_genai": settings.google_api_key,
    }.get(provider)
    if provider_key is not None:
        return provider_key.get_secret_value()
    return None


def require_model_api_key(settings: Settings, *, provider: str | None = None) -> None:
    """Raise if the active provider needs an API key that is not configured."""
    active_provider = provider or settings.model_provider
    if active_provider != "ollama" and _api_key_for_provider(settings, active_provider) is None:
        raise RuntimeError(
            f"{_api_key_env_for_provider(settings, active_provider)} is required when TRADING_AGENT_ENABLE_LLM_SUPERVISOR=true"
        )


def model_identifier(settings: Settings) -> str:
    """Provider-prefixed model string understood by init_chat_model and deepagents."""
    return f"{settings.model_provider}:{settings.resolved_model_name()}"


def create_model(settings: Settings, *, identifier_override: str | None = None) -> BaseChatModel:
    """Build a chat model for the configured provider.

    Swapping providers can be explicit through MODEL_PROVIDER or inferred from
    provider-specific env vars. MODEL_BASE_URL points OpenAI-compatible
    providers at local servers such as vLLM or LM Studio; OLLAMA_BASE_URL does
    the same for Ollama. Azure OpenAI uses its deployment-aware endpoint fields.

    ``identifier_override`` ("model" or "provider:model") builds a different
    model on the same account; used for the cheaper REVIEW-tier model. When the
    override names a different provider, that provider is honored but the active
    provider's key/base_url are reused (so keep the quiet model on the same
    provider unless you have set up the other provider's credentials).
    """
    provider = settings.model_provider
    name = settings.resolved_model_name()
    if identifier_override:
        if ":" in identifier_override:
            override_provider, override_name = identifier_override.split(":", 1)
            provider = override_provider or provider
            name = override_name or name
        else:
            name = identifier_override
    require_model_api_key(settings, provider=provider)
    api_key = _api_key_for_provider(settings, provider)
    if provider == "azure_openai":
        if not settings.azure_openai_endpoint:
            raise RuntimeError("AZURE_OPENAI_ENDPOINT is required when MODEL_PROVIDER=azure_openai")
        if not name:
            raise RuntimeError("AZURE_OPENAI_DEPLOYMENT or MODEL_NAME is required when MODEL_PROVIDER=azure_openai")
        if not settings.azure_openai_api_version:
            raise RuntimeError("AZURE_OPENAI_API_VERSION is required when MODEL_PROVIDER=azure_openai")
        return AzureChatOpenAI(
            azure_deployment=name,
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=api_key,
            api_version=settings.azure_openai_api_version,
            # Parallel subagent fan-out can transiently exhaust the tokens-per-min
            # quota; retry 429s with backoff instead of failing the cycle.
            max_retries=6,
            # The deep-agent stack (skills/files seeded into state) sends
            # file-content blocks. Azure Chat Completions rejects those
            # ("does not support file URLs"); the Responses API accepts them and
            # matches how the OpenAI provider path behaves. Requires an api-version
            # that exposes /openai/responses (>= 2025-03-01-preview).
            use_responses_api=True,
        )
    kwargs: dict[str, object] = {}
    if api_key is not None:
        kwargs["api_key"] = api_key
    if provider == "ollama":
        kwargs["base_url"] = settings.model_base_url or settings.ollama_base_url
    elif provider == "openai":
        kwargs["base_url"] = settings.model_base_url or settings.openai_base_url
    elif settings.model_base_url:
        kwargs["base_url"] = settings.model_base_url
    if provider in {"openai", "openrouter"}:
        # OpenAI only returns usage_metadata while streaming when explicitly
        # asked; the cycle streams for observability, so without this the
        # per-cycle token/cost accounting sees zero tokens.
        kwargs["stream_usage"] = True
        # Parallel subagent fan-out can transiently exhaust the tokens-per-min
        # quota; retry 429s with backoff instead of failing the cycle.
        kwargs["max_retries"] = 6
    return init_chat_model(name, model_provider=provider, **kwargs)
