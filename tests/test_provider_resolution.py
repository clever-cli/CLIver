"""Tests for provider class detection and base URL resolution.

Covers the full resolution chain:
1. Explicit ``api_url`` in config (model-level or provider-level) — honored exactly
2. No ``api_url``, ``provider_name`` available — class from name, URL from class defaults
3. No ``api_url``, no ``provider_name``, URL-based detection — class from URL
4. Neither name nor URL signal — fallback to OpenAIProvider
"""

from __future__ import annotations

import pytest

from cliver.provider.providers import (
    create_provider,
    detect_provider_class,
    resolve_base_url,
)
from cliver.provider.providers.anthropic import AnthropicProvider
from cliver.provider.providers.deepseek import DeepSeekProvider
from cliver.provider.providers.glm import GLMProvider
from cliver.provider.providers.minimax import MiniMaxProvider
from cliver.provider.providers.ollama import OllamaProvider
from cliver.provider.providers.openai import OpenAIProvider

# ── detect_provider_class ──────────────────────────────────────────


class TestDetectProviderClassByName:
    """Provider name → correct class via _NAME_PROVIDER_MAP."""

    @pytest.mark.parametrize(
        "name, expected_class",
        [
            # Exact canonical names
            ("deepseek", DeepSeekProvider),
            ("minimax", MiniMaxProvider),
            ("openai", OpenAIProvider),
            ("anthropic", AnthropicProvider),
            ("glm", GLMProvider),
            ("zhipu", GLMProvider),
            ("bigmodel", GLMProvider),
            ("ollama", OllamaProvider),
            # Case-insensitive
            ("DeepSeek", DeepSeekProvider),
            ("MINIMAX", MiniMaxProvider),
            ("OpenAI", OpenAIProvider),
            # Fuzzy / user-defined names (substring match)
            ("my-deepseek", DeepSeekProvider),
            ("deepseek-prod", DeepSeekProvider),
            ("deepseek/china", DeepSeekProvider),
            ("my-minimax-gateway", MiniMaxProvider),
            ("minimax-v2", MiniMaxProvider),
            ("custom-openai-endpoint", OpenAIProvider),
            ("anthropic-us", AnthropicProvider),
            ("my-zhipu-proxy", GLMProvider),
            ("bigmodel-internal", GLMProvider),
            ("glm-4-flash", GLMProvider),
            ("local-ollama", OllamaProvider),
        ],
    )
    def test_name_resolves_to_correct_class(self, name, expected_class):
        """Provider name (exact or fuzzy) detects the correct class."""
        cls = detect_provider_class("", provider_name=name)
        assert cls is expected_class, (
            f"provider_name={name!r} should resolve to {expected_class.__name__}, got {cls.__name__}"
        )

    def test_name_takes_priority_over_url(self):
        """When both name and URL are given, name wins."""
        # Name says "deepseek", URL says "openai.com" — name wins
        cls = detect_provider_class("https://api.openai.com/v1", provider_name="deepseek")
        assert cls is DeepSeekProvider

    def test_name_takes_priority_over_url_reverse(self):
        """Name says 'openai', URL says 'deepseek.com' — name still wins."""
        cls = detect_provider_class("https://api.deepseek.com/v1", provider_name="openai")
        assert cls is OpenAIProvider

    def test_zhipu_before_glm_in_ordering(self):
        """'zhipu' is listed before 'glm' so 'zhipu' provider name gets GLMProvider."""
        # Both "zhipu" and "glm" match GLMProvider; ordering ensures no confusion.
        cls = detect_provider_class("", provider_name="zhipu")
        assert cls is GLMProvider


class TestDetectProviderClassByUrl:
    """URL substring → correct class via _URL_PROVIDER_MAP."""

    @pytest.mark.parametrize(
        "url, expected_class",
        [
            ("https://api.deepseek.com/v1", DeepSeekProvider),
            ("https://api.deepseek.com/anthropic", DeepSeekProvider),
            ("https://api.minimaxi.com/v1", MiniMaxProvider),
            ("https://api.minimaxi.com/anthropic", MiniMaxProvider),
            ("https://api.openai.com/v1", OpenAIProvider),
            ("https://api.anthropic.com", AnthropicProvider),
            ("https://open.bigmodel.cn/api/paas/v4", GLMProvider),
            ("http://localhost:11434/v1", OllamaProvider),
            # Case-insensitive URL matching
            ("HTTPS://API.DEEPSEEK.COM/V1", DeepSeekProvider),
        ],
    )
    def test_url_resolves_to_correct_class(self, url, expected_class):
        cls = detect_provider_class(url)
        assert cls is expected_class, f"URL {url!r} should resolve to {expected_class.__name__}, got {cls.__name__}"


class TestDetectProviderClassFallback:
    """Unknown name + unknown URL → OpenAIProvider."""

    def test_empty_url_no_name_falls_back_to_openai(self):
        cls = detect_provider_class("")
        assert cls is OpenAIProvider

    def test_unknown_url_falls_back_to_openai(self):
        cls = detect_provider_class("https://some-random-llm.example.com/v1")
        assert cls is OpenAIProvider

    def test_unknown_name_falls_back_to_url_then_openai(self):
        """Unknown name → skip name map → try URL map → fallback to OpenAI."""
        cls = detect_provider_class("", provider_name="completely-unknown-provider")
        assert cls is OpenAIProvider


# ── resolve_base_url ───────────────────────────────────────────────


class TestResolveBaseUrl:
    """URL resolution: config → class defaults hierarchy."""

    def test_model_url_takes_highest_priority(self):
        url = resolve_base_url(
            model_api_url="https://model.example.com/v1",
            provider_api_url="https://provider.example.com/v1",
            provider_cls=DeepSeekProvider,
        )
        assert url == "https://model.example.com/v1"

    def test_provider_url_used_when_no_model_url(self):
        url = resolve_base_url(
            model_api_url=None,
            provider_api_url="https://provider.example.com/v1",
            provider_cls=DeepSeekProvider,
        )
        assert url == "https://provider.example.com/v1"

    def test_falls_back_to_protocol_specific_default(self):
        """DeepSeek has _default_base_urls with per-protocol URLs."""
        url = resolve_base_url(protocol="openai", provider_cls=DeepSeekProvider)
        assert url == "https://api.deepseek.com/v1"

    def test_falls_back_to_protocol_specific_anthropic(self):
        url = resolve_base_url(protocol="anthropic", provider_cls=DeepSeekProvider)
        assert url == "https://api.deepseek.com/anthropic"

    def test_falls_back_to_default_base_url_when_no_protocol_match(self):
        """OpenAIProvider has default_base_url but no _default_base_urls."""
        url = resolve_base_url(protocol="openai", provider_cls=OpenAIProvider)
        assert url == "https://api.openai.com/v1"

    def test_falls_back_to_default_base_url_when_protocol_not_in_map(self):
        """MiniMax has _default_base_urls for 'openai' and 'anthropic'.
        A hypothetical unknown protocol falls back to default_base_url."""
        url = resolve_base_url(protocol="unknown-proto", provider_cls=OpenAIProvider)
        assert url == "https://api.openai.com/v1"

    def test_empty_when_no_cls(self):
        url = resolve_base_url()
        assert url == ""

    def test_glm_default_urls(self):
        """GLM only has default_base_url (no _default_base_urls)."""
        url = resolve_base_url(protocol="openai", provider_cls=GLMProvider)
        assert url == "https://open.bigmodel.cn/api/paas/v4"

    def test_ollama_default_urls(self):
        url = resolve_base_url(protocol="openai", provider_cls=OllamaProvider)
        assert url == "http://localhost:11434/v1"

    def test_minimax_protocol_specific_urls(self):
        """MiniMax has different URLs for openai vs anthropic protocols."""
        url_openai = resolve_base_url(protocol="openai", provider_cls=MiniMaxProvider)
        assert url_openai == "https://api.minimaxi.com/v1"

        url_anthro = resolve_base_url(protocol="anthropic", provider_cls=MiniMaxProvider)
        assert url_anthro == "https://api.minimaxi.com/anthropic"


# ── create_provider — integration tests ───────────────────────────


class TestCreateProviderWithExplicitUrl:
    """When api_url IS set in config, it must be honored exactly."""

    def test_explicit_url_openai_protocol(self):
        p = create_provider(
            api_key="sk-test",
            base_url="https://custom.example.com/v1",
            protocol="openai",
        )
        assert p.base_url == "https://custom.example.com/v1"
        # URL detection: "openai" not in the custom URL, falls back to OpenAI
        assert isinstance(p, OpenAIProvider)

    def test_explicit_url_with_provider_name_still_honors_url(self):
        """Provider name identifies the class, but explicit URL is kept."""
        p = create_provider(
            api_key="sk-test",
            base_url="https://custom.example.com/v1",
            protocol="openai",
            provider_name="deepseek",
        )
        assert isinstance(p, DeepSeekProvider)
        assert p.base_url == "https://custom.example.com/v1"

    def test_explicit_url_overrides_class_defaults(self):
        """Even when provider class has default URLs, explicit base_url wins."""
        p = create_provider(
            api_key="sk-test",
            base_url="https://my-proxy.deepseek.example.com/v1",
            protocol="openai",
            provider_name="deepseek",
        )
        assert p.base_url == "https://my-proxy.deepseek.example.com/v1"

    def test_explicit_url_cross_protocol_anthropic(self):
        p = create_provider(
            api_key="sk-test",
            base_url="https://custom-anthropic.example.com",
            protocol="anthropic",
            provider_name="deepseek",
        )
        assert isinstance(p, DeepSeekProvider)
        assert p.base_url == "https://custom-anthropic.example.com"


class TestCreateProviderByNameNoUrl:
    """When NO api_url is set, provider_name determines class + default URL."""

    @pytest.mark.parametrize(
        "name, protocol, expected_class, expected_url",
        [
            ("deepseek", "openai", DeepSeekProvider, "https://api.deepseek.com/v1"),
            ("deepseek", "anthropic", DeepSeekProvider, "https://api.deepseek.com/anthropic"),
            ("minimax", "openai", MiniMaxProvider, "https://api.minimaxi.com/v1"),
            ("minimax", "anthropic", MiniMaxProvider, "https://api.minimaxi.com/anthropic"),
            ("openai", "openai", OpenAIProvider, "https://api.openai.com/v1"),
            ("anthropic", "anthropic", AnthropicProvider, "https://api.anthropic.com"),
            ("glm", "openai", GLMProvider, "https://open.bigmodel.cn/api/paas/v4"),
            ("zhipu", "openai", GLMProvider, "https://open.bigmodel.cn/api/paas/v4"),
            ("bigmodel", "openai", GLMProvider, "https://open.bigmodel.cn/api/paas/v4"),
            ("ollama", "openai", OllamaProvider, "http://localhost:11434/v1"),
        ],
    )
    def test_name_resolves_class_and_default_url(self, name, protocol, expected_class, expected_url):
        p = create_provider(api_key="sk-test", protocol=protocol, provider_name=name)
        assert isinstance(p, expected_class), (
            f"provider_name={name!r} proto={protocol}: expected {expected_class.__name__}, got {type(p).__name__}"
        )
        assert p.base_url == expected_url, (
            f"provider_name={name!r} proto={protocol}: expected URL {expected_url!r}, got {p.base_url!r}"
        )

    def test_fuzzy_name_gets_correct_default_url(self):
        """User names provider 'my-deepseek-prod' → DeepSeek + its default URL."""
        p = create_provider(api_key="sk-test", protocol="openai", provider_name="my-deepseek-prod")
        assert isinstance(p, DeepSeekProvider)
        assert p.base_url == "https://api.deepseek.com/v1"

    def test_fuzzy_name_minimax_gets_correct_default_url(self):
        p = create_provider(api_key="sk-test", protocol="openai", provider_name="minimax-china")
        assert isinstance(p, MiniMaxProvider)
        assert p.base_url == "https://api.minimaxi.com/v1"


class TestCreateProviderByUrlNoName:
    """When no provider_name but URL is given, URL detection determines class."""

    def test_deepseek_url_detects_class_and_keeps_url(self):
        p = create_provider(api_key="sk-test", base_url="https://api.deepseek.com/v1")
        assert isinstance(p, DeepSeekProvider)
        assert p.base_url == "https://api.deepseek.com/v1"

    def test_unknown_url_falls_back_to_openai_with_url_preserved(self):
        p = create_provider(
            api_key="sk-test",
            base_url="https://random-llm.example.com/v1",
        )
        assert isinstance(p, OpenAIProvider)
        assert p.base_url == "https://random-llm.example.com/v1"


class TestCreateProviderFallback:
    """No URL, no name, no match → OpenAIProvider with its default URL."""

    def test_completely_empty_falls_back_to_openai(self):
        p = create_provider(api_key="sk-test")
        assert isinstance(p, OpenAIProvider)
        assert p.base_url == "https://api.openai.com/v1"

    def test_unknown_name_openai_protocol(self):
        p = create_provider(
            api_key="sk-test",
            protocol="openai",
            provider_name="some-random-provider",
        )
        # Name doesn't match → URL is empty → fallback to OpenAI
        assert isinstance(p, OpenAIProvider)
        assert p.base_url == "https://api.openai.com/v1"

    def test_unknown_name_anthropic_protocol(self):
        p = create_provider(
            api_key="sk-test",
            protocol="anthropic",
            provider_name="some-random-provider",
        )
        # Falls back to OpenAI from detection, but protocol "anthropic" not in
        # OpenAIProvider.supported_protocols → switches to AnthropicProvider
        assert isinstance(p, AnthropicProvider)
        assert p.base_url == "https://api.anthropic.com"


class TestCreateProviderProtocolMismatch:
    """When the detected class doesn't support the requested protocol."""

    def test_openai_class_with_anthropic_protocol_switches(self):
        """OpenAIProvider only supports 'openai'.  Requesting 'anthropic'
        protocol should switch to AnthropicProvider."""
        p = create_provider(
            api_key="sk-test",
            protocol="anthropic",
            # No name, no URL → detects OpenAIProvider → protocol mismatch → switch
        )
        assert isinstance(p, AnthropicProvider)
        assert p.base_url == "https://api.anthropic.com"

    def test_ollama_with_anthropic_protocol_switches(self):
        """OllamaProvider only supports 'openai'.  Anthropic protocol → switch."""
        p = create_provider(
            api_key="sk-test",
            protocol="anthropic",
            provider_name="ollama",
        )
        # Name detects OllamaProvider, but it doesn't support 'anthropic'
        # → switches to AnthropicProvider
        assert isinstance(p, AnthropicProvider)
        assert p.base_url == "https://api.anthropic.com"


class TestCreateProviderExplicitClass:
    """When provider_class is explicitly passed, detection is bypassed."""

    def test_explicit_class_overrides_name_detection(self):
        p = create_provider(
            api_key="sk-test",
            protocol="openai",
            provider_class=DeepSeekProvider,
            provider_name="openai",  # would normally detect OpenAIProvider
        )
        assert isinstance(p, DeepSeekProvider)
        assert p.base_url == "https://api.deepseek.com/v1"

    def test_explicit_class_overrides_url_detection(self):
        p = create_provider(
            api_key="sk-test",
            protocol="openai",
            provider_class=GLMProvider,
            base_url="https://api.openai.com/v1",  # would normally detect OpenAI
        )
        assert isinstance(p, GLMProvider)
        assert p.base_url == "https://api.openai.com/v1"  # URL preserved


class TestCreateProviderUserAgent:
    """User-Agent is threaded through to the engine."""

    def test_user_agent_passed_to_engine(self):
        p = create_provider(
            api_key="sk-test",
            protocol="openai",
            provider_name="openai",
            user_agent="CLIver/1.0",
        )
        assert p.engine.user_agent == "CLIver/1.0"

    def test_user_agent_none_by_default(self):
        p = create_provider(api_key="sk-test", provider_name="openai")
        assert p.engine.user_agent is None


class TestAnthropicBearerAuth:
    """Third-party providers use Bearer auth (auth_token) for Anthropic protocol,
    while native Anthropic uses x-api-key (api_key)."""

    def test_deepseek_anthropic_uses_bearer_auth(self):
        p = create_provider(api_key="sk-test", protocol="anthropic", provider_name="deepseek")
        assert p.engine.client.auth_token == "sk-test"

    def test_minimax_anthropic_uses_bearer_auth(self):
        p = create_provider(api_key="sk-test", protocol="anthropic", provider_name="minimax")
        assert p.engine.client.auth_token == "sk-test"

    def test_native_anthropic_uses_api_key(self):
        """AnthropicProvider does NOT set _anthropic_use_bearer_auth,
        so the engine passes the key as api_key (x-api-key header)."""
        p = create_provider(api_key="sk-test", protocol="anthropic", provider_name="anthropic")
        assert p.engine.client.api_key == "sk-test"

    def test_deepseek_openai_unaffected(self):
        """Bearer auth flag only affects anthropic protocol, not openai."""
        p = create_provider(api_key="sk-test", protocol="openai", provider_name="deepseek")
        # OpenAI engine doesn't have auth_token/api_key distinction
        assert p.engine.base_url == "https://api.deepseek.com/v1"


# ── Resolution order: config URL > name > URL > fallback ──────────


class TestResolutionPriority:
    """Full integration: the complete priority chain is respected."""

    def test_priority_1_config_url_wins(self):
        """Config URL honored even when provider_name would give a different one."""
        p = create_provider(
            api_key="sk-test",
            base_url="https://my-gateway.example.com/v1",
            protocol="openai",
            provider_name="deepseek",
        )
        assert p.base_url == "https://my-gateway.example.com/v1"

    def test_priority_2_provider_name_without_url(self):
        """Without config URL, provider name determines class and default URL."""
        p = create_provider(
            api_key="sk-test",
            protocol="openai",
            provider_name="minimax",
        )
        assert isinstance(p, MiniMaxProvider)
        assert p.base_url == "https://api.minimaxi.com/v1"

    def test_priority_3_url_detection_without_name(self):
        """Without config URL or provider name, URL detection kicks in."""
        p = create_provider(
            api_key="sk-test",
            base_url="https://api.deepseek.com/v1",
        )
        assert isinstance(p, DeepSeekProvider)
        assert p.base_url == "https://api.deepseek.com/v1"

    def test_priority_4_fallback_openai(self):
        """With nothing at all, OpenAIProvider + its default URL."""
        p = create_provider(api_key="sk-test")
        assert isinstance(p, OpenAIProvider)
        assert p.base_url == "https://api.openai.com/v1"

    def test_full_config_simulation_minimax_no_url(self):
        """Simulates a real config: provider=minimax, no api_url, openai protocol."""
        p = create_provider(
            api_key="sk-test",
            protocol="openai",
            provider_name="minimax",
        )
        assert isinstance(p, MiniMaxProvider)
        assert p.base_url == "https://api.minimaxi.com/v1"
        # Engine should also have the correct URL
        assert p.engine.base_url == "https://api.minimaxi.com/v1"

    def test_full_config_simulation_deepseek_anthropic_no_url(self):
        """Simulates: provider=deepseek, type=anthropic, no api_url."""
        p = create_provider(
            api_key="sk-test",
            protocol="anthropic",
            provider_name="deepseek",
        )
        assert isinstance(p, DeepSeekProvider)
        assert p.base_url == "https://api.deepseek.com/anthropic"
        assert p.engine.base_url == "https://api.deepseek.com/anthropic"
