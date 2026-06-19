import yaml

from cliver.config import ConfigManager, ModelConfig, ProviderConfig, RateLimitConfig


class TestRateLimitConfig:
    def test_basic(self):
        rl = RateLimitConfig(requests=5000, period="5h")
        assert rl.requests == 5000
        assert rl.period == "5h"
        assert rl.margin == 0.1  # default

    def test_custom_margin(self):
        rl = RateLimitConfig(requests=1000, period="1h", margin=0.2)
        assert rl.margin == 0.2


class TestProviderConfig:
    def test_basic(self):
        p = ProviderConfig(name="minimax", type="openai", api_url="https://api.minimaxi.com/v1")
        assert p.name == "minimax"
        assert p.type == "openai"
        assert p.api_url == "https://api.minimaxi.com/v1"
        assert p.api_key is None
        assert p.rate_limit is None

    def test_with_api_key(self):
        p = ProviderConfig(name="mm", type="openai", api_url="http://x", api_key="sk-123")
        assert p.get_api_key() == "sk-123"

    def test_with_rate_limit(self):
        p = ProviderConfig(
            name="mm",
            type="openai",
            api_url="http://x",
            rate_limit=RateLimitConfig(requests=5000, period="5h"),
        )
        assert p.rate_limit.requests == 5000

    def test_model_dump_excludes_name_and_nulls(self):
        p = ProviderConfig(name="mm", type="openai", api_url="http://x")
        dumped = p.model_dump()
        assert "name" not in dumped
        assert "api_key" not in dumped
        assert dumped["type"] == "openai"
        assert dumped["api_url"] == "http://x"


class TestModelConfigProviderResolution:
    def _make_provider(self, **kwargs):
        defaults = {"name": "mm", "type": "openai", "api_url": "https://api.mm.com/v1", "api_key": "sk-prov"}
        defaults.update(kwargs)
        return ProviderConfig(**defaults)

    def test_url_from_provider(self):
        prov = self._make_provider()
        mc = ModelConfig(name="m1", provider="mm", model="m1")
        assert mc.get_resolved_url(pc=prov) == "https://api.mm.com/v1"

    def test_url_always_from_provider(self):
        """URL always comes from provider config; model has no url field."""
        prov = self._make_provider()
        mc = ModelConfig(name="m1", provider="mm", model="m1")
        assert mc.get_resolved_url(pc=prov) == "https://api.mm.com/v1"

    def test_api_key_from_provider(self):
        prov = self._make_provider()
        mc = ModelConfig(name="m1", provider="mm", model="m1")
        assert mc.get_api_key(pc=prov) == "sk-prov"

    def test_api_key_always_from_provider(self):
        """API key always comes from provider config; model has no api_key field."""
        prov = self._make_provider(api_key="sk-new")
        mc = ModelConfig(name="m1", provider="mm", model="m1")
        assert mc.get_api_key(pc=prov) == "sk-new"

    def test_provider_protocol(self):
        prov = self._make_provider(type="anthropic")
        mc = ModelConfig(name="m1", provider="mm", model="m1")
        assert mc.get_provider_protocol(pc=prov) == "anthropic"

    def test_provider_protocol_default(self):
        """Without a ProviderConfig, defaults to 'openai'."""
        mc = ModelConfig(name="m1", provider="openai", model="m1")
        assert mc.get_provider_protocol() == "openai"

    def test_resolved_url_no_provider_no_url(self):
        mc = ModelConfig(name="m1", provider="openai", model="m1")
        assert mc.get_resolved_url() is None

    def test_model_dump_no_internal_fields(self):
        mc = ModelConfig(name="m1", provider="mm", model="m1")
        dumped = mc.model_dump()
        assert "_provider_config" not in dumped


class TestConfigLoadingWithProviders:
    def test_load_providers_section(self, tmp_path):
        config_yaml = {
            "providers": {
                "minimax": {
                    "type": "openai",
                    "api_url": "https://api.minimaxi.com/v1",
                    "api_key": "sk-test",
                    "rate_limit": {"requests": 5000, "period": "5h"},
                }
            },
            "models": {
                "text": {
                    "MiniMax-M2.7": {
                        "provider": "minimax",
                        "model": "MiniMax-M2.7",
                    }
                }
            },
            "default_model": "MiniMax-M2.7",
        }
        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_yaml, f)

        cm = ConfigManager(tmp_path)
        assert "minimax" in cm.config.providers
        prov = cm.config.providers["minimax"]
        assert prov.type == "openai"
        assert prov.rate_limit.requests == 5000

        m1 = cm.all_models()["MiniMax-M2.7"]
        pc = cm.get_provider_config(m1.provider)
        assert m1.get_provider_protocol(pc) == "openai"
        assert m1.get_resolved_url(pc) == "https://api.minimaxi.com/v1"
        assert m1.get_api_key(pc) == "sk-test"


class TestModelsLinkedToProvider:
    def test_model_linked_to_provider(self, tmp_path):
        """Models nested under a provider get linked to that provider config."""
        config_yaml = {
            "providers": {
                "local": {
                    "type": "openai",
                    "api_url": "http://localhost:8080/v1",
                    "api_key": "sk-old",
                }
            },
            "models": {
                "text": {
                    "qwen3": {
                        "provider": "local",
                        "model": "qwen3",
                    }
                }
            },
            "default_model": "qwen3",
        }
        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_yaml, f)

        cm = ConfigManager(tmp_path)
        m = cm.all_models()["qwen3"]
        pc = cm.get_provider_config(m.provider)
        assert m.get_provider_protocol(pc) == "openai"
        assert m.get_resolved_url(pc) == "http://localhost:8080/v1"
        assert m.get_api_key(pc) == "sk-old"

    def test_models_share_provider(self, tmp_path):
        """Multiple models under the same provider share the same ProviderConfig."""
        config_yaml = {
            "providers": {
                "openai": {
                    "type": "openai",
                    "api_url": "http://api.com/v1",
                    "api_key": "sk-1",
                }
            },
            "models": {
                "text": {
                    "m1": {"provider": "openai", "model": "m1"},
                    "m2": {"provider": "openai", "model": "m2"},
                }
            },
            "default_model": "m1",
        }
        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_yaml, f)

        cm = ConfigManager(tmp_path)
        m1 = cm.all_models()["m1"]
        m2 = cm.all_models()["m2"]
        # Both models reference the same provider — resolve to the same ProviderConfig.
        assert cm.get_provider_config(m1.provider) is cm.get_provider_config(m2.provider)
