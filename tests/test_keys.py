"""
Tests for cache key (namespace) generation.

These tests verify:
  1. Same inputs → same namespace (deterministic)
  2. Different system prompts → different namespaces (isolation)
  3. Different temperatures → different namespaces
  4. Different models → different namespaces
  5. Namespace is always a 16-char hex string (format)
"""

import pytest

from semcache.cache.keys import (
    build_namespace,
    hash_system_prompt,
    normalize_cache_tag,
    parse_cache_tags,
)


class TestBuildNamespace:
    """Test the build_namespace function."""

    def test_same_inputs_same_namespace(self):
        """Identical inputs must produce identical namespaces."""
        ns1 = build_namespace(model="gemini-3.5-flash", system_prompt="You are helpful.")
        ns2 = build_namespace(model="gemini-3.5-flash", system_prompt="You are helpful.")
        assert ns1 == ns2

    def test_different_system_prompts_different_namespaces(self):
        """
        Different system prompts must produce different namespaces.
        This is THE critical test — it prevents cross-contamination between
        features that use different system prompts.
        """
        ns1 = build_namespace(model="gemini-3.5-flash", system_prompt="You are a support agent.")
        ns2 = build_namespace(model="gemini-3.5-flash", system_prompt="You are a code reviewer.")
        assert ns1 != ns2

    def test_different_temperatures_different_namespaces(self):
        """Temperature affects output style, so it must affect the namespace."""
        ns1 = build_namespace(model="gemini-3.5-flash", temperature=0.0)
        ns2 = build_namespace(model="gemini-3.5-flash", temperature=1.0)
        assert ns1 != ns2

    def test_different_models_different_namespaces(self):
        """Different models give different answers, so different namespaces."""
        ns1 = build_namespace(model="gemini-3.5-flash")
        ns2 = build_namespace(model="gpt-4o")
        assert ns1 != ns2

    def test_namespace_is_16_char_hex(self):
        """Namespace should be a 16-character hexadecimal string."""
        ns = build_namespace(model="test-model", system_prompt="test")
        assert len(ns) == 16
        # Verify it's valid hex by trying to convert it.
        int(ns, 16)  # raises ValueError if not valid hex

    def test_none_system_prompt_treated_as_empty(self):
        """None and empty string system prompts should produce the same namespace."""
        ns1 = build_namespace(model="gemini-3.5-flash", system_prompt=None)
        ns2 = build_namespace(model="gemini-3.5-flash", system_prompt="")
        # Both should map to the same namespace since None is treated as "".
        assert ns1 == ns2

    def test_max_tokens_affects_namespace(self):
        """Different max_tokens means different expected outputs."""
        ns1 = build_namespace(model="gemini-3.5-flash", max_tokens=100)
        ns2 = build_namespace(model="gemini-3.5-flash", max_tokens=1000)
        assert ns1 != ns2


class TestHashSystemPrompt:
    def test_same_prompt_same_hash(self):
        assert hash_system_prompt("You are helpful.") == hash_system_prompt("You are helpful.")

    def test_none_and_empty_hash_the_same(self):
        assert hash_system_prompt(None) == hash_system_prompt("")

    def test_is_independent_of_model_and_params(self):
        """Unlike the namespace, it only depends on the system prompt."""
        ns_a = build_namespace(model="a", system_prompt="S", temperature=0.0)
        ns_b = build_namespace(model="b", system_prompt="S", temperature=1.0)
        assert ns_a != ns_b
        assert len(hash_system_prompt("S")) == 16


class TestCacheTags:
    def test_parse_normalizes_dedupes_and_sorts(self):
        assert parse_cache_tags(" Support , billing:V2,support,, ") == ["billing:v2", "support"]

    def test_parse_empty(self):
        assert parse_cache_tags(None) == []
        assert parse_cache_tags("") == []

    @pytest.mark.parametrize("bad", ["has space", "semi;colon", "-leading-dash", "x" * 65, "*"])
    def test_invalid_tags_are_rejected(self, bad):
        with pytest.raises(ValueError, match="Invalid cache tag"):
            normalize_cache_tag(bad)

    def test_too_many_tags(self):
        with pytest.raises(ValueError, match="At most 10"):
            parse_cache_tags(",".join(f"t{i}" for i in range(11)))
