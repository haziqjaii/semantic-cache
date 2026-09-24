"""
Tests for cache key (namespace) generation.

These tests verify:
  1. Same inputs → same namespace (deterministic)
  2. Different system prompts → different namespaces (isolation)
  3. Different temperatures → different namespaces
  4. Different models → different namespaces
  5. Namespace is always a 16-char hex string (format)
"""

from semcache.cache.keys import build_namespace


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
