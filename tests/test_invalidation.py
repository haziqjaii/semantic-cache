"""
Tests for cache invalidation: EntryFilter semantics and CacheEngine.invalidate.

These use the in-memory store, which filters with EntryFilter.matches — the
reference semantics the Redis store's queries must match (see the Redis
integration tests).
"""

import pytest

from semcache.cache.engine import CacheEngine
from semcache.cache.keys import hash_system_prompt
from semcache.cache.store.base import CacheEntry, EntryFilter


class TestEntryFilter:
    @pytest.mark.parametrize("field", ["namespace", "model", "system_prompt_hash", "tag", "tag_prefix"])
    def test_empty_values_are_rejected(self, field):
        """Redis would treat an empty value as a wildcard and match everything."""
        with pytest.raises(ValueError, match="must not be empty"):
            EntryFilter(**{field: "  "})

    def test_empty_filter_matches_everything(self):
        entry = CacheEntry(prompt="p", response="r", model="m", namespace="ns")
        assert EntryFilter().is_empty()
        assert EntryFilter().matches(entry)

    def test_fields_are_anded_and_case_insensitive(self):
        entry = CacheEntry(
            prompt="p", response="r", model="Gemini-3.5-Flash", namespace="ns",
            system_prompt_hash="abc", tags=["support", "billing:v2"],
        )
        assert EntryFilter(model="gemini-3.5-flash", tag="SUPPORT").matches(entry)
        assert EntryFilter(tag_prefix="bill").matches(entry)
        assert not EntryFilter(model="gemini-3.5-flash", tag="sales").matches(entry)
        assert not EntryFilter(tag_prefix="sup:").matches(entry)


@pytest.fixture
def engine(mock_embedder, memory_store) -> CacheEngine:
    return CacheEngine(embedder=mock_embedder, store=memory_store)


async def _cache(engine, prompt, *, model="m1", system_prompt=None, temperature=None, tags=None):
    result = await engine.lookup(
        prompt=prompt, model=model, system_prompt=system_prompt, temperature=temperature
    )
    await engine.store(
        lookup_result=result, prompt=prompt, response=f"answer to {prompt}", model=model, tags=tags
    )


class TestInvalidate:
    @pytest.mark.asyncio
    async def test_system_prompt_spans_models_and_params(self, engine):
        """One system prompt's entries go, across models and temperatures."""
        await _cache(engine, "q1", model="m1", system_prompt="Support bot", temperature=0.0)
        await _cache(engine, "q2", model="m2", system_prompt="Support bot", temperature=1.0)
        await _cache(engine, "q3", model="m1", system_prompt="Code reviewer")

        deleted = await engine.invalidate(
            EntryFilter(system_prompt_hash=hash_system_prompt("Support bot"))
        )

        assert deleted == 2
        assert await engine.count() == 1

    @pytest.mark.asyncio
    async def test_by_model(self, engine):
        await _cache(engine, "q1", model="gemini-3.5-flash")
        await _cache(engine, "q2", model="gemini-3.5-flash-lite")

        assert await engine.invalidate(EntryFilter(model="gemini-3.5-flash")) == 1
        assert await engine.count(EntryFilter(model="gemini-3.5-flash-lite")) == 1

    @pytest.mark.asyncio
    async def test_by_tag_and_tag_prefix(self, engine):
        await _cache(engine, "q1", tags=["billing:v1"])
        await _cache(engine, "q2", tags=["billing:v2", "support"])
        await _cache(engine, "q3", tags=["support"])
        await _cache(engine, "q4")

        assert await engine.count(EntryFilter(tag="support")) == 2
        assert await engine.invalidate(EntryFilter(tag_prefix="billing:")) == 2
        assert await engine.count() == 2

    @pytest.mark.asyncio
    async def test_empty_filter_is_refused_unless_explicit(self, engine):
        await _cache(engine, "q1")
        await _cache(engine, "q2")

        with pytest.raises(ValueError, match="empty filter"):
            await engine.invalidate(EntryFilter())
        assert await engine.count() == 2

        assert await engine.invalidate(EntryFilter(), allow_all=True) == 2
        assert await engine.count() == 0

    @pytest.mark.asyncio
    async def test_invalidated_entry_is_no_longer_served(self, engine):
        await _cache(engine, "What is Python?", model="m1", tags=["docs"])
        assert (await engine.lookup(prompt="What is Python?", model="m1")).hit

        await engine.invalidate(EntryFilter(tag="docs"))

        assert not (await engine.lookup(prompt="What is Python?", model="m1")).hit

    @pytest.mark.asyncio
    async def test_store_records_system_prompt_hash_and_tags(self, engine, memory_store):
        await _cache(engine, "q1", system_prompt="Support bot", tags=["support"])

        (_, entry), = memory_store._entries
        assert entry.system_prompt_hash == hash_system_prompt("Support bot")
        assert entry.tags == ["support"]
