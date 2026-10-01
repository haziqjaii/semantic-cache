"""
Threshold tuning endpoints: feedback, near-miss analyzer, tuner, thresholds.

HOW THEY FIT TOGETHER
    1. Every cacheable request is logged with its closest cached question
       and their similarity (see cache/lookup_log.py). Responses carry the
       log id in the X-Cache-Lookup-Id header.
    2. People label lookups with POST /feedback:
         - a HIT:       was the cached answer right for this question?
         - a near miss: would the cached answer have been right?
    3. GET /near-misses lists lookups that almost matched, to label and
       to spot phrasings the cache is missing.
    4. GET /tuner shows hit rate vs. wrong-answer rate at each threshold.
    5. Labels teach per-intent thresholds that lookups start using
       immediately (GET /thresholds shows them; see cache/tuning.py).

Feedback and the near-miss list are admin-only: feedback changes how the
cache behaves, and near misses show other users' questions.
"""

from collections import Counter
from dataclasses import asdict
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from semcache.api.admin import require_admin
from semcache.api.dependencies import get_engine
from semcache.cache.engine import CacheEngine
from semcache.cache.lookup_log import NEAR_MISS, LookupEvent
from semcache.cache.policy import TASK_POLICIES
from semcache.cache.tuning import MIN_LABELS, MIN_SUPPORT, TARGET_PRECISION, tune

router = APIRouter()

# Intents that can be cached, so can have hits, near misses and thresholds.
_CACHEABLE_INTENTS = [name for name, policy in TASK_POLICIES.items() if policy.ttl_seconds > 0]


def _event_json(event: LookupEvent) -> dict:
    return {
        "lookup_id": event.id,
        "created_at": event.created_at.isoformat(),
        "prompt": event.prompt,
        "model": event.model,
        "outcome": event.outcome,
        "similarity": event.similarity,
        "required_similarity": event.required_similarity,
        "intent": event.intent,
        "candidate_prompt": event.candidate_prompt,
        "good_match": event.good_match,
    }


class FeedbackRequest(BaseModel):
    lookup_id: str = Field(description="From the X-Cache-Lookup-Id response header.")
    good_match: bool = Field(
        description="Does the cached question's answer fit this question? "
        "For a hit: was the served answer right? For a near miss: would it have been?"
    )


@router.post("/feedback", dependencies=[Depends(require_admin)])
async def feedback(
    request: FeedbackRequest,
    engine: Annotated[CacheEngine, Depends(get_engine)],
) -> dict:
    """Label a lookup, and re-learn thresholds from all labels so far."""
    try:
        event = await engine.label_lookup(request.lookup_id, request.good_match)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"lookup": _event_json(event), "learned_thresholds": engine.learned_thresholds}


@router.get("/lookups/{lookup_id}", dependencies=[Depends(require_admin)])
async def lookup(lookup_id: str, engine: Annotated[CacheEngine, Depends(get_engine)]) -> dict:
    """
    One lookup, by the id in its X-Cache-Lookup-Id header: on a hit,
    candidate_prompt is the cached question whose answer was served.
    """
    event = await engine.get_lookup(lookup_id)
    if event is None:
        raise HTTPException(status_code=404, detail=f"No lookup with id {lookup_id!r} (unknown or expired).")
    return {"lookup": _event_json(event)}


@router.get("/near-misses", dependencies=[Depends(require_admin)])
async def near_misses(
    engine: Annotated[CacheEngine, Depends(get_engine)],
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
) -> dict:
    """
    Recent lookups that found a similar cached question but fell below its
    threshold, newest first.

    Label them to teach thresholds: if many near misses at 0.93 are really
    the same question, the threshold for that intent is too strict.
    """
    events = [e for e in await engine.recent_lookups() if e.outcome == NEAR_MISS]
    return {
        "total": len(events),
        "labelled": sum(1 for e in events if e.good_match is not None),
        "near_misses": [_event_json(e) for e in events[:limit]],
    }


@router.get("/tuner")
async def tuner(
    engine: Annotated[CacheEngine, Depends(get_engine)],
    intent: Annotated[str | None, Query(description="Only lookups whose closest match had this intent.")] = None,
) -> dict:
    """
    What-if analysis over recent lookups: at each threshold, how many would
    have been hits, and how many of those were labelled wrong.
    """
    if intent is not None and intent not in _CACHEABLE_INTENTS:
        raise HTTPException(status_code=422, detail=f"intent must be one of {_CACHEABLE_INTENTS}")
    events = await engine.recent_lookups()
    if intent is not None:
        events = [e for e in events if e.intent == intent]
    return {
        "intent": intent,
        "lookups_analysed": len(events),
        "labelled": sum(1 for e in events if e.good_match is not None),
        "rows": [asdict(row) for row in tune(events)],
    }


@router.get("/thresholds")
async def thresholds(engine: Annotated[CacheEngine, Depends(get_engine)]) -> dict:
    """The threshold each intent uses now: its default, or one learned from feedback."""
    learned = await engine.refresh_learned_thresholds()
    # Count labels for every intent, including ones still short of MIN_LABELS.
    labels = Counter(e.intent for e in await engine.labelled_lookups() if e.has_candidate)
    return {
        "learning": {
            "min_labels": MIN_LABELS,
            "min_support": MIN_SUPPORT,
            "target_precision": TARGET_PRECISION,
        },
        "intents": [
            {
                "intent": name,
                "default": TASK_POLICIES[name].similarity_threshold,
                "learned": learned[name].threshold if name in learned else None,
                "in_use": learned[name].threshold if name in learned else TASK_POLICIES[name].similarity_threshold,
                "labels": labels[name],
                "precision": learned[name].precision if name in learned else None,
            }
            for name in _CACHEABLE_INTENTS
        ],
    }
