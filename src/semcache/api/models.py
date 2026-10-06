"""
GET /v1/models: the models a client can choose from.

The same shape as OpenAI's endpoint of that name, so OpenAI clients (and
the playground's model dropdown) can list what this proxy offers: the
Gemini models, plus the second provider's when one is configured.

The list is a convenience. A model missing from it can still be requested
by name in /v1/chat/completions.
"""

from typing import Annotated

from fastapi import APIRouter, Depends

from semcache.api.dependencies import get_provider
from semcache.config import PRICING_TABLE
from semcache.providers.base import LLMProvider
from semcache.providers.router import is_google_model

router = APIRouter()


@router.get("/models")
async def list_models(provider: Annotated[LLMProvider, Depends(get_provider)]) -> dict:
    return {
        "object": "list",
        "data": [
            {
                "id": model,
                "object": "model",
                "owned_by": "google" if is_google_model(model) else "openai-compatible",
                # Not part of OpenAI's shape: whether cost figures include this model.
                "priced": model in PRICING_TABLE["models"],
            }
            for model in await provider.list_models()
        ],
    }
