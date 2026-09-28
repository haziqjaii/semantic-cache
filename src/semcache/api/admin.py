from fastapi import APIRouter, Depends
from semcache.api.dependencies import get_engine
from semcache.cache.engine import CacheEngine

router = APIRouter()

@router.get("/stats")
async def get_stats(engine: CacheEngine = Depends(get_engine)):
    """
    Get live cache state.
    """
    stats = await engine.stats()
    return stats
