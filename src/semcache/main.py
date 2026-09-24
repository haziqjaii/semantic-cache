"""
FastAPI application entry point.
"""

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from semcache.api import chat
from semcache.api.dependencies import lifespan

app = FastAPI(
    title="Semantic Cache",
    description="Semantic caching layer for LLM APIs",
    version="0.1.0",
    lifespan=lifespan,
)

# Register the v1 router
app.include_router(chat.router, prefix="/v1")


@app.get("/health")
async def health_check():
    """Basic health check."""
    return JSONResponse(content={"status": "ok", "version": "0.1.0"})


def run() -> None:
    """Entry point for the `semcache` CLI command (see pyproject.toml)."""
    import uvicorn

    from semcache.config import get_settings

    # Safe to call here since run() is explicit execution, not import-time
    settings = get_settings()
    
    uvicorn.run(
        "semcache.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,  # auto-restart on code changes during development
    )


if __name__ == "__main__":
    run()
