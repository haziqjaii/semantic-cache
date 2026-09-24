"""
FastAPI application entry point.

This is a minimal placeholder for Phase 1. The full API routes
(chat completions proxy, admin endpoints) come in Phase 2.

For now, it just starts the server with a health check endpoint
so we can verify the project structure works.
"""

import uvicorn
from fastapi import FastAPI

app = FastAPI(
    title="Semantic Cache",
    description="Semantic caching layer for LLM APIs",
    version="0.1.0",
)


@app.get("/health")
async def health_check():
    """Basic health check — confirms the server is running."""
    return {"status": "ok", "version": "0.1.0"}


def run() -> None:
    """Entry point for the `semcache` CLI command (see pyproject.toml)."""
    uvicorn.run(
        "semcache.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # auto-restart on code changes during development
    )


if __name__ == "__main__":
    run()
