"""
The playground: a self-contained web page for trying the cache by hand.

It calls the public API from the browser (chat completions, analytics,
invalidation), so everything it shows is what any client would see.
"""

from importlib.resources import files

from fastapi import APIRouter
from fastapi.responses import HTMLResponse, RedirectResponse

router = APIRouter(include_in_schema=False)

_PAGE = files("semcache") / "static" / "playground.html"


@router.get("/")
def root() -> RedirectResponse:
    return RedirectResponse("/playground")


@router.get("/playground")
def playground() -> HTMLResponse:
    # Read per request so edits to the page show up without a restart.
    return HTMLResponse(_PAGE.read_text(encoding="utf-8"))
