#!/usr/bin/env python3
"""
HTM-Monitor Demo Server
========================
A lightweight FastAPI server that serves the demo frontend and data.

Quick start:
    cd htm-monitor/demo
    pip install -r requirements_demo.txt
    python server.py

Then open http://localhost:8000 in your browser.

Default credentials:  demo / htm-monitor-2024
Change them below or via environment variables:
    HTM_DEMO_USER=youruser HTM_DEMO_PASS=yourpass python server.py
"""

import json
import os
import secrets
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from fastapi import Cookie, Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEMO_USER     = os.environ.get("HTM_DEMO_USER", "demo")
DEMO_PASS     = os.environ.get("HTM_DEMO_PASS", "htm-monitor-2024")
SESSION_TTL_S = int(os.environ.get("HTM_SESSION_TTL", 86400))   # 24 hours
PORT          = int(os.environ.get("HTM_PORT", 8000))
HOST          = os.environ.get("HTM_HOST", "0.0.0.0")

STATIC_DIR  = Path(__file__).parent / "static"
DATA_DIR    = STATIC_DIR / "demo_data"

# In-memory session store  { token: expiry_unix_ts }
_sessions: dict[str, float] = {}

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="HTM-Monitor Demo", docs_url=None, redoc_url=None)


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------

def _new_token() -> str:
    return secrets.token_urlsafe(32)


def _is_valid(token: Optional[str]) -> bool:
    if not token:
        return False
    exp = _sessions.get(token)
    if exp is None:
        return False
    if time.time() > exp:
        del _sessions[token]
        return False
    return True


def require_auth(session: Optional[str] = Cookie(default=None)) -> str:
    if not _is_valid(session):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
        )
    return session


# ---------------------------------------------------------------------------
# Auth endpoints
# ---------------------------------------------------------------------------

class LoginRequest(BaseModel):
    username: str
    password: str


@app.post("/auth/login")
def login(body: LoginRequest, response: Response):
    if body.username != DEMO_USER or body.password != DEMO_PASS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
        )
    token = _new_token()
    _sessions[token] = time.time() + SESSION_TTL_S
    response.set_cookie(
        key="session",
        value=token,
        httponly=True,
        samesite="lax",
        max_age=SESSION_TTL_S,
    )
    return {"ok": True}


@app.post("/auth/logout")
def logout(response: Response, session: Optional[str] = Cookie(default=None)):
    if session and session in _sessions:
        del _sessions[session]
    response.delete_cookie("session")
    return {"ok": True}


@app.get("/auth/me")
def me(session: str = Depends(require_auth)):
    return {"username": DEMO_USER, "authenticated": True}


# ---------------------------------------------------------------------------
# Use-case catalog
# ---------------------------------------------------------------------------

@app.get("/api/usecases")
def list_usecases(session: str = Depends(require_auth)):
    """Return available use cases (metadata only, no time-series data)."""
    usecases = []
    for json_file in DATA_DIR.glob("*.json"):
        try:
            with open(json_file) as f:
                doc = json.load(f)
            meta = doc.get("meta", {})
            usecases.append({
                "id":          meta.get("id", json_file.stem),
                "title":       meta.get("title", json_file.stem),
                "subtitle":    meta.get("subtitle", ""),
                "description": meta.get("description", ""),
                "events":      meta.get("events", []),
                "time_range":  meta.get("time_range", {}),
                "total_steps": meta.get("total_steps", 0),
            })
        except Exception:
            pass
    return usecases


@app.get("/api/usecases/{usecase_id}/meta")
def usecase_meta(usecase_id: str, session: str = Depends(require_auth)):
    """Return full metadata for a use case (no time-series data)."""
    data_file = DATA_DIR / f"{usecase_id}.json"
    if not data_file.exists():
        raise HTTPException(status_code=404, detail=f"Use case not found: {usecase_id}")
    with open(data_file) as f:
        doc = json.load(f)
    return doc.get("meta", {})


@app.get("/api/usecases/{usecase_id}/data")
def usecase_data(
    usecase_id: str,
    start: int = 0,
    count: int = 0,
    session: str = Depends(require_auth),
):
    """
    Return time-series data for a use case.
    start=0, count=0  →  return ALL data (default for demo, data is ~5–10 MB)
    start=N, count=M  →  return slice [N, N+M)
    """
    data_file = DATA_DIR / f"{usecase_id}.json"
    if not data_file.exists():
        raise HTTPException(status_code=404, detail=f"Use case not found: {usecase_id}")
    with open(data_file) as f:
        doc = json.load(f)

    data = doc.get("data", {})

    if count > 0:
        end = start + count
        sliced = {
            "timestamps":  data["timestamps"][start:end],
            "in_warmup":   data["in_warmup"][start:end],
            "signals":     {k: v[start:end] for k, v in data["signals"].items()},
            "models":      {m: {k: v[start:end] for k, v in mv.items()} for m, mv in data["models"].items()},
            "groups":      {g: {k: v[start:end] for k, v in gv.items()} for g, gv in data["groups"].items()},
            "system":      {k: v[start:end] for k, v in data["system"].items()},
        }
        return JSONResponse({"meta": doc["meta"], "data": sliced})

    return JSONResponse(doc)


# ---------------------------------------------------------------------------
# Static files + SPA fallback
# ---------------------------------------------------------------------------

# Serve everything in static/ (including the generated demo_data/*.json)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
def index():
    index_path = STATIC_DIR / "index.html"
    return HTMLResponse(index_path.read_text())


# Catch-all: serve index.html for any unknown path (SPA routing)
@app.exception_handler(404)
async def spa_fallback(request: Request, exc: HTTPException):
    # Don't fall back for API or static routes
    path = str(request.url.path)
    if path.startswith("/api/") or path.startswith("/auth/") or path.startswith("/static/"):
        return JSONResponse({"detail": exc.detail}, status_code=404)
    index_path = STATIC_DIR / "index.html"
    return HTMLResponse(index_path.read_text())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    print(f"\n  HTM-Monitor Demo Server")
    print(f"  ─────────────────────────────")
    print(f"  URL:      http://localhost:{PORT}")
    print(f"  Login:    {DEMO_USER} / {DEMO_PASS}")
    print(f"  Data dir: {DATA_DIR}")
    print()
    uvicorn.run("server:app", host=HOST, port=PORT, reload=False, log_level="warning")
