import asyncio
import json
import logging
import os
import re
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, List, Optional

try:
    from contextlib import aclosing  # Python 3.10+
except ImportError: 
    class aclosing:  # noqa: N801
        def __init__(self, thing):
            self.thing = thing

        async def __aenter__(self):
            return self.thing

        async def __aexit__(self, *exc):
            await self.thing.aclose()

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from graph import (
    GENERATED_DIR,
    MAX_BUILD_RETRIES,
    SAMPLE_FILES,
    State,
    _safe_session_id,
    _session_path,
    app_graph_a,
    app_graph_b,
    build_zip,
    cleanup_old_sessions,
    latest_version,
    save_version,
)

logger = logging.getLogger("appmentor")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(os.path.dirname(BASE_DIR), "frontend")
os.makedirs(GENERATED_DIR, exist_ok=True)
print(f"AppMentor: generated apps are saved in {GENERATED_DIR}")

ALLOWED_FILES = {"index.html", "style.css", "app.js"}

# How long an idle session's files are kept. Raise this if people come back later.
SESSION_MAX_AGE_HOURS = int(os.getenv("SESSION_MAX_AGE_HOURS", "24"))

# Comma-separated list of allowed origins, e.g. "https://myapp.com,http://localhost:3000".
# Comma-separated list of allowed origins
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "*,https://appmentor-frontend.vercel.app").split(",") if o.strip()]
# Generated code is untrusted. By default previews are served in a CSP sandbox so they
# cannot touch this app's origin. A sandboxed page has an opaque origin, so the browser's
# real localStorage throws; we inject a localStorage shim (see _ERROR_SCRIPT) that keeps
# the data in the parent page instead. Set PREVIEW_SANDBOX=0 to disable the CSP sandbox.
PREVIEW_SANDBOX = os.getenv("PREVIEW_SANDBOX", "1") != "0"

LEVELS = {"beginner", "intermediate", "advanced"}


# --------------------------------------------------------------------------
# App setup
# --------------------------------------------------------------------------

async def _periodic_cleanup() -> None:
    while True:
        try:
            await asyncio.to_thread(cleanup_old_sessions, max_age_hours=SESSION_MAX_AGE_HOURS)
        except Exception:
            logger.exception("session cleanup failed")
        await asyncio.sleep(3600)


@asynccontextmanager
async def lifespan(_: FastAPI):
    task = asyncio.create_task(_periodic_cleanup())
    try:
        yield
    finally:
        task.cancel()


app = FastAPI(lifespan=lifespan)

# Wildcard origins and credentials cannot be combined (browsers reject it).
# No cookies are used here, so credentials are off.
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Safe check for static files (Railway compatible)
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
async def root():
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {
        "status": "online",
        "message": "Backend is running successfully! Frontend is hosted on Vercel."
    }


# --------------------------------------------------------------------------
# Request models + validation helpers
# --------------------------------------------------------------------------

class ApiRequest(BaseModel):
    session_id: Optional[str] = None
    user_idea: str = Field(default="", max_length=4000)
    experience_level: str = "beginner"
    questions: List[str] = Field(default_factory=list)
    answers: List[str] = Field(default_factory=list)
    plan: Dict[str, Any] = Field(default_factory=dict)
    understanding: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = Field(default=None, max_length=4000)


def _new_session_id() -> str:
    return uuid.uuid4().hex[:8]


def _session_id_or_new(sid: Optional[str]) -> str:
    if not sid:
        return _new_session_id()
    if not _safe_session_id(sid):
        raise HTTPException(status_code=400, detail="invalid session_id")
    return sid


def _require_session(sid: str) -> None:
    if not _safe_session_id(sid):
        raise HTTPException(status_code=400, detail="invalid session_id")


def _safe_version(v: str) -> bool:
    """'latest', or a plain name that starts with a letter/digit (blocks '.', '..', slashes)."""
    return v == "latest" or bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,31}", v))


def _resolve_within(base: str, *parts: str) -> Optional[str]:
    """Join paths and make sure the result stays inside base (no traversal/symlink escape)."""
    real_base = os.path.realpath(base)
    path = os.path.realpath(os.path.join(real_base, *parts))
    try:
        if os.path.commonpath([real_base, path]) != real_base:
            return None
    except ValueError:
        return None
    return path


def _clean_list(items: List[str], max_items: int = 20, max_len: int = 2000) -> List[str]:
    return [str(x)[:max_len] for x in (items or [])][:max_items]


def _level(value: str) -> str:
    v = (value or "").strip().lower()
    return v if v in LEVELS else "beginner"


def _base_state(req: ApiRequest, sid: str) -> State:
    return {
        "session_id": sid,
        "user_idea": req.user_idea.strip(),
        "experience_level": _level(req.experience_level),
        "questions": _clean_list(req.questions),
        "answers": _clean_list(req.answers),
        "retry_count": 0,
        "errors": [],
    }


def _read_version_files(sid: str, version: str) -> Dict[str, str]:
    vdir = _resolve_within(_session_path(sid), version)
    files: Dict[str, str] = {}
    if not vdir:
        return files
    for name in sorted(ALLOWED_FILES):
        p = os.path.join(vdir, name)
        if os.path.isfile(p):
            with open(p, "r", encoding="utf-8") as f:
                files[name] = f.read()
    return files


# --------------------------------------------------------------------------
# Preview + download
# --------------------------------------------------------------------------

_ERROR_SCRIPT = (
    "<script>(function(){"
    "function P(m){try{window.parent.postMessage(m,'*');}catch(e){}}"
    "window.addEventListener('error',function(ev){P({type:'preview-error',message:String(ev.error||ev.message)});});"
    "window.addEventListener('unhandledrejection',function(ev){P({type:'preview-error',message:String(ev.reason)});});"
    "var d={};"
    "try{if(location.hash.indexOf('#store=')===0)d=JSON.parse(decodeURIComponent(location.hash.slice(7)))||{};}catch(e){d={};}"
    "function sync(){P({type:'storage-sync',data:d});}"
    "var s={"
    "getItem:function(k){k=String(k);return Object.prototype.hasOwnProperty.call(d,k)?d[k]:null;},"
    "setItem:function(k,v){d[String(k)]=String(v);sync();},"
    "removeItem:function(k){delete d[String(k)];sync();},"
    "clear:function(){d={};sync();},"
    "key:function(i){return Object.keys(d)[i]||null;}};"
    "Object.defineProperty(s,'length',{get:function(){return Object.keys(d).length;}});"
    "try{Object.defineProperty(window,'localStorage',{get:function(){return s;},configurable:true});}catch(e){}"
    "})();</script>"
)


def _inject_error_script(html: str) -> str:
    if not html:
        return html
    m = re.search(r"<head[^>]*>", html, re.IGNORECASE)
    if m:
        return html[: m.end()] + _ERROR_SCRIPT + html[m.end():]
    return _ERROR_SCRIPT + html


def _preview_headers() -> Dict[str, str]:
    headers = {"X-Content-Type-Options": "nosniff", "Cache-Control": "no-store"}
    if PREVIEW_SANDBOX:
        headers["Content-Security-Policy"] = "sandbox allow-scripts allow-forms allow-modals allow-popups"
    return headers


@app.get("/preview/{session_id}/{version}/{filename}")
async def preview_file(session_id: str, version: str, filename: str):
    _require_session(session_id)
    if filename not in ALLOWED_FILES:
        raise HTTPException(status_code=403, detail="forbidden")
    if not _safe_version(version):
        raise HTTPException(status_code=400, detail="invalid version")

    v = latest_version(session_id) if version == "latest" else version
    if not v:
        raise HTTPException(status_code=404, detail="not found")

    fpath = _resolve_within(_session_path(session_id), v, filename)
    if not fpath or not os.path.isfile(fpath):
        raise HTTPException(status_code=404, detail="not found")

    headers = _preview_headers()
    if filename == "index.html":
        with open(fpath, "r", encoding="utf-8") as f:
            html = _inject_error_script(f.read())
        return Response(content=html, media_type="text/html; charset=utf-8", headers=headers)

    media = "text/css; charset=utf-8" if filename.endswith(".css") else "text/javascript; charset=utf-8"
    return FileResponse(fpath, media_type=media, headers=headers)


@app.get("/api/files/{session_id}/{version}")
async def get_files(session_id: str, version: str):
    _require_session(session_id)
    if not _safe_version(version):
        raise HTTPException(status_code=400, detail="invalid version")
    v = latest_version(session_id) if version == "latest" else version
    files = await asyncio.to_thread(_read_version_files, session_id, v) if v else {}
    if not files:
        raise HTTPException(status_code=404, detail="not found")
    return {"version": v, "files": files}


@app.get("/api/download/{session_id}")
async def download_zip(session_id: str, version: Optional[str] = Query(default=None)):
    _require_session(session_id)
    if version is not None and not _safe_version(version):
        raise HTTPException(status_code=400, detail="invalid version")
    if version == "latest":
        version = None
    data, v = await asyncio.to_thread(build_zip, session_id, version)
    if not data:
        raise HTTPException(status_code=404, detail="not found")
    safe_v = re.sub(r"[^A-Za-z0-9_.-]", "", str(v or "latest"))
    name = f"appmentor-{session_id}-{safe_v}.zip"
    return Response(
        content=data,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{name}"'},
    )


# --------------------------------------------------------------------------
# SSE helpers
# --------------------------------------------------------------------------

def sse(event: str, data: Dict, sid: Optional[str] = None) -> str:
    payload = dict(data)
    if sid:
        payload["session_id"] = sid
    return f"data: {json.dumps({'event': event, 'data': payload})}\n\n"


def _sse_response(gen: AsyncGenerator[str, None]) -> StreamingResponse:
    return StreamingResponse(
        gen,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


def _error_event(e: Exception, sid: str) -> str:
    logger.exception("request failed (session %s)", sid)
    return sse("error", {"message": str(e)[:300]}, sid)


async def stream_graph_a(state: State, sid: str, clarified: bool) -> AsyncGenerator[str, None]:
    try:
        async for chunk in app_graph_a.astream(state, stream_mode="updates"):
            up = (chunk or {}).get("understand_plan")
            if not up:
                continue
            questions = up.get("questions") or []
            yield sse(
                "understand_done",
                {"understanding": up.get("understanding"), "questions": questions, "plan": up.get("plan")},
                sid,
            )
            if questions and not clarified:
                yield sse("needs_clarification", {"questions": questions}, sid)
            else:
                yield sse("plan_done", {"plan": up.get("plan")}, sid)
        yield sse("done", {}, sid)
    except Exception as e:
        yield _error_event(e, sid)


async def stream_build(state: State, sid: str, *, explain: bool) -> AsyncGenerator[str, None]:
    merged: Dict[str, Any] = dict(state)
    try:
        async with aclosing(app_graph_b.astream(state, stream_mode="updates")) as stream:
            async for chunk in stream:
                for node, update in (chunk or {}).items():
                    if not update:
                        continue
                    merged.update(update)

                    if node == "validate":
                        errs = update.get("errors") or []
                        if errs:
                            if (merged.get("retry_count") or 0) > MAX_BUILD_RETRIES:
                                msg = "Generated code failed validation: " + "; ".join(errs)
                                yield sse("error", {"message": msg[:300]}, sid)
                                return
                            continue
                        files = merged.get("code_files") or {}
                        version = await asyncio.to_thread(save_version, sid, files)
                        yield sse("build_done", {"version": version, "files": files}, sid)
                        if not explain:
                            yield sse("done", {}, sid)
                            return

                    elif node == "explain_learn":
                        explanation = update.get("explanation") or {}
                        yield sse(
                            "explain_done",
                            {"explanation": explanation, "explanation_error": update.get("explanation_error")},
                            sid,
                        )
                        yield sse("learn_done", {"learning": update.get("learning") or explanation}, sid)
        yield sse("done", {}, sid)
    except Exception as e:
        yield _error_event(e, sid)


# --------------------------------------------------------------------------
# API routes
# --------------------------------------------------------------------------

@app.post("/api/start")
async def api_start(req: ApiRequest):
    if not req.user_idea.strip():
        raise HTTPException(status_code=400, detail="user_idea is required")
    sid = _new_session_id()
    return _sse_response(stream_graph_a(_base_state(req, sid), sid, clarified=False))


@app.post("/api/clarify")
async def api_clarify(req: ApiRequest):
    if not req.user_idea.strip():
        raise HTTPException(status_code=400, detail="user_idea is required")
    answers = _clean_list(req.answers)
    if not answers:
        raise HTTPException(status_code=400, detail="answers are required")
    sid = _session_id_or_new(req.session_id)
    return _sse_response(stream_graph_a(_base_state(req, sid), sid, clarified=True))


@app.post("/api/approve")
async def api_approve(req: ApiRequest):
    if not req.plan:
        raise HTTPException(status_code=400, detail="plan is required")
    sid = _session_id_or_new(req.session_id)
    state = _base_state(req, sid)
    state.update({"plan": req.plan, "understanding": req.understanding, "code_files": {}})
    return _sse_response(stream_build(state, sid, explain=True))


@app.post("/api/fix")
async def api_fix(req: ApiRequest):
    if not req.session_id:
        raise HTTPException(status_code=400, detail="session_id is required")
    _require_session(req.session_id)
    sid = req.session_id

    lv = latest_version(sid)
    if not lv:
        raise HTTPException(status_code=404, detail="no version")
    files_cur = await asyncio.to_thread(_read_version_files, sid, lv)
    if not files_cur:
        raise HTTPException(status_code=404, detail="no files in latest version")

    state = _base_state(req, sid)
    state.update(
        {
            "plan": req.plan,
            "understanding": req.understanding,
            "errors": [req.error] if req.error else [],
            "code_files": files_cur,
        }
    )
    return _sse_response(stream_build(state, sid, explain=False))


@app.post("/api/sample")
async def api_sample():
    sid = _new_session_id()
    try:
        v = await asyncio.to_thread(save_version, sid, SAMPLE_FILES)
    except Exception:
        logger.exception("failed to save sample")
        raise HTTPException(status_code=500, detail="could not create sample")

    async def gen():
        yield sse(
            "understand_done",
            {
                "understanding": {
                    "app_type": "Habit tracker",
                    "target_users": "self-improvers",
                    "core_features": {"features": ["add habits", "toggle complete", "stats"]},
                    "is_clear": True,
                },
                "questions": [],
                "plan": {},
            },
            sid,
        )
        yield sse(
            "plan_done",
            {
                "plan": {
                    "screens": [
                        {"name": "Home", "purpose": "List habits"},
                        {"name": "Stats", "purpose": "Show progress"},
                    ],
                    "features": ["Add habit", "Toggle done"],
                    "tech_stack": "HTML, CSS, Vanilla JS",
                    "data_model": "Saved in localStorage (JSON)",
                    "build_steps": ["Build UI", "Add interactions"],
                }
            },
            sid,
        )
        yield sse("build_done", {"version": v, "files": SAMPLE_FILES}, sid)
        yield sse("explain_done", {"explanation": {}}, sid)
        yield sse("learn_done", {"learning": {}}, sid)
        yield sse("done", {}, sid)

    return _sse_response(gen())
