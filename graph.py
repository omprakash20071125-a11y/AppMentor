import io
import json
import logging
import os
import random
import re
import shutil
import subprocess
import tempfile
import time
import zipfile
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple, Type, TypedDict, TypeVar

from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, model_validator

try: 
    from google.api_core import exceptions as _gexc

    _TRANSIENT_TYPES: Tuple[type, ...] = (
        _gexc.ResourceExhausted,
        _gexc.TooManyRequests,
        _gexc.ServiceUnavailable,
        _gexc.DeadlineExceeded,
        _gexc.InternalServerError,
    )
except Exception:  # pragma: no cover
    _TRANSIENT_TYPES = ()

load_dotenv()
logger = logging.getLogger("appmentor")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Default is outside the project folder on purpose: Live Server / uvicorn --reload / IDE watchers
# reload the page or server whenever a file inside the project changes, and every build writes files.
GENERATED_DIR = os.getenv("GENERATED_DIR") or os.path.join(os.path.expanduser("~"), ".appmentor", "generated")

ALLOWED_FILES = ("index.html", "style.css", "app.js")
MAX_BUILD_RETRIES = 1      # extra build attempts after the first one
LLM_MAX_ATTEMPTS = 3       # network / rate-limit retries per LLM call
DEFAULT_MAX_TOKENS = 16384
MAX_TOKENS_CAP = 65536     # upper bound when we raise the limit after a truncated reply
MAX_SCREENS = 4
TECH_STACK = "HTML, CSS, Vanilla JS"

def get_llm(max_tokens: Optional[int] = None) -> ChatGoogleGenerativeAI:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY (or GOOGLE_API_KEY) in your .env file")
    model_name = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
    temperature = float(os.getenv("GEMINI_TEMPERATURE", "0.1"))
    kwargs: Dict[str, Any] = dict(
        model=model_name,
        temperature=temperature,
        google_api_key=api_key,
        response_mime_type="application/json",  # makes Gemini emit strictly valid JSON
        max_output_tokens=max_tokens or int(os.getenv("GEMINI_MAX_TOKENS", str(DEFAULT_MAX_TOKENS))),
    )
    thinking = os.getenv("GEMINI_THINKING_BUDGET")
    if thinking:
        kwargs["thinking_budget"] = int(thinking)
    return ChatGoogleGenerativeAI(**kwargs)

# pydantic schemas

class CoreFeatures(BaseModel):
    features: List[str] = Field(default_factory=list)


class Understanding(BaseModel):
    app_type: str = Field(default="")
    target_users: str = Field(default="")
    core_features: CoreFeatures = Field(default_factory=CoreFeatures)
    is_clear: bool = Field(default=True)


class PlanScreen(BaseModel):
    name: str = Field(default="")
    purpose: str = Field(default="")


class Plan(BaseModel):
    screens: List[PlanScreen] = Field(default_factory=list)
    features: List[str] = Field(default_factory=list)
    tech_stack: str = Field(default="")
    data_model: str = Field(default="")
    build_steps: List[str] = Field(default_factory=list)


class UnderstandPlanOutput(BaseModel):
    understanding: Understanding
    questions: List[str] = Field(default_factory=list)
    plan: Plan = Field(default_factory=Plan)


class BuildOutput(BaseModel):
    files: Dict[str, str] = Field(default_factory=dict)


class Concept(BaseModel):
    term: str
    definition: str


class QuizOption(BaseModel):
    text: str


class QuizQuestion(BaseModel):
    question: str
    options: List[QuizOption]
    correct_answer: str
    explanation: str

    @model_validator(mode="after")
    def _answer_must_be_an_option(self):
        texts = [o.text for o in self.options]
        if self.correct_answer in texts:
            return self
        norm = self.correct_answer.strip().lower()
        for t in texts:  # tolerate case / whitespace differences
            if t.strip().lower() == norm:
                self.correct_answer = t
                return self
        m = re.fullmatch(r"\(?([A-Da-d])[\).:]?", self.correct_answer.strip())
        if m: 
            idx = ord(m.group(1).upper()) - ord("A")
            if idx < len(texts):
                self.correct_answer = texts[idx]
                return self
        raise ValueError("correct_answer must be the exact text of one of the options")


class Exercise(BaseModel):
    title: str
    description: str


class ExplainLearnOutput(BaseModel):
    architecture_overview: str
    per_file_explanations: Dict[str, str] = Field(default_factory=dict)
    key_concepts: List[Concept] = Field(default_factory=list)
    quiz: List[QuizQuestion] = Field(default_factory=list)
    exercises: List[Exercise] = Field(default_factory=list)
    next_steps: List[str] = Field(default_factory=list)

# prompts template 

UNDERSTAND_PLAN_PROMPT = """You are an AI product assistant for AppMentor. Role: Understand + Plan.

INPUT: user_idea (string), experience_level (string: beginner|intermediate|advanced),
optionally clarifying_answers (the user's answers to earlier questions).

OUTPUT: Return valid JSON matching the UnderstandPlanOutput schema exactly. No markdown fences.

RULES:
- understanding: infer app_type, target_users, core_features (max 5), is_clear (bool).
- If is_clear is false: questions = at most 3 clarifying questions (concise). In that case, return plan as empty/default (do NOT fill plan).
- If is_clear is true: questions = []. Create a minimal, actionable plan with:
  - screens: max 4. Each {name, purpose}
  - features: max 6 (actionable items)
  - tech_stack: exactly "HTML, CSS, Vanilla JS" (the app runs in a sandboxed preview: no backend, no libraries)
  - data_model: one short line; user data is saved with localStorage (JSON), no backend, no cookies, no other browser storage
  - build_steps: max 6 (ordered)
- If clarifying_answers are provided, treat them as final: questions must be [] and the plan must be filled.
- Keep it compact. No extra text.
"""

BUILD_PROMPT = """You are a front-end developer building a mobile-style web app for AppMentor.

INPUT: user_idea, understanding, plan, experience_level, clarifying_answers (optional),
errors_to_fix (optional list), current_files (optional, labeled code blocks).

OUTPUT: Return ONLY JSON: { "files": { "index.html": "...", "style.css": "...", "app.js": "..." } } No markdown fences, no extra text.

RULES (must follow exactly):
- Mobile-style for 375x812. Plain HTML, CSS, vanilla JS. No libraries, CDNs, remote images, fonts, or API calls. Emoji or inline SVG for icons only.
- Exactly three files: index.html, style.css, app.js, linked with relative paths (href="style.css", src="app.js").
- Mock data only, no remote APIs. Persist user data with window.localStorage (JSON): load it on start and save it on every change, always inside try/catch with a safe default. No sessionStorage, cookies or IndexedDB.
- Do NOT use alert(), prompt() or confirm() (blocked in the sandbox). Use inline inputs, buttons and on-page messages instead.
- Screens are <section> elements that app.js shows/hides. Use a bottom tab bar OR back buttons for navigation. 3-4 screens max, working interactions.
- Every element id that app.js looks up must exist in index.html. Put <script src="app.js"></script> at the end of <body>.
- No placeholders, no TODOs. Output valid, complete, non-empty code for all three. Keep the code compact.
- If errors_to_fix is provided, fix them with minimal changes preserving structure (current_files shows the existing code).
"""

EXPLAIN_LEARN_PROMPT = """You are a mentor + teacher for AppMentor.

INPUT: user_idea, understanding, plan, code_files (labeled code blocks), experience_level

OUTPUT: Valid JSON matching ExplainLearnOutput. No markdown fences.

REQUIREMENTS:
- architecture_overview: max 5 sentences
- per_file_explanations: keys exactly index.html, style.css, app.js (if present). 2-3 sentences each, short.
- key_concepts: exactly 4. Each {term, definition} (one-line definition)
- quiz: exactly 3 multiple-choice questions. Each has 3-4 options. correct_answer = exact text of the correct option. explanation = one line.
- exercises: exactly 2. Each {title, description}
- next_steps: exactly 3 (concise)
Adapt depth to experience_level (beginner = clearer/simpler). Keep concise overall.
"""



# SAMPLE APP (shown by /api/sample, no LLM needed)


SAMPLE_FILES = {
    "index.html": """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no" />
  <title>Habit Tracker</title>
  <link rel="stylesheet" href="style.css" />
</head>
<body>
  <main class="app">
    <section id="home" class="screen active">
      <header class="header">
        <h1>Habits</h1>
      </header>
      <div class="add-row">
        <input id="habitInput" type="text" placeholder="New habit" maxlength="40" />
        <button id="addBtn" class="btn-primary">+ Add</button>
      </div>
      <ul id="habitList" class="habit-list"></ul>
      <nav class="tabbar">
        <button class="tab active" data-screen="home">Home</button>
        <button class="tab" data-screen="stats">Stats</button>
      </nav>
    </section>
    <section id="stats" class="screen">
      <header class="header">
        <h1>Stats</h1>
      </header>
      <div class="stats">
        <div class="stat-card">
          <span class="stat-label">Total</span>
          <span id="statTotal" class="stat-value">0</span>
        </div>
        <div class="stat-card">
          <span class="stat-label">Completed</span>
          <span id="statDone" class="stat-value">0</span>
        </div>
      </div>
      <nav class="tabbar">
        <button class="tab" data-screen="home">Home</button>
        <button class="tab active" data-screen="stats">Stats</button>
      </nav>
    </section>
  </main>
  <script src="app.js"></script>
</body>
</html>
""",
    "style.css": """* { box-sizing: border-box; margin: 0; padding: 0; }
html, body { height: 100%; }
body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; background: #f5f7ff; color: #111827; }
.app { width: 375px; height: 812px; margin: 0 auto; background: #fff; border: 1px solid #e5e7eb; border-radius: 24px; overflow: hidden; display: flex; flex-direction: column; position: relative; }
.screen { flex: 1; display: none; flex-direction: column; padding: 16px 16px 64px; overflow: auto; }
.screen.active { display: flex; }
.header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }
h1 { font-size: 20px; }
.add-row { display: flex; gap: 8px; margin-bottom: 12px; }
.add-row input { flex: 1; min-width: 0; padding: 8px 10px; border: 1px solid #e5e7eb; border-radius: 10px; font: inherit; }
.btn-primary { background: #4F46E5; color: #fff; border: 0; padding: 8px 10px; border-radius: 10px; }
.habit-list { list-style: none; display: flex; flex-direction: column; gap: 8px; }
.habit-item { display: flex; justify-content: space-between; align-items: center; background: #f8faff; padding: 10px 12px; border-radius: 12px; border: 1px solid #eef2ff; }
.habit-item.done { background: #eef2ff; }
.tabbar { position: absolute; bottom: 0; left: 0; right: 0; height: 56px; display: flex; border-top: 1px solid #f1f5f9; background: #fff; }
.tab { flex: 1; border: 0; background: #fff; color: #6b7280; }
.tab.active { color: #4F46E5; font-weight: 600; }
.stats { display: flex; gap: 10px; }
.stat-card { flex: 1; background: #f8faff; border: 1px solid #eef2ff; border-radius: 12px; padding: 14px; text-align: center; }
.stat-label { display: block; color: #6b7280; font-size: 12px; }
.stat-value { font-size: 24px; font-weight: 700; color: #4F46E5; }
""",
    "app.js": """const KEY = 'habits-v1';
let habits = [];
try { habits = JSON.parse(localStorage.getItem(KEY)) || []; } catch (e) { habits = []; }
let id = habits.reduce((m, h) => Math.max(m, h.id), 0);
function save() { try { localStorage.setItem(KEY, JSON.stringify(habits)); } catch (e) {} }

const habitList = document.getElementById('habitList');
const habitInput = document.getElementById('habitInput');
const addBtn = document.getElementById('addBtn');
const statTotal = document.getElementById('statTotal');
const statDone = document.getElementById('statDone');
const tabs = document.querySelectorAll('.tab');

function showScreen(name) {
  document.querySelectorAll('.screen').forEach(s => s.classList.toggle('active', s.id === name));
}

tabs.forEach(t => t.addEventListener('click', () => showScreen(t.dataset.screen)));

function addHabit() {
  const name = habitInput.value.trim();
  if (!name) return;
  habits.push({ id: ++id, name: name, done: false });
  habitInput.value = '';
  render();
}

addBtn.addEventListener('click', addHabit);
habitInput.addEventListener('keydown', e => { if (e.key === 'Enter') addHabit(); });

function toggleHabit(hid) {
  const h = habits.find(x => x.id === hid);
  if (h) { h.done = !h.done; render(); }
}

function render() {
  habitList.innerHTML = '';
  habits.forEach(h => {
    const li = document.createElement('li');
    li.className = 'habit-item' + (h.done ? ' done' : '');
    const span = document.createElement('span');
    span.textContent = h.name;
    const btn = document.createElement('button');
    btn.textContent = h.done ? 'Undo' : 'Done';
    btn.addEventListener('click', () => toggleHabit(h.id));
    li.appendChild(span);
    li.appendChild(btn);
    habitList.appendChild(li);
  });
  statTotal.textContent = habits.length;
  statDone.textContent = habits.filter(h => h.done).length;
  save();
}

render();
""",
}

_VERSION_DIR_RE = re.compile(r"v(\d+)$")


def _safe_session_id(s: str) -> bool:
    return bool(s) and bool(re.fullmatch(r"[A-Za-z0-9_-]+", s))


def _session_path(session_id: str) -> str:
    return os.path.join(GENERATED_DIR, session_id)


def _touch(path: str) -> None:
    """Mark a session as recently used so cleanup doesn't delete it mid-use."""
    try:
        os.utime(path, None)
    except OSError:
        pass


def _write_atomic(path: str, text: str) -> None:
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def save_version(session_id: str, files: Dict[str, str]) -> str:
    if not _safe_session_id(session_id):
        raise ValueError("invalid session_id")
    os.makedirs(GENERATED_DIR, exist_ok=True)
    sp = _session_path(session_id)
    os.makedirs(sp, exist_ok=True)

    # Start from the highest existing vN directory (not just latest.txt) and claim the
    # directory atomically, so concurrent saves can never overwrite each other.
    existing = [
        int(m.group(1))
        for d in os.listdir(sp)
        if (m := _VERSION_DIR_RE.match(d)) and os.path.isdir(os.path.join(sp, d))
    ]
    next_n = (max(existing) if existing else 0) + 1
    while True:
        vname = f"v{next_n}"
        vdir = os.path.join(sp, vname)
        try:
            os.makedirs(vdir)  
            break
        except FileExistsError:
            next_n += 1

    
    files = files or {}
    ignored = [k for k in files if k not in ALLOWED_FILES]
    if ignored:
        logger.warning("save_version: ignoring unexpected file keys %r", ignored)
    for name in ALLOWED_FILES:
        if name in files:
            with open(os.path.join(vdir, name), "w", encoding="utf-8") as f:
                f.write(files[name] or "")

    _write_atomic(os.path.join(sp, "latest.txt"), vname)
    _touch(sp)
    return vname


def latest_version(session_id: str) -> Optional[str]:
    if not _safe_session_id(session_id):
        return None
    sp = _session_path(session_id)
    try:
        with open(os.path.join(sp, "latest.txt"), "r", encoding="utf-8") as f:
            v = f.read().strip() or None
    except Exception:
        return None
    if v:
        _touch(sp)
    return v


def build_zip(session_id: str, version: Optional[str]) -> Tuple[Optional[bytes], Optional[str]]:
    if not _safe_session_id(session_id):
        return None, None
    if version is None or version == "latest":
        version = latest_version(session_id)
    if version is None or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,31}", version):
        return None, None

    vdir = os.path.join(_session_path(session_id), version)
    if not os.path.isdir(vdir):
        return None, None

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for n in ALLOWED_FILES:
            p = os.path.join(vdir, n)
            if os.path.exists(p):
                z.write(p, arcname=n)
    _touch(_session_path(session_id))
    return buf.getvalue(), version


def _session_last_activity(sp: str) -> float:
    """Newest mtime of the session folder, its latest.txt and its version folders."""
    newest = os.path.getmtime(sp)
    try:
        for name in os.listdir(sp):
            try:
                newest = max(newest, os.path.getmtime(os.path.join(sp, name)))
            except OSError:
                pass
    except OSError:
        pass
    return newest


def cleanup_old_sessions(max_age_hours: int = 6) -> int:
    removed = 0
    cutoff = time.time() - max_age_hours * 3600
    try:
        if not os.path.isdir(GENERATED_DIR):
            return 0
        for name in os.listdir(GENERATED_DIR):
            sp = os.path.join(GENERATED_DIR, name)
            if os.path.isdir(sp) and _session_last_activity(sp) < cutoff:
                try:
                    shutil.rmtree(sp)
                    removed += 1
                except Exception:
                    pass
    except Exception:
        pass
    return removed


# main graph

T = TypeVar("T", bound=BaseModel)


class State(TypedDict, total=False):
    session_id: str
    user_idea: str
    experience_level: str
    understanding: Dict[str, Any]
    questions: List[str]
    answers: List[str]
    plan: Dict[str, Any]
    code_files: Dict[str, str]
    errors: List[str]
    retry_count: int
    version: str
    explanation: Dict[str, Any]
    learning: Dict[str, Any]
    status: str                 
    explanation_error: str      


@lru_cache(maxsize=4)
def _llm(max_tokens: Optional[int] = None):
    """Create the LLM client once per token limit."""
    return get_llm(max_tokens)


def _is_transient(exc: Exception) -> bool:
    """True only for errors worth retrying (rate limit, 5xx, timeouts)."""
    if _TRANSIENT_TYPES and isinstance(exc, _TRANSIENT_TYPES):
        return True
    msg = str(exc).lower()
    if re.search(r"\b(429|500|502|503|504)\b", msg):
        return True
    return any(
        k in msg
        for k in ("resource exhausted", "rate limit", "unavailable", "overloaded", "deadline", "timed out", "timeout")
    )


def _invoke(prompt: str, max_tokens: Optional[int] = None):
    """Call the model, retrying only transient errors with exponential backoff."""
    last: Optional[Exception] = None
    for attempt in range(LLM_MAX_ATTEMPTS):
        try:
            return _llm(max_tokens).invoke(prompt)
        except Exception as e:
            if not _is_transient(e) or attempt == LLM_MAX_ATTEMPTS - 1:
                raise
            last = e
            delay = min(2 ** attempt, 20) + random.random()
            logger.warning("transient LLM error (%s); retrying in %.1fs", e, delay)
            time.sleep(delay)
    raise last  # type: ignore[misc]  # unreachable


def _was_truncated(res) -> bool:
    meta = getattr(res, "response_metadata", None) or {}
    return "MAX_TOKENS" in str(meta.get("finish_reason", "")).upper()


def _clean_json(text: str) -> str:
    """Strip code fences and any text around the outermost JSON object."""
    if not text:
        return text
    t = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", t, re.DOTALL)
    if fence:
        t = fence.group(1).strip()
    start, end = t.find("{"), t.rfind("}")
    if start != -1 and end > start:
        t = t[start : end + 1]
    return t


def _safe_parse(parser: PydanticOutputParser, text: str):
    """Try several increasingly lenient ways to turn model text into the schema."""
    cleaned = _clean_json(text)
    attempts = (
        lambda: parser.parse(text),
        lambda: parser.parse(cleaned),
        # strict=False tolerates raw newlines/tabs inside string values (common in code)
        lambda: parser.pydantic_object.model_validate(json.loads(cleaned, strict=False)),
    )
    last: Exception = ValueError("empty output")
    for attempt in attempts:
        try:
            return attempt()
        except Exception as e:
            last = e
    preview = (text or "")[:300].replace("\n", " ")
    raise ValueError(
        f"Could not parse LLM output into {parser.pydantic_object.__name__}: {last}. "
        f"Output started with: {preview!r}"
    ) from last


def _llm_text(res) -> str:
    """Extract plain text from a model reply (ignores 'thinking' blocks)."""
    content = getattr(res, "content", None)
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, str):
                parts.append(b)
            elif isinstance(b, dict) and b.get("type", "text") == "text":
                parts.append(b.get("text", ""))
        content = "".join(parts)
    return content if isinstance(content, str) else str(res)


def _run_llm(base_prompt: str, schema: Type[T], variables: Dict[str, str], attempts: int = 2) -> T:
    parser = PydanticOutputParser(pydantic_object=schema)
    parts = [base_prompt, parser.get_format_instructions()]
    parts += [f"{k}: {v}" for k, v in variables.items()]
    prompt = "\n\n".join(parts)

    base_tokens = int(os.getenv("GEMINI_MAX_TOKENS", str(DEFAULT_MAX_TOKENS)))
    max_tokens: Optional[int] = None
    hint: Optional[str] = None
    last_err: Optional[Exception] = None

    for i in range(attempts):
        full = prompt
        if hint:
            full += "\n\n" + hint
        res = _invoke(full, max_tokens)
        text = _llm_text(res)
        truncated = _was_truncated(res)
        try:
            return _safe_parse(parser, text)
        except ValueError as e:
            last_err = e
            if truncated:
                # The reply was cut off, so "escape your quotes" would not help.
                max_tokens = min((max_tokens or base_tokens) * 2, MAX_TOKENS_CAP)
                hint = (
                    "Your previous reply was cut off by the output limit. "
                    "Reply again with ONE complete valid JSON object and keep the code compact."
                )
                logger.warning("output truncated (attempt %d/%d); raising limit to %d", i + 1, attempts, max_tokens)
            else:
                hint = (
                    f"Your previous reply could not be parsed ({str(e)[:300]}). "
                    "Reply again with ONLY one valid JSON object that matches the schema. "
                    "Escape every quote and newline inside string values."
                )
                logger.warning("parse failed (attempt %d/%d): %s\nRAW OUTPUT:\n%s", i + 1, attempts, e, text[:2000])
    raise last_err  # type: ignore[misc]


def _format_files(files: Dict[str, str]) -> str:
    if not files:
        return "(none)"
    return "\n\n".join(f"--- FILE: {n} ---\n```\n{c}\n```" for n, c in files.items())


def _format_qa(questions: List[str], answers: List[str]) -> str:
    if not answers:
        return "(none)"
    lines = []
    for i, a in enumerate(answers):
        q = questions[i] if i < len(questions) else f"Question {i + 1}"
        lines.append(f"Q: {q}\nA: {a}")
    return "\n".join(lines)


def _normalize_plan(plan: Plan) -> Dict[str, Any]:
    """Enforce in code the limits the prompt only asks for."""
    d = plan.model_dump()
    d["screens"] = d["screens"][:MAX_SCREENS]
    d["features"] = d["features"][:6]
    d["build_steps"] = d["build_steps"][:6]
    d["tech_stack"] = TECH_STACK
    d["data_model"] = d["data_model"].strip() or "Saved in localStorage (JSON)"
    return d


# ---- Graph A: nodes --------------------------------------------------------

def node_understand_plan(state: State) -> State:
    answers = state.get("answers") or []
    variables = {
        "user_idea": state.get("user_idea", ""),
        "experience_level": state.get("experience_level", "beginner"),
    }
    if answers:
        variables["clarifying_answers"] = _format_qa(state.get("questions") or [], answers)
        variables["note"] = (
            "The user has already answered the clarifying questions above. "
            "Treat them as final: return an empty questions list and produce the plan."
        )

    parsed: Optional[UnderstandPlanOutput] = None
    wants_plan = True
    for _ in range(2):
        parsed = _run_llm(UNDERSTAND_PLAN_PROMPT, UnderstandPlanOutput, variables)
        # A plan is required when answers were given, the idea is clear, or the model
        # said "unclear" but asked no questions (nothing to ask means nothing to wait for).
        wants_plan = bool(answers) or parsed.understanding.is_clear or not parsed.questions
        if not wants_plan or (parsed.plan.screens and parsed.plan.build_steps):
            break
        variables["note"] = (
            variables.get("note", "")
            + " Your previous reply had an empty plan. The plan is REQUIRED: "
            "fill screens, features, data_model and build_steps."
        ).strip()

    assert parsed is not None
    if wants_plan and not parsed.plan.screens:
        raise ValueError("Model returned an empty plan for a clear idea")

    und = parsed.understanding.model_dump()
    und["core_features"]["features"] = und["core_features"]["features"][:5]

    if wants_plan:
        und["is_clear"] = True
        out: State = {"understanding": und, "plan": _normalize_plan(parsed.plan)}
        if not answers:
            out["questions"] = []
        # With answers, leave state["questions"] alone so the Q/A pairing is preserved.
        return out

    return {
        "understanding": und,
        "questions": [q for q in parsed.questions if q.strip()][:3],
        "plan": Plan().model_dump(),
    }


# ---- Graph B: nodes --------------------------------------------------------

def node_init_build(state: State) -> State:
    """Reset the retry budget. `errors` is left alone: /api/fix passes the preview error in
    through it, and node_validate overwrites it after every build anyway."""
    return {"retry_count": 0, "status": "building"}


def node_build(state: State) -> State:
    parsed = _run_llm(
        BUILD_PROMPT,
        BuildOutput,
        {
            "user_idea": state.get("user_idea", ""),
            "understanding": json.dumps(state.get("understanding") or {}),
            "plan": json.dumps(state.get("plan") or {}),
            "experience_level": state.get("experience_level", "beginner"),
            "clarifying_answers": _format_qa(
                state.get("questions") or [], state.get("answers") or []
            ),
            "errors_to_fix": json.dumps(state.get("errors") or []),
            "current_files": _format_files(state.get("code_files") or {}),
        },
    )
    files = parsed.files or {}
    extra = [k for k in files if k not in ALLOWED_FILES]
    if extra:
        logger.warning("build returned unexpected file keys %r; dropping them", extra)
    # retry_count is owned by node_validate; build only replaces the files.
    return {"code_files": {k: v for k, v in files.items() if k in ALLOWED_FILES}}


# Remove comments and string literals before counting brackets / scanning for APIs, so
# text like ":)" or "{" inside a string doesn't cause false errors.
_JS_NOISE = re.compile(
    r"//[^\n]*"
    r"|/\*.*?\*/"
    r"|'(?:\\.|[^'\\\n])*'"
    r'|"(?:\\.|[^"\\\n])*"'
    r"|`(?:\\.|[^`\\])*`",
    re.DOTALL,
)
_CSS_NOISE = re.compile(
    r"/\*.*?\*/"
    r"|'(?:\\.|[^'\\\n])*'"
    r'|"(?:\\.|[^"\\\n])*"',
    re.DOTALL,
)

# APIs that the prompt forbids and the sandboxed preview blocks or throws on.
_BANNED_STORAGE = re.compile(r"\b(sessionStorage|indexedDB)\b|document\s*\.\s*cookie")
_BANNED_DIALOGS = re.compile(r"(?<![\w$.])(?:window\s*\.\s*)?(?:alert|confirm|prompt)\s*\(")
_BANNED_NETWORK = re.compile(r"(?<![\w$.])fetch\s*\(|\bXMLHttpRequest\b|\bnew\s+WebSocket\b|\bsendBeacon\b|\bimport\s*\(")
_REMOTE_ATTR = re.compile(r"""\b(?:src|href)\s*=\s*["']\s*(?:https?:)?//""", re.IGNORECASE)
_REMOTE_CSS = re.compile(r"""url\(\s*["']?\s*(?:https?:)?//|@import\b""", re.IGNORECASE)

_SCRIPT_TAG = re.compile(r"<script\b[^>]*src\s*=\s*[\"']\.?/?app\.js[\"'][^>]*>", re.IGNORECASE)
_LINK_TAG = re.compile(r"<link\b[^>]*href\s*=\s*[\"']\.?/?style\.css[\"'][^>]*>", re.IGNORECASE)


def _node_syntax_error(js: str) -> Optional[str]:
    """Real JS syntax check via `node --check` when Node is installed."""
    node = shutil.which("node")
    if not node:
        return None
    path = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False, encoding="utf-8") as f:
            f.write(js)
            path = f.name
        r = subprocess.run([node, "--check", path], capture_output=True, text=True, timeout=10)
        if r.returncode != 0:
            lines = [ln for ln in r.stderr.splitlines() if ln.strip()]
            detail = next((ln for ln in lines if "Error" in ln), lines[0] if lines else "syntax error")
            return f"app.js syntax error: {detail[:250]}"
    except Exception as e:  # node missing/timeout: skip, don't block the build
        logger.warning("node --check skipped: %s", e)
    finally:
        if path:
            try:
                os.remove(path)
            except OSError:
                pass
    return None


def validate_files(files: Dict[str, str]) -> List[str]:
    errs: List[str] = []

    for name in ALLOWED_FILES:
        if not files or name not in files:
            errs.append(f"missing {name}")
        elif not files[name] or not files[name].strip():
            errs.append(f"{name} is empty")
    if errs:
        return errs

    html, raw_js, raw_css = files["index.html"], files["app.js"], files["style.css"]

    # --- HTML wiring ---
    if not _LINK_TAG.search(html):
        errs.append('index.html must include <link rel="stylesheet" href="style.css">')
    script = _SCRIPT_TAG.search(html)
    if not script:
        errs.append('index.html must include <script src="app.js"></script>')
    else:
        head_end = html.lower().find("</head>")
        in_head = head_end != -1 and script.start() < head_end
        if in_head and not re.search(r"\b(defer|type\s*=\s*[\"']module[\"'])", script.group(0), re.IGNORECASE):
            errs.append("app.js is loaded in <head> without defer; move the script to the end of <body>")

    n_sections = len(re.findall(r"<section\b", html, re.IGNORECASE))
    if n_sections < 1 or n_sections > MAX_SCREENS:
        errs.append(f"index.html has {n_sections} <section> screens; use 1 to {MAX_SCREENS}")

    # --- Banned APIs / remote resources (strings and comments removed first) ---
    js = _JS_NOISE.sub("", raw_js)
    css = _CSS_NOISE.sub("", raw_css)
    if _BANNED_STORAGE.search(js):
        errs.append("app.js uses sessionStorage/indexedDB/cookies; use localStorage (inside try/catch) instead")
    if _BANNED_DIALOGS.search(js):
        errs.append("app.js uses alert()/confirm()/prompt(); use on-page messages and inline inputs")
    if _BANNED_NETWORK.search(js):
        errs.append("app.js makes network requests (fetch/XHR/WebSocket/import()); use mock data only")
    if _REMOTE_ATTR.search(html):
        errs.append("index.html references a remote URL in src/href; no CDNs or remote assets")
    if _REMOTE_CSS.search(raw_css):
        errs.append("style.css uses a remote url() or @import; no remote assets or fonts")

    # --- Cheap structural sanity checks ---
    for name, code, pairs in (
        ("app.js", js, (("{", "}"), ("(", ")"), ("[", "]"))),
        ("style.css", css, (("{", "}"),)),
    ):
        for o, c in pairs:
            if code.count(o) != code.count(c):
                errs.append(f"{name} has unbalanced '{o}{c}'")

    # --- Real syntax check (if Node is available) ---
    syntax = _node_syntax_error(raw_js)
    if syntax:
        errs.append(syntax)

    # --- JS looks up ids that don't exist in the HTML ---
    html_ids = set(re.findall(r"""\bid\s*=\s*["']([^"']+)["']""", html, re.IGNORECASE))
    used_ids = set(re.findall(r"""getElementById\(\s*["']([^"']+)["']\s*\)""", raw_js))
    used_ids |= set(re.findall(r"""querySelector(?:All)?\(\s*["']#([\w-]+)["']\s*\)""", raw_js))
    missing = sorted(used_ids - html_ids)
    if missing:
        errs.append(f"app.js references ids that are not in index.html: {missing}")

    return errs[:10]


def node_validate(state: State) -> State:
    errs = validate_files(state.get("code_files") or {})
    rc = state.get("retry_count") or 0
    return {"errors": errs, "retry_count": rc + 1 if errs else rc}


def should_retry_build(state: State) -> str:
    errs = state.get("errors") or []
    rc = state.get("retry_count") or 0
    if not errs:
        return "explain_learn"
    if rc <= MAX_BUILD_RETRIES:
        return "retry_build"
    return "fail"  # still invalid after all retries


def node_fail(state: State) -> State:
    """Explicit failure state so callers can check `status` instead of inferring from errors."""
    return {"status": "failed"}


def _empty_explanation() -> Dict[str, Any]:
    return {
        "architecture_overview": "",
        "per_file_explanations": {},
        "key_concepts": [],
        "quiz": [],
        "exercises": [],
        "next_steps": [],
    }


def node_explain_learn(state: State) -> State:
    """Explanation is a bonus: if it fails, keep the successful build."""
    try:
        parsed = _run_llm(
            EXPLAIN_LEARN_PROMPT,
            ExplainLearnOutput,
            {
                "user_idea": state.get("user_idea", ""),
                "understanding": json.dumps(state.get("understanding") or {}),
                "plan": json.dumps(state.get("plan") or {}),
                "code_files": _format_files(state.get("code_files") or {}),
                "experience_level": state.get("experience_level", "beginner"),
            },
        )
    except Exception as e:
        logger.exception("explain_learn failed; returning build without explanation")
        payload = _empty_explanation()
        return {
            "explanation": payload,
            "learning": json.loads(json.dumps(payload)),
            "explanation_error": str(e)[:300],
            "status": "ok",
        }

    payload = {
        "architecture_overview": parsed.architecture_overview,
        "per_file_explanations": {
            k: v for k, v in (parsed.per_file_explanations or {}).items() if k in ALLOWED_FILES
        },
        "key_concepts": [c.model_dump() for c in parsed.key_concepts][:4],
        "quiz": [q.model_dump() for q in parsed.quiz][:3],
        "exercises": [e.model_dump() for e in parsed.exercises][:2],
        "next_steps": (parsed.next_steps or [])[:3],
    }
    return {"explanation": payload, "learning": json.loads(json.dumps(payload)), "status": "ok"}


# ---- Graph A: understand + plan --------------------------------------------

graph_a = StateGraph(State)
graph_a.add_node("understand_plan", node_understand_plan)
graph_a.add_edge(START, "understand_plan")
graph_a.add_edge("understand_plan", END)

# ---- Graph B: init -> build -> validate -> (retry | explain | fail) --------

graph_b = StateGraph(State)
graph_b.add_node("init", node_init_build)
graph_b.add_node("build", node_build)
graph_b.add_node("validate", node_validate)
graph_b.add_node("explain_learn", node_explain_learn)
graph_b.add_node("fail", node_fail)

graph_b.add_edge(START, "init")
graph_b.add_edge("init", "build")
graph_b.add_edge("build", "validate")
graph_b.add_conditional_edges(
    "validate",
    should_retry_build,
    {"retry_build": "build", "explain_learn": "explain_learn", "fail": "fail"},
)
graph_b.add_edge("explain_learn", END)
graph_b.add_edge("fail", END)

app_graph_a = graph_a.compile()
app_graph_b = graph_b.compile()