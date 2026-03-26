"""
fred_ui.py — Streamlit UI for the FRED Financial Q&A Agent

Run:
    uv run --with anthropic --with requests --with pandas --with plotly --with streamlit \\
        streamlit run fred_ui.py
"""

import json
import os
from datetime import date
from pathlib import Path

# Load .env from the project directory before anything else reads os.environ
_env_file = Path(__file__).parent / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())

import anthropic
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="FRED Financial Q&A",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

FRED_BASE = "https://api.stlouisfed.org/fred"
MODEL = "claude-sonnet-4-6"

# Keys: .env (local) → st.secrets (Streamlit Cloud) → env var
def _secret(key: str) -> str:
    if val := os.environ.get(key):
        return val
    try:
        return st.secrets.get(key, "")
    except Exception:
        return ""

FRED_KEY = _secret("FRED_API_KEY")
CLAUDE_KEY = _secret("ANTHROPIC_API_KEY")

EXAMPLE_QUESTIONS = [
    "What was the Fed Funds Rate during the 2008 financial crisis?",
    "How did CPI inflation change after COVID?",
    "Compare unemployment and GDP growth since 2015",
    "Show the yield curve inversion before the 2020 recession",
    "How has the Fed balance sheet changed since 2008?",
]


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def _init_state() -> None:
    defaults: dict = {
        "claude_history": [],    # raw API messages (incl. tool_use blocks)
        "display_history": [],   # [{role, text, charts: [fig, ...]}]
        "data_store": {},        # series_id → pd.DataFrame
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# ---------------------------------------------------------------------------
# FRED tool functions
# ---------------------------------------------------------------------------

def _fred_get(endpoint: str, params: dict) -> dict:
    r = requests.get(
        f"{FRED_BASE}/{endpoint}",
        params={**params, "api_key": FRED_KEY, "file_type": "json"},
        timeout=15,
    )
    r.raise_for_status()
    return r.json()


def tool_search_fred_series(query: str, limit: int = 6) -> str:
    data = _fred_get("series/search", {"search_text": query, "limit": limit})
    series = data.get("seriess", [])
    if not series:
        return json.dumps({"error": f"No results for: {query}"})
    results = sorted(
        [{"id": s["id"], "title": s["title"],
          "frequency": s.get("frequency_short", ""),
          "units": s.get("units_short", ""),
          "popularity": s.get("popularity", 0)}
         for s in series],
        key=lambda x: -x["popularity"],
    )
    return json.dumps({"series": results})


def tool_fetch_fred_data(series_id: str, start_date: str, end_date: str) -> str:
    try:
        raw = _fred_get("series/observations", {
            "series_id": series_id,
            "observation_start": start_date,
            "observation_end": end_date,
        })
    except requests.HTTPError as exc:
        return json.dumps({"error": str(exc)})

    obs = [o for o in raw.get("observations", []) if o["value"] != "."]
    if not obs:
        return json.dumps({"error": f"No data for {series_id} in {start_date}–{end_date}"})

    df = pd.DataFrame(obs)[["date", "value"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"])
    st.session_state.data_store[series_id] = df   # cache for charting

    meta = _fred_get("series", {"series_id": series_id}).get("seriess", [{}])[0]
    s = df["value"].describe()

    def snap(rows: pd.DataFrame) -> list:
        return (rows[["date", "value"]]
                .assign(date=rows["date"].dt.strftime("%Y-%m-%d"))
                .to_dict("records"))

    return json.dumps({
        "series_id": series_id,
        "title": meta.get("title", series_id),
        "units": meta.get("units", ""),
        "frequency": meta.get("frequency_short", ""),
        "n_observations": len(df),
        "date_range": f"{df['date'].min().date()} → {df['date'].max().date()}",
        "mean": round(s["mean"], 4),
        "std": round(s["std"], 4),
        "min": round(s["min"], 4),
        "max": round(s["max"], 4),
        "latest": {
            "date": df.iloc[-1]["date"].strftime("%Y-%m-%d"),
            "value": round(df.iloc[-1]["value"], 4),
        },
        "first_3": snap(df.head(3)),
        "last_3": snap(df.tail(3)),
    })


def _build_figure(series_ids: list[str], title: str) -> go.Figure:
    fig = go.Figure()
    for sid in series_ids:
        df = st.session_state.data_store[sid]
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["value"], name=sid, mode="lines",
            hovertemplate="%{x|%b %Y}<br>%{y:.3f}<extra></extra>",
        ))
    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", font=dict(size=16)),
        height=420, template="plotly_white", hovermode="x unified",
        legend=dict(orientation="h", y=1.05),
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def tool_create_chart(series_ids: list[str], title: str, pending_charts: list) -> str:
    missing = [sid for sid in series_ids if sid not in st.session_state.data_store]
    if missing:
        return json.dumps({"error": f"Not fetched yet: {missing}. Call fetch_fred_data first."})
    fig = _build_figure(series_ids, title)
    pending_charts.append(fig)
    return json.dumps({"success": True})


# ---------------------------------------------------------------------------
# Tool schema & system prompt
# ---------------------------------------------------------------------------

TOOLS: list[dict] = [
    {
        "name": "search_fred_series",
        "description": (
            "Search the FRED database for series matching a topic. "
            "Returns IDs, titles, frequency, units sorted by popularity."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "e.g. 'consumer price index', 'unemployment rate'"},
                "limit": {"type": "integer", "default": 6},
            },
            "required": ["query"],
        },
    },
    {
        "name": "fetch_fred_data",
        "description": (
            "Fetch observations for a FRED series between two dates. "
            "Returns summary stats. Caches data for charting. Use ISO dates YYYY-MM-DD."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "series_id": {"type": "string", "description": "FRED series ID e.g. 'FEDFUNDS'"},
                "start_date": {"type": "string"},
                "end_date": {"type": "string"},
            },
            "required": ["series_id", "start_date", "end_date"],
        },
    },
    {
        "name": "create_chart",
        "description": (
            "Create an interactive chart for one or more fetched FRED series. "
            "Displayed inline in the UI. Each series must be fetched first."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "series_ids": {"type": "array", "items": {"type": "string"}},
                "title": {"type": "string"},
            },
            "required": ["series_ids", "title"],
        },
    },
]

SYSTEM = f"""You are a financial data analyst with access to FRED (Federal Reserve Economic Data).
Today: {date.today().isoformat()}.

Common series IDs:
  FEDFUNDS (Fed rate monthly), DFF (daily), CPIAUCSL (CPI), CPILFESL (core CPI),
  UNRATE (unemployment), GDP (quarterly), T10Y2Y (yield curve), VIXCLS, SP500,
  MORTGAGE30US, PAYEMS (nonfarm payrolls), WALCL (Fed balance sheet).

For each question:
1. If you don't know the series ID, call search_fred_series.
2. Call fetch_fred_data with a date range matching the question's context.
3. Analyze data with economic context — mention Fed policy, recessions, crises.
4. Call create_chart when a chart helps understanding.
5. Give a concise, data-grounded narrative answer with key numbers highlighted."""


def _run_tool(name: str, inp: dict, pending_charts: list) -> str:
    try:
        if name == "search_fred_series":
            return tool_search_fred_series(**inp)
        if name == "fetch_fred_data":
            return tool_fetch_fred_data(**inp)
        if name == "create_chart":
            return tool_create_chart(inp["series_ids"], inp["title"], pending_charts)
        return json.dumps({"error": f"Unknown tool: {name}"})
    except Exception as exc:
        return json.dumps({"error": str(exc)})


# ---------------------------------------------------------------------------
# Agent loop (renders directly into current st.chat_message context)
# ---------------------------------------------------------------------------

def run_agent(client: anthropic.Anthropic, question: str) -> None:
    st.session_state.claude_history.append({"role": "user", "content": question})

    text_placeholder = st.empty()
    full_text = ""
    pending_charts: list = []

    while True:
        with client.messages.stream(
            model=MODEL,
            max_tokens=8192,
            thinking={"type": "adaptive"},
            system=SYSTEM,
            tools=TOOLS,
            messages=st.session_state.claude_history,
        ) as stream:
            for event in stream:
                if (event.type == "content_block_delta"
                        and hasattr(event, "delta")
                        and event.delta.type == "text_delta"):
                    full_text += event.delta.text
                    text_placeholder.markdown(full_text + "▌")
            response = stream.get_final_message()

        if full_text:
            text_placeholder.markdown(full_text)

        if response.stop_reason == "end_turn":
            assistant_text = next((b.text for b in response.content if b.type == "text"), "")
            st.session_state.claude_history.append({"role": "assistant", "content": assistant_text})
            break

        if response.stop_reason == "tool_use":
            st.session_state.claude_history.append({"role": "assistant", "content": response.content})
            tool_results = []

            for block in response.content:
                if block.type != "tool_use":
                    continue
                with st.status(f"⚡ {block.name}", expanded=False) as status:
                    result_str = _run_tool(block.name, block.input, pending_charts)
                    parsed = json.loads(result_str)
                    if "error" in parsed:
                        status.update(label=f"✗ {block.name}: {parsed['error']}", state="error")
                    else:
                        status.update(label=f"✓ {block.name}", state="complete")
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_str,
                })

            st.session_state.claude_history.append({"role": "user", "content": tool_results})
        else:
            break

    # Render charts inline and persist to display history
    for fig in pending_charts:
        st.plotly_chart(fig, use_container_width=True)

    st.session_state.display_history.append({
        "role": "assistant",
        "text": full_text,
        "charts": pending_charts,
    })


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("📈 FRED Q&A Agent")
    st.caption("Powered by Claude + Federal Reserve Economic Data")
    st.divider()

    # API key status
    fred_ok = bool(FRED_KEY)
    claude_ok = bool(CLAUDE_KEY)
    st.markdown("**API Keys**")
    st.markdown(f"{'🟢' if fred_ok else '🔴'} FRED API Key")
    st.markdown(f"{'🟢' if claude_ok else '🔴'} Anthropic API Key")

    if not fred_ok or not claude_ok:
        missing = [k for k, ok in [("FRED_API_KEY", fred_ok), ("ANTHROPIC_API_KEY", claude_ok)] if not ok]
        st.error(f"Missing: {', '.join(missing)}")

    st.divider()

    # Example questions
    st.markdown("**Try an example**")
    for q in EXAMPLE_QUESTIONS:
        if st.button(q, use_container_width=True, key=f"ex_{q[:20]}"):
            st.session_state["_queued"] = q

    st.divider()

    # Cache info + reset
    n_cached = len(st.session_state.data_store)
    if n_cached:
        st.caption(f"Cached series: {', '.join(st.session_state.data_store.keys())}")

    if st.button("🗑 Clear conversation", use_container_width=True):
        st.session_state.claude_history = []
        st.session_state.display_history = []
        st.session_state.data_store = {}
        st.rerun()


# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------

st.title("📈 FRED Financial Data Q&A")
st.caption("Ask questions about any economic indicator — the agent fetches data, analyzes it, and charts it.")

# Replay existing conversation
for msg in st.session_state.display_history:
    with st.chat_message(msg["role"]):
        if msg["text"]:
            st.markdown(msg["text"])
        for fig in msg.get("charts", []):
            st.plotly_chart(fig, use_container_width=True)

# Handle input (chat box or sidebar example button)
user_input = st.chat_input("Ask about any economic indicator...")
if not user_input:
    user_input = st.session_state.pop("_queued", None)

if user_input:
    if not fred_ok or not claude_ok:
        st.error("Please set FRED_API_KEY and ANTHROPIC_API_KEY before asking questions.")
        st.stop()

    # Display user message and persist
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.display_history.append({"role": "user", "text": user_input, "charts": []})

    # Run agent inside assistant bubble
    client = anthropic.Anthropic(api_key=CLAUDE_KEY)
    with st.chat_message("assistant"):
        try:
            run_agent(client, user_input)
        except anthropic.APIError as exc:
            st.error(f"Claude API error: {exc}")
        except requests.RequestException as exc:
            st.error(f"FRED request failed: {exc}")
