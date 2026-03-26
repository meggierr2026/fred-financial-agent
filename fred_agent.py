"""
fred_agent.py — Interactive financial data Q&A agent (FRED + Claude)

Usage:
    uv run --with anthropic --with requests --with pandas --with plotly --with rich \\
        --with python-dotenv python3 fred_agent.py
"""

import json
import os
import sys
import webbrowser
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
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

FRED_BASE = "https://api.stlouisfed.org/fred"
MODEL = "claude-opus-4-6"
CHART_FILE = "fred_chart.html"

console = Console()
_store: dict[str, pd.DataFrame] = {}   # series_id → DataFrame (in-memory cache)


# ---------------------------------------------------------------------------
# FRED helpers
# ---------------------------------------------------------------------------

def _fred_get(endpoint: str, params: dict) -> dict:
    full_params = {**params, "api_key": os.environ["FRED_API_KEY"], "file_type": "json"}
    r = requests.get(f"{FRED_BASE}/{endpoint}", params=full_params, timeout=15)
    r.raise_for_status()
    return r.json()


def tool_search_fred_series(query: str, limit: int = 6) -> str:
    data = _fred_get("series/search", {"search_text": query, "limit": limit})
    series = data.get("seriess", [])
    if not series:
        return json.dumps({"error": f"No series found for: {query}"})
    results = sorted(
        [
            {
                "id": s["id"],
                "title": s["title"],
                "frequency": s.get("frequency_short", ""),
                "units": s.get("units_short", ""),
                "popularity": s.get("popularity", 0),
            }
            for s in series
        ],
        key=lambda x: -x["popularity"],
    )
    return json.dumps({"series": results})


def tool_fetch_fred_data(series_id: str, start_date: str, end_date: str) -> str:
    try:
        raw = _fred_get(
            "series/observations",
            {"series_id": series_id, "observation_start": start_date, "observation_end": end_date},
        )
    except requests.HTTPError as exc:
        return json.dumps({"error": f"FRED HTTP error: {exc}"})

    obs = [o for o in raw.get("observations", []) if o["value"] != "."]
    if not obs:
        return json.dumps({"error": f"No data for {series_id} in {start_date}–{end_date}"})

    df = pd.DataFrame(obs)[["date", "value"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"])
    _store[series_id] = df  # cache for charting

    meta_raw = _fred_get("series", {"series_id": series_id}).get("seriess", [{}])
    meta = meta_raw[0] if meta_raw else {}

    s = df["value"].describe()
    snap = lambda rows: (
        rows[["date", "value"]]
        .assign(date=rows["date"].dt.strftime("%Y-%m-%d"))
        .to_dict("records")
    )

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
        "latest": {"date": df.iloc[-1]["date"].strftime("%Y-%m-%d"), "value": round(df.iloc[-1]["value"], 4)},
        "first_3": snap(df.head(3)),
        "last_3": snap(df.tail(3)),
    })


def tool_create_chart(series_ids: list, title: str) -> str:
    missing = [sid for sid in series_ids if sid not in _store]
    if missing:
        return json.dumps({"error": f"Not fetched yet: {missing}. Call fetch_fred_data first."})

    fig = go.Figure()
    for sid in series_ids:
        df = _store[sid]
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["value"],
            name=sid, mode="lines",
            hovertemplate="%{x|%b %Y}<br>%{y:.3f}<extra></extra>",
        ))

    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", font=dict(size=17)),
        height=520, template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.05),
    )

    out = os.path.abspath(CHART_FILE)
    fig.write_html(out, include_plotlyjs="cdn")
    webbrowser.open(f"file://{out}")
    return json.dumps({"success": True, "file": out})


# ---------------------------------------------------------------------------
# Tool schema & dispatch
# ---------------------------------------------------------------------------

TOOLS: list[dict] = [
    {
        "name": "search_fred_series",
        "description": (
            "Search the FRED database for series matching a topic. Returns series IDs, "
            "titles, frequency, and units sorted by popularity. Use this when you don't "
            "know the exact series ID."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search terms, e.g. 'CPI inflation', 'unemployment rate'"},
                "limit": {"type": "integer", "description": "Max results (default 6)", "default": 6},
            },
            "required": ["query"],
        },
    },
    {
        "name": "fetch_fred_data",
        "description": (
            "Fetch historical observations for a FRED series between two dates. "
            "Returns summary statistics. Also caches data internally for charting. "
            "Use ISO dates YYYY-MM-DD."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "series_id": {"type": "string", "description": "FRED series ID, e.g. 'FEDFUNDS', 'CPIAUCSL'"},
                "start_date": {"type": "string", "description": "Start date YYYY-MM-DD"},
                "end_date": {"type": "string", "description": "End date YYYY-MM-DD"},
            },
            "required": ["series_id", "start_date", "end_date"],
        },
    },
    {
        "name": "create_chart",
        "description": (
            "Create an interactive Plotly chart for one or more FRED series and open it "
            "in the browser. Each series must have been fetched with fetch_fred_data first."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "series_ids": {
                    "type": "array", "items": {"type": "string"},
                    "description": "FRED series IDs to plot",
                },
                "title": {"type": "string", "description": "Chart title"},
            },
            "required": ["series_ids", "title"],
        },
    },
]

_DISPATCH = {
    "search_fred_series": lambda i: tool_search_fred_series(**i),
    "fetch_fred_data": lambda i: tool_fetch_fred_data(**i),
    "create_chart": lambda i: tool_create_chart(**i),
}


def execute_tool(name: str, tool_input: dict) -> str:
    fn = _DISPATCH.get(name)
    if not fn:
        return json.dumps({"error": f"Unknown tool: {name}"})
    try:
        return fn(tool_input)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM = f"""You are a financial data analyst with access to FRED (Federal Reserve Economic Data).

Today: {date.today().isoformat()}

Commonly used series IDs:
  FEDFUNDS  Federal Funds Rate (monthly)      DFF      Fed Funds Rate (daily)
  CPIAUCSL  CPI All Items (inflation)         CPILFESL  Core CPI (ex food/energy)
  UNRATE    Unemployment Rate                 PAYEMS   Nonfarm Payrolls
  GDP       Real GDP (quarterly)             T10Y2Y   10Y-2Y Treasury Spread
  VIXCLS    VIX Volatility Index             SP500    S&P 500
  MORTGAGE30US 30-Year Mortgage Rate         WALCL    Fed Balance Sheet

Workflow for each question:
1. If you don't know the series ID, call search_fred_series.
2. Call fetch_fred_data with an appropriate date range for the question.
3. Analyze the statistics and contextualize them with economic events.
4. Call create_chart when a visualization would help the user understand the data.
5. Give a clear narrative answer grounded in the actual numbers.

Always interpret data in its economic context — mention relevant Fed decisions,
recessions (shading dates), crises, or policy shifts that explain the patterns."""


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

def run_agent(client: anthropic.Anthropic, question: str, history: list) -> None:
    """Run one full user turn: stream text, handle tool calls, loop until end_turn."""
    history.append({"role": "user", "content": question})

    while True:
        header_printed = False

        with client.messages.stream(
            model=MODEL,
            max_tokens=8192,
            thinking={"type": "adaptive"},
            system=SYSTEM,
            tools=TOOLS,
            messages=history,
        ) as stream:
            for event in stream:
                if event.type == "content_block_start":
                    if hasattr(event, "content_block") and event.content_block.type == "text":
                        if not header_printed:
                            console.print()
                            console.rule("[cyan]Assistant[/cyan]")
                            header_printed = True
                elif event.type == "content_block_delta":
                    if hasattr(event, "delta") and event.delta.type == "text_delta":
                        console.print(event.delta.text, end="", markup=False, highlight=False)

            response = stream.get_final_message()

        if header_printed:
            console.print("\n")

        if response.stop_reason == "end_turn":
            assistant_text = next(
                (b.text for b in response.content if b.type == "text"), ""
            )
            history.append({"role": "assistant", "content": assistant_text})
            break

        if response.stop_reason == "tool_use":
            history.append({"role": "assistant", "content": response.content})
            tool_results = []

            for block in response.content:
                if block.type != "tool_use":
                    continue
                arg_preview = json.dumps(block.input)[:70]
                console.print(f"  [dim cyan]⚡ {block.name}[/dim cyan] [dim]{arg_preview}[/dim]")
                result_str = execute_tool(block.name, block.input)
                parsed = json.loads(result_str)
                if "error" in parsed:
                    console.print(f"    [red]✗ {parsed['error']}[/red]")
                elif block.name == "create_chart":
                    console.print("    [green]✓ Chart opened in browser[/green]")
                else:
                    console.print(f"    [dim green]✓ done[/dim green]")
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_str,
                })

            history.append({"role": "user", "content": tool_results})
        else:
            break  # unexpected stop reason


# ---------------------------------------------------------------------------
# REPL
# ---------------------------------------------------------------------------

EXAMPLES = (
    "What was the Fed Funds Rate during the 2008 financial crisis?\n"
    "  How did inflation change after COVID?\n"
    "  Compare unemployment and GDP since 2010\n"
    "  Show me the yield curve inversion before the 2020 recession"
)


def main() -> None:
    for var in ("FRED_API_KEY", "ANTHROPIC_API_KEY"):
        if not os.environ.get(var):
            sys.exit(f"Error: {var} environment variable is not set.")

    client = anthropic.Anthropic()
    history: list = []

    console.print(Panel.fit(
        "[bold cyan]FRED Financial Data Q&A Agent[/bold cyan]\n"
        "[dim]Claude + Federal Reserve Economic Data[/dim]\n\n"
        f"[dim]Examples:[/dim]\n  {EXAMPLES}\n\n"
        "[dim]Commands: reset · quit[/dim]",
        border_style="cyan",
    ))

    while True:
        try:
            question = Prompt.ask("\n[bold green]You[/bold green]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Bye![/dim]")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            console.print("[dim]Bye![/dim]")
            break
        if question.lower() == "reset":
            history.clear()
            _store.clear()
            console.print("[dim]Conversation reset.[/dim]")
            continue

        try:
            run_agent(client, question, history)
        except anthropic.APIError as exc:
            console.print(f"[red]Claude API error: {exc}[/red]")
        except requests.RequestException as exc:
            console.print(f"[red]FRED request failed: {exc}[/red]")


if __name__ == "__main__":
    main()
