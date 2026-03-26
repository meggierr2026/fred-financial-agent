"""Interactive dashboard: Fed Funds Rate vs VIX correlation (last 10 years).

Usage:
    export FRED_API_KEY=<your_key>
    uv run --with pandas --with requests --with plotly python3 ffr_vix_dashboard.py
"""

import os
import sys
from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
import requests
from plotly.subplots import make_subplots

FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"
OUTPUT_FILE = "dashboard.html"


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def get_api_key() -> str:
    key = os.environ.get("FRED_API_KEY", "")
    if not key:
        sys.exit("Error: FRED_API_KEY environment variable is not set.")
    return key


def fetch_fred_series(series_id: str, api_key: str, start: str) -> pd.DataFrame:
    """Fetch a FRED series and return a DataFrame with date + value columns."""
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start,
    }
    try:
        response = requests.get(FRED_BASE_URL, params=params, timeout=15)
        response.raise_for_status()
    except requests.RequestException as exc:
        sys.exit(f"Failed to fetch {series_id}: {exc}")

    observations = response.json().get("observations", [])
    df = pd.DataFrame(observations)[["date", "value"]]
    df = df[df["value"] != "."].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"])
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_data(ffr_raw: pd.DataFrame, vix_raw: pd.DataFrame) -> pd.DataFrame:
    """Resample both series to month-start, merge, and return a clean DataFrame."""
    ffr = (
        ffr_raw.set_index("date")["value"]
        .resample("MS")
        .mean()
        .rename("ffr")
    )
    vix = (
        vix_raw.set_index("date")["value"]
        .resample("MS")
        .mean()
        .rename("vix")
    )
    merged = pd.merge(
        ffr.reset_index(),
        vix.reset_index(),
        on="date",
        how="inner",
    )
    if merged.empty:
        raise ValueError("No overlapping dates between FFR and VIX series.")
    return merged.sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def compute_correlations(df: pd.DataFrame) -> tuple[float, pd.Series]:
    overall_r = df["ffr"].corr(df["vix"])
    rolling = df["ffr"].rolling(12).corr(df["vix"])
    return overall_r, rolling


def compute_regression(ffr: pd.Series, vix: pd.Series) -> tuple[float, float, float]:
    """Return (slope, intercept, r_squared) via manual OLS (no numpy/scipy needed)."""
    slope = ffr.cov(vix) / ffr.var()
    intercept = vix.mean() - slope * ffr.mean()
    r_squared = ffr.corr(vix) ** 2
    return slope, intercept, r_squared


def compute_summary_stats(df: pd.DataFrame, overall_r: float) -> dict:
    ffr_stats = df["ffr"].describe()
    vix_stats = df["vix"].describe()
    return {
        "ffr_mean": round(ffr_stats["mean"], 2),
        "ffr_std": round(ffr_stats["std"], 2),
        "ffr_min": round(ffr_stats["min"], 2),
        "ffr_max": round(ffr_stats["max"], 2),
        "vix_mean": round(vix_stats["mean"], 2),
        "vix_std": round(vix_stats["std"], 2),
        "vix_min": round(vix_stats["min"], 2),
        "vix_max": round(vix_stats["max"], 2),
        "overall_r": round(overall_r, 4),
    }


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

def build_dashboard(
    df: pd.DataFrame,
    rolling_corr: pd.Series,
    stats: dict,
    regression: tuple[float, float, float],
) -> go.Figure:
    slope, intercept, r_squared = regression

    fig = make_subplots(
        rows=3,
        cols=2,
        specs=[
            [{"colspan": 2, "secondary_y": True}, None],
            [{"colspan": 2}, None],
            [{"type": "xy"}, {"type": "table"}],
        ],
        subplot_titles=[
            "Fed Funds Rate vs VIX (Monthly)",
            "12-Month Rolling Correlation (FFR × VIX)",
            "FFR vs VIX Scatter + OLS Regression",
            "Summary Statistics",
        ],
        vertical_spacing=0.10,
        row_heights=[0.38, 0.28, 0.34],
    )

    # --- Row 1: dual-axis time series ---
    fig.add_trace(
        go.Scatter(
            x=df["date"], y=df["ffr"],
            name="Fed Funds Rate (%)",
            line={"color": "#1f77b4", "width": 2},
            hovertemplate="%{x|%b %Y}<br>FFR: %{y:.2f}%<extra></extra>",
        ),
        row=1, col=1, secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=df["date"], y=df["vix"],
            name="VIX",
            line={"color": "#d62728", "width": 2},
            hovertemplate="%{x|%b %Y}<br>VIX: %{y:.2f}<extra></extra>",
        ),
        row=1, col=1, secondary_y=True,
    )

    # --- Row 2: rolling correlation ---
    fig.add_trace(
        go.Scatter(
            x=df["date"], y=rolling_corr,
            name="12-mo Rolling r",
            line={"color": "#9467bd", "width": 2},
            fill="tozeroy",
            fillcolor="rgba(148,103,189,0.15)",
            hovertemplate="%{x|%b %Y}<br>r = %{y:.3f}<extra></extra>",
        ),
        row=2, col=1,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1, row=2, col=1)
    fig.add_hline(
        y=stats["overall_r"],
        line_dash="dot",
        line_color="#9467bd",
        line_width=1,
        annotation_text=f"Overall r = {stats['overall_r']}",
        annotation_position="bottom right",
        row=2, col=1,
    )

    # --- Row 3 left: scatter + regression ---
    reg_x = pd.Series([df["ffr"].min(), df["ffr"].max()])
    reg_y = slope * reg_x + intercept

    fig.add_trace(
        go.Scatter(
            x=df["ffr"], y=df["vix"],
            mode="markers",
            name="Monthly obs",
            marker={"color": "#2ca02c", "size": 6, "opacity": 0.65},
            hovertemplate="FFR: %{x:.2f}%<br>VIX: %{y:.2f}<extra></extra>",
        ),
        row=3, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=reg_x, y=reg_y,
            mode="lines",
            name=f"OLS (r²={r_squared:.3f})",
            line={"color": "#ff7f0e", "width": 2, "dash": "dash"},
            hoverinfo="skip",
        ),
        row=3, col=1,
    )

    # --- Row 3 right: summary table ---
    fig.add_trace(
        go.Table(
            header=dict(
                values=["<b>Metric</b>", "<b>Fed Funds Rate (%)</b>", "<b>VIX</b>"],
                fill_color="#1f77b4",
                font=dict(color="white", size=12),
                align="left",
            ),
            cells=dict(
                values=[
                    ["Mean", "Std Dev", "Min", "Max", "Correlation (r)"],
                    [stats["ffr_mean"], stats["ffr_std"], stats["ffr_min"], stats["ffr_max"], stats["overall_r"]],
                    [stats["vix_mean"], stats["vix_std"], stats["vix_min"], stats["vix_max"], "—"],
                ],
                fill_color=[["#f0f0f0", "white"] * 3],
                align="left",
                font=dict(size=12),
            ),
        ),
        row=3, col=2,
    )

    # --- Layout ---
    fig.update_layout(
        title=dict(
            text="<b>Federal Funds Rate vs VIX — 10-Year Analysis</b>",
            font=dict(size=20),
        ),
        height=950,
        template="plotly_white",
        legend=dict(orientation="h", y=1.02, x=0),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Fed Funds Rate (%)", secondary_y=False, row=1, col=1)
    fig.update_yaxes(title_text="VIX", secondary_y=True, row=1, col=1)
    fig.update_yaxes(title_text="Pearson r", row=2, col=1)
    fig.update_xaxes(title_text="Fed Funds Rate (%)", row=3, col=1)
    fig.update_yaxes(title_text="VIX", row=3, col=1)

    return fig


def save_dashboard(fig: go.Figure, output_path: str) -> None:
    fig.write_html(output_path, include_plotlyjs="cdn")
    print(f"Dashboard saved → {os.path.abspath(output_path)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    api_key = get_api_key()
    start = (date.today() - timedelta(days=365 * 10)).isoformat()

    print("Fetching FEDFUNDS...")
    ffr_raw = fetch_fred_series("FEDFUNDS", api_key, start)
    print("Fetching VIXCLS...")
    vix_raw = fetch_fred_series("VIXCLS", api_key, start)

    df = prepare_data(ffr_raw, vix_raw)
    print(f"Merged dataset: {len(df)} monthly observations ({df['date'].min().date()} – {df['date'].max().date()})")

    overall_r, rolling_corr = compute_correlations(df)
    regression = compute_regression(df["ffr"], df["vix"])
    stats = compute_summary_stats(df, overall_r)

    print(f"Overall Pearson r (FFR × VIX): {overall_r:.4f}")
    print(f"OLS: VIX = {regression[0]:.3f} × FFR + {regression[1]:.3f}  (r² = {regression[2]:.3f})")

    fig = build_dashboard(df, rolling_corr, stats, regression)
    save_dashboard(fig, OUTPUT_FILE)


if __name__ == "__main__":
    main()
