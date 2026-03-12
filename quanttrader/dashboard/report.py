"""
dashboard/report.py
────────────────────
HTML report generator using Plotly + Jinja2.

Produces a self-contained interactive HTML report for a BacktestResult,
containing:
  - Key metric cards
  - Equity curve + drawdown chart
  - Monthly returns heatmap
  - Trade P&L distribution
  - Parameter table
  - Rolling Sharpe chart
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Chart builders ────────────────────────────────────────────────────────────

def _equity_drawdown_chart(equity: pd.Series) -> str:
    dd = (equity / equity.cummax() - 1) * 100
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.65, 0.35],
        vertical_spacing=0.04,
        subplot_titles=("Equity Curve", "Drawdown (%)"),
    )
    fig.add_trace(go.Scatter(
        x=equity.index, y=equity.values,
        name="Equity", line=dict(color="#00C897", width=2),
        fill="tozeroy", fillcolor="rgba(0,200,151,0.08)",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=dd.index, y=dd.values,
        name="Drawdown", line=dict(color="#FF5E5B", width=1.5),
        fill="tozeroy", fillcolor="rgba(255,94,91,0.15)",
    ), row=2, col=1)
    fig.update_layout(
        height=550, template="plotly_dark",
        showlegend=False, margin=dict(l=60, r=20, t=40, b=20),
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def _monthly_heatmap(equity: pd.Series) -> str:
    monthly = equity.resample("ME").last().pct_change() * 100
    monthly.index = monthly.index.to_period("M")
    df = monthly.reset_index()
    df.columns = ["period", "ret"]
    df["year"]  = df["period"].dt.year
    df["month"] = df["period"].dt.month
    pivot = df.pivot(index="year", columns="month", values="ret").fillna(0)
    month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
    pivot.columns = [month_names[m-1] for m in pivot.columns]
    fig = go.Figure(go.Heatmap(
        z=pivot.values, x=list(pivot.columns), y=[str(y) for y in pivot.index],
        colorscale=[
            [0.0, "#FF5E5B"], [0.5, "#1a1a2e"], [1.0, "#00C897"],
        ],
        zmid=0,
        text=[[f"{v:.1f}%" for v in row] for row in pivot.values],
        texttemplate="%{text}",
        showscale=True,
    ))
    fig.update_layout(
        title="Monthly Returns (%)", height=300,
        template="plotly_dark", margin=dict(l=60, r=20, t=50, b=20),
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def _trade_pnl_chart(trades: List[Any]) -> str:
    if not trades:
        return "<p>No trades.</p>"
    pnls = [t.pnl if hasattr(t, "pnl") else t.get("pnl", 0) for t in trades]
    colors = ["#00C897" if p >= 0 else "#FF5E5B" for p in pnls]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=pnls, marker_color=colors, name="Trade P&L",
    ))
    fig.add_trace(go.Histogram(
        x=pnls, name="Distribution",
        marker_color="#7B61FF", opacity=0.7,
        xaxis="x2", yaxis="y2",
    ))
    fig.update_layout(
        title="Trade P&L",
        template="plotly_dark",
        height=350,
        xaxis2=dict(overlaying="x", side="top", visible=False),
        yaxis2=dict(overlaying="y", side="right", title="Count"),
        margin=dict(l=60, r=60, t=50, b=20),
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def _rolling_sharpe_chart(equity: pd.Series, window: int = 63) -> str:
    from backtest.metrics import rolling_sharpe
    rs = rolling_sharpe(equity, window).dropna()
    fig = go.Figure(go.Scatter(
        x=rs.index, y=rs.values, name="Rolling Sharpe",
        line=dict(color="#7B61FF", width=1.5),
        fill="tozeroy", fillcolor="rgba(123,97,255,0.1)",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="#888")
    fig.update_layout(
        title=f"Rolling Sharpe ({window}-bar)",
        template="plotly_dark", height=280,
        showlegend=False, margin=dict(l=60, r=20, t=40, b=20),
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


# ── Main generator ────────────────────────────────────────────────────────────

TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{{ strategy_name }} Report</title>
<script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
<style>
  :root { --bg:#0d0d1a; --card:#1a1a2e; --accent:#00C897; --warn:#FF5E5B;
          --purple:#7B61FF; --text:#e0e0e0; --muted:#888; }
  * { box-sizing:border-box; margin:0; padding:0; }
  body { background:var(--bg); color:var(--text); font-family:'Segoe UI',sans-serif;
         font-size:14px; padding:24px; }
  h1 { color:var(--accent); font-size:1.8rem; margin-bottom:4px; }
  .subtitle { color:var(--muted); margin-bottom:24px; }
  .metric-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(160px,1fr));
                 gap:12px; margin-bottom:28px; }
  .metric-card { background:var(--card); border-radius:8px; padding:16px;
                 border-left:3px solid var(--purple); }
  .metric-card.good { border-left-color:var(--accent); }
  .metric-card.bad  { border-left-color:var(--warn); }
  .metric-label { font-size:11px; color:var(--muted); text-transform:uppercase;
                  letter-spacing:.05em; margin-bottom:4px; }
  .metric-value { font-size:1.5rem; font-weight:700; }
  .chart-box { background:var(--card); border-radius:8px; padding:16px;
               margin-bottom:20px; }
  .params-table { width:100%; border-collapse:collapse; }
  .params-table th,td { padding:8px 12px; text-align:left;
                        border-bottom:1px solid #2a2a3e; }
  .params-table th { background:#2a2a3e; color:var(--muted); font-size:11px;
                     text-transform:uppercase; }
  .tag { display:inline-block; background:#2a2a3e; color:var(--purple);
         border-radius:4px; padding:2px 8px; font-size:11px; margin:2px; }
  footer { margin-top:40px; text-align:center; color:var(--muted); font-size:12px; }
</style>
</head>
<body>
<h1>{{ strategy_name }}</h1>
<div class="subtitle">{{ symbols }} · {{ timeframe }} · {{ start }} → {{ end }}</div>

<div class="metric-grid">
{% for m in metric_cards %}
<div class="metric-card {{ m.cls }}">
  <div class="metric-label">{{ m.label }}</div>
  <div class="metric-value" style="color:{{ m.color }}">{{ m.value }}</div>
</div>
{% endfor %}
</div>

<div class="chart-box">{{ equity_chart }}</div>
<div class="chart-box">{{ monthly_chart }}</div>
<div class="chart-box">{{ rolling_sharpe_chart }}</div>
<div class="chart-box">{{ trade_chart }}</div>

<div class="chart-box">
<h3 style="margin-bottom:12px;color:var(--muted)">Parameters</h3>
<table class="params-table">
<tr><th>Parameter</th><th>Value</th></tr>
{% for k,v in params.items() %}
<tr><td>{{ k }}</td><td>{{ v }}</td></tr>
{% endfor %}
</table>
</div>

<footer>Generated by QuantTrader · {{ generated_at }}</footer>
</body>
</html>
"""


def generate_html_report(
    result: Any,
    output_path: Optional[str] = None,
    open_browser: bool = False,
) -> str:
    """
    Generate an interactive HTML report for a BacktestResult.
    Returns the HTML string; optionally writes to *output_path*.
    """
    from jinja2 import Template
    from datetime import datetime

    m = result.metrics

    def pct(val) -> str:
        return f"{val:.2f}%" if val is not None else "—"
    def num(val, decimals=3) -> str:
        return f"{val:.{decimals}f}" if val is not None else "—"

    metric_cards = [
        {"label": "Total Return",  "value": pct(m.get("total_return_pct")),
         "cls": "good" if m.get("total_return_pct", 0) >= 0 else "bad",
         "color": "#00C897" if m.get("total_return_pct", 0) >= 0 else "#FF5E5B"},
        {"label": "CAGR",          "value": pct(m.get("cagr_pct")),
         "cls": "good" if m.get("cagr_pct", 0) >= 0 else "bad",
         "color": "#00C897" if m.get("cagr_pct", 0) >= 0 else "#FF5E5B"},
        {"label": "Sharpe",        "value": num(m.get("sharpe")),
         "cls": "good" if m.get("sharpe", 0) >= 1 else ("" if m.get("sharpe",0) >= 0 else "bad"),
         "color": "#00C897" if m.get("sharpe", 0) >= 1 else "#e0e0e0"},
        {"label": "Sortino",       "value": num(m.get("sortino")),
         "cls": "", "color": "#7B61FF"},
        {"label": "Calmar",        "value": num(m.get("calmar")),
         "cls": "", "color": "#7B61FF"},
        {"label": "Max Drawdown",  "value": pct(m.get("max_drawdown_pct")),
         "cls": "bad", "color": "#FF5E5B"},
        {"label": "Win Rate",      "value": pct(m.get("win_rate_pct")),
         "cls": "", "color": "#e0e0e0"},
        {"label": "Profit Factor", "value": num(m.get("profit_factor")),
         "cls": "good" if m.get("profit_factor", 0) >= 1.5 else "",
         "color": "#00C897" if m.get("profit_factor", 0) >= 1.5 else "#e0e0e0"},
        {"label": "# Trades",      "value": str(m.get("n_trades", 0)),
         "cls": "", "color": "#e0e0e0"},
        {"label": "VaR 95%",       "value": pct(m.get("var_95_pct")),
         "cls": "bad", "color": "#FF5E5B"},
    ]

    symbols = result.symbol if isinstance(result.symbol, str) else ", ".join(result.symbol)

    html = Template(TEMPLATE).render(
        strategy_name=result.strategy_name,
        symbols=symbols,
        timeframe=result.timeframe,
        start=result.start,
        end=result.end,
        metric_cards=metric_cards,
        equity_chart=_equity_drawdown_chart(result.equity),
        monthly_chart=_monthly_heatmap(result.equity),
        rolling_sharpe_chart=_rolling_sharpe_chart(result.equity),
        trade_chart=_trade_pnl_chart(result.trades),
        params=result.params,
        generated_at=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
    )

    if output_path:
        Path(output_path).write_text(html, encoding="utf-8")

    if open_browser:
        import webbrowser, tempfile
        tmp = tempfile.mktemp(suffix=".html")
        Path(tmp).write_text(html, encoding="utf-8")
        webbrowser.open(f"file://{tmp}")

    return html
