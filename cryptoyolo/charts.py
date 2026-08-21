"""Feature 1c: customizable price charts with bands.

Plotly candlesticks with band overlays, volume, and RSI. Everything is driven by
a BandConfig, so the notebook can toggle overlays without touching this file.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .config import CONFIG, TIMEFRAMES, BandConfig, Config
from . import indicators, prices

# Colorblind-safe, readable on both light and dark templates.
C_UP, C_DOWN = "#1a9850", "#d73027"
C_BAND, C_BAND_FILL = "#4575b4", "rgba(69,117,180,0.10)"
C_MID, C_KC, C_DC = "#762a83", "#e08214", "#5aae61"
MA_COLORS = ["#377eb8", "#ff7f00", "#984ea3", "#a65628"]


def price_chart(symbol: str, timeframe: str = "1w", cfg: Config = CONFIG,
                bands: BandConfig | None = None, df: pd.DataFrame | None = None,
                height: int = 720, template: str = "plotly_white") -> go.Figure:
    """Candlestick + bands (+ volume, RSI) for one asset and timeframe."""
    bands = bands or cfg.bands
    if df is None:
        raw, source = prices.get_ohlcv(symbol, timeframe, cfg)
    else:
        raw, source = df, "supplied"
    data = indicators.enrich(raw, bands)

    rows = 1 + int(bands.show_volume) + int(bands.show_rsi)
    heights = {1: [1.0], 2: [0.76, 0.24], 3: [0.64, 0.18, 0.18]}[rows]
    titles = ["", "Volume", "RSI (14)"][:rows]

    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True,
                        vertical_spacing=0.035, row_heights=heights,
                        subplot_titles=titles)

    fig.add_trace(go.Candlestick(
        x=data.index, open=data["open"], high=data["high"],
        low=data["low"], close=data["close"], name=symbol,
        increasing_line_color=C_UP, decreasing_line_color=C_DOWN,
    ), row=1, col=1)

    # -- Bollinger: widest band drawn as a filled envelope, inner as dashed lines
    if bands.bollinger:
        widest = max(bands.bollinger_stds)
        for k in sorted(bands.bollinger_stds, reverse=True):
            tag = str(k).replace(".", "_")
            hi, lo = f"bb_upper_{tag}", f"bb_lower_{tag}"
            if hi not in data:
                continue
            outer = k == widest
            fig.add_trace(go.Scatter(
                x=data.index, y=data[hi], name=f"BB +{k}σ",
                line=dict(color=C_BAND, width=1.2 if outer else 0.8,
                          dash="solid" if outer else "dot"),
                legendgroup=f"bb{k}",
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=data.index, y=data[lo], name=f"BB −{k}σ",
                line=dict(color=C_BAND, width=1.2 if outer else 0.8,
                          dash="solid" if outer else "dot"),
                fill="tonexty" if outer else None, fillcolor=C_BAND_FILL,
                legendgroup=f"bb{k}", showlegend=False,
            ), row=1, col=1)
        if "bb_mid" in data:
            fig.add_trace(go.Scatter(
                x=data.index, y=data["bb_mid"], name=f"BB mid ({bands.bollinger_window})",
                line=dict(color=C_MID, width=1.1),
            ), row=1, col=1)

    if bands.keltner and "kc_upper" in data:
        for col, nm in (("kc_upper", "Keltner upper"), ("kc_lower", "Keltner lower")):
            fig.add_trace(go.Scatter(x=data.index, y=data[col], name=nm,
                                     line=dict(color=C_KC, width=1, dash="dash")),
                          row=1, col=1)

    if bands.donchian and "dc_upper" in data:
        for col, nm in (("dc_upper", "Donchian high"), ("dc_lower", "Donchian low")):
            fig.add_trace(go.Scatter(x=data.index, y=data[col], name=nm,
                                     line=dict(color=C_DC, width=1, dash="dashdot")),
                          row=1, col=1)

    for i, w in enumerate(bands.moving_averages):
        col = f"sma_{w}"
        if col in data and data[col].notna().any():
            fig.add_trace(go.Scatter(
                x=data.index, y=data[col], name=f"SMA {w}",
                line=dict(color=MA_COLORS[i % len(MA_COLORS)], width=1.1),
            ), row=1, col=1)

    r = 2
    if bands.show_volume:
        colors = [C_UP if c >= o else C_DOWN
                  for c, o in zip(data["close"], data["open"])]
        fig.add_trace(go.Bar(x=data.index, y=data["volume"], name="Volume",
                             marker_color=colors, opacity=0.55, showlegend=False),
                      row=r, col=1)
        r += 1

    if bands.show_rsi and "rsi_14" in data:
        fig.add_trace(go.Scatter(x=data.index, y=data["rsi_14"], name="RSI(14)",
                                 line=dict(color="#666", width=1.2), showlegend=False),
                      row=r, col=1)
        for lvl, color in ((70, C_DOWN), (30, C_UP)):
            fig.add_hline(y=lvl, line=dict(color=color, width=0.8, dash="dot"),
                          row=r, col=1)
        fig.update_yaxes(range=[0, 100], row=r, col=1)

    last = data["close"].iloc[-1]
    first = data["close"].iloc[0]
    change = (last - first) / first * 100 if first else 0.0
    arrow = "▲" if change >= 0 else "▼"

    fig.update_layout(
        title=dict(text=f"<b>{symbol}/USD</b> — {TIMEFRAMES[timeframe]['label']} "
                        f"&nbsp; <span style='color:{C_UP if change>=0 else C_DOWN}'>"
                        f"{arrow} {change:+.2f}%</span>"
                        f"<br><sub>last {last:,.4f} · {len(data)} × "
                        f"{TIMEFRAMES[timeframe]['interval']} candles · source: {source}</sub>",
                   x=0.01, xanchor="left"),
        height=height, template=template, hovermode="x unified",
        xaxis_rangeslider_visible=False, margin=dict(l=60, r=30, t=90, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0, font=dict(size=10)),
    )
    fig.update_xaxes(rangebreaks=[])           # crypto trades 24/7; no gaps to hide
    return fig


def grid(symbols: list[str], timeframe: str = "1w", cfg: Config = CONFIG,
         cols: int = 2, height_per_row: int = 300,
         template: str = "plotly_white") -> go.Figure:
    """Compact multi-asset comparison: close line + widest Bollinger envelope."""
    bands = cfg.bands
    rows = (len(symbols) + cols - 1) // cols
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=symbols,
                        vertical_spacing=0.10, horizontal_spacing=0.07)

    tag = str(max(bands.bollinger_stds)).replace(".", "_")
    for i, sym in enumerate(symbols):
        r, c = i // cols + 1, i % cols + 1
        try:
            raw, _ = prices.get_ohlcv(sym, timeframe, cfg)
            data = indicators.enrich(raw, bands)
        except Exception as exc:  # noqa: BLE001 - annotate the cell and continue
            fig.add_annotation(text=f"{sym}: {exc}", row=r, col=c,
                               showarrow=False, font=dict(size=9, color=C_DOWN))
            continue

        change = (data["close"].iloc[-1] - data["close"].iloc[0]) / data["close"].iloc[0]
        line_color = C_UP if change >= 0 else C_DOWN
        if bands.bollinger and f"bb_upper_{tag}" in data:
            fig.add_trace(go.Scatter(x=data.index, y=data[f"bb_upper_{tag}"],
                                     line=dict(width=0), showlegend=False,
                                     hoverinfo="skip"), row=r, col=c)
            fig.add_trace(go.Scatter(x=data.index, y=data[f"bb_lower_{tag}"],
                                     line=dict(width=0), fill="tonexty",
                                     fillcolor=C_BAND_FILL, showlegend=False,
                                     hoverinfo="skip"), row=r, col=c)
        fig.add_trace(go.Scatter(x=data.index, y=data["close"], name=sym,
                                 line=dict(color=line_color, width=1.5),
                                 showlegend=False), row=r, col=c)
        fig.layout.annotations[i].text = f"{sym}  <span style='color:{line_color}'>{change*100:+.1f}%</span>"

    fig.update_layout(height=height_per_row * rows, template=template,
                      title=dict(text=f"<b>Universe — {TIMEFRAMES[timeframe]['label']}</b>",
                                 x=0.01, xanchor="left"),
                      margin=dict(l=50, r=25, t=70, b=35))
    fig.update_annotations(font_size=11)
    return fig


def score_chart(scores: pd.DataFrame, template: str = "plotly_white") -> go.Figure:
    """Stacked contribution of each feature family to the composite score."""
    df = scores.sort_values("composite")
    fig = go.Figure()
    for col, color, label in (
        ("w_technical", "#377eb8", "Technical"),
        ("w_social", "#ff7f00", "Social"),
        ("w_catalyst", "#4daf4a", "Catalyst"),
        ("w_positioning", "#984ea3", "Positioning"),
    ):
        if col in df:
            fig.add_trace(go.Bar(y=df["symbol"], x=df[col], name=label,
                                 orientation="h", marker_color=color))
    fig.update_layout(
        barmode="relative", height=max(340, 26 * len(df) + 140), template=template,
        title=dict(text="<b>Composite score by contribution</b>"
                        "<br><sub>weighted; bars right of zero are net bullish</sub>",
                   x=0.01, xanchor="left"),
        xaxis_title="weighted score contribution", margin=dict(l=70, r=30, t=80, b=45),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
    )
    fig.add_vline(x=0, line=dict(color="#888", width=1))
    return fig
