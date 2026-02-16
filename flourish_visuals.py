"""
Flourish Visualization Builder — Bar Chart Race, Heatmap, Bump Chart

Generates Flourish-ready CSVs and local Plotly HTML previews for:
  1. Bar Chart Race — REER levels over time
  2. Heatmap — REER deviation from regime mean
  3. Bump Chart — Currency rankings over time

Usage:
    python flourish_visuals.py
    python flourish_visuals.py --visual bar_race
    python flourish_visuals.py --visual heatmap
    python flourish_visuals.py --visual bump
"""

import pandas as pd
import numpy as np
import ruptures as rpt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import argparse
import os

# ============================================================================
# CONFIGURATION
# ============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
ANIM_DIR = os.path.join(SCRIPT_DIR, 'animations')
REER_HIST_CSV = os.path.join(DATA_DIR, 'reer_historical.csv')

CURRENCIES = ['USA', 'ZAR', 'JPY', 'CNY', 'GBP', 'MXN', 'AUD', 'CAD', 'INR']

COLORS = {
    'USA': '#2196f3', 'ZAR': '#c62828', 'JPY': '#1565c0', 'CNY': '#e65100',
    'GBP': '#2e7d32', 'MXN': '#6a1b9a', 'AUD': '#00838f', 'CAD': '#ef6c00',
    'INR': '#4527a0',
}

# Structural break config (mirrored from reer.py)
BREAK_CONFIG = {
    'bai_perron_penalty': 20,
    'min_segment_months': 24,
    'min_regime_months': 12,
}
EM_BREAK_PENALTY = {'ZAR': 10, 'MXN': 15, 'INR': 15, 'CNY': 15}


# ============================================================================
# DATA LOADING
# ============================================================================
def load_reer():
    """Load and forward-fill REER historical data."""
    if not os.path.exists(REER_HIST_CSV):
        raise FileNotFoundError(
            f"REER data not found at {REER_HIST_CSV}. Run data_downloader.py first."
        )
    df = pd.read_csv(REER_HIST_CSV, index_col=0, parse_dates=True)
    df.index.name = 'Date'
    return df.ffill()


# ============================================================================
# STRUCTURAL BREAK DETECTION (self-contained, from reer.py)
# ============================================================================
def bai_perron_breaks(series, penalty=20, min_size=24):
    """Bai-Perron break detection via PELT algorithm."""
    if len(series) < min_size * 2:
        return []
    try:
        signal = series.values.reshape(-1, 1)
        model = rpt.Pelt(model="rbf", min_size=min_size).fit(signal)
        breaks = model.predict(pen=penalty)
        return [b for b in breaks if b < len(signal)]
    except Exception:
        return []


def get_regime_mean(series, country=None):
    """Compute regime mean for a REER series using Bai-Perron breaks."""
    clean = series.dropna()
    if len(clean) < 36:
        return clean.mean()

    penalty = EM_BREAK_PENALTY.get(country, BREAK_CONFIG['bai_perron_penalty'])
    breaks = bai_perron_breaks(clean, penalty=penalty,
                               min_size=BREAK_CONFIG['min_segment_months'])

    if not breaks:
        return clean.mean()

    last_break = max(breaks)
    regime_start = min(last_break + 1, len(clean) - 1)
    regime = clean.iloc[regime_start:]

    # Blend with previous regime if too short
    if len(regime) < BREAK_CONFIG['min_regime_months'] and len(breaks) > 1:
        prev_break = breaks[-2] if len(breaks) >= 2 else 0
        regime = clean.iloc[prev_break:]

    return regime.mean()


# ============================================================================
# VISUAL 1: BAR CHART RACE — REER Levels Over Time
# ============================================================================
def build_bar_chart_race(reer_df):
    """Export Flourish CSV + Plotly animated bar chart for REER levels."""
    print("\n--- Visual 1: Bar Chart Race (REER Levels) ---")

    # Flourish CSV (wide format, all 9 currencies)
    out = reer_df.copy().round(2)
    out.index = out.index.strftime('%Y-%m')
    out.index.name = 'Date'
    csv_path = os.path.join(DATA_DIR, 'flourish_bar_chart_race.csv')
    out.to_csv(csv_path)
    print(f"  CSV  → {csv_path}")

    # Plotly: animated bar chart (one frame per month)
    df = reer_df.dropna(how='all')
    dates = df.index

    # Build frames
    frames = []
    for date in dates:
        row = df.loc[date].dropna().sort_values(ascending=True)
        frames.append(go.Frame(
            data=[go.Bar(
                x=row.values, y=row.index, orientation='h',
                marker_color=[COLORS.get(c, '#888') for c in row.index],
                text=[f'{v:.1f}' for v in row.values],
                textposition='outside',
            )],
            name=date.strftime('%Y-%m'),
            layout=go.Layout(title_text=f'REER Levels — {date.strftime("%b %Y")}'),
        ))

    # Initial frame
    init = df.iloc[0].dropna().sort_values(ascending=True)
    fig = go.Figure(
        data=[go.Bar(
            x=init.values, y=init.index, orientation='h',
            marker_color=[COLORS.get(c, '#888') for c in init.index],
            text=[f'{v:.1f}' for v in init.values],
            textposition='outside',
        )],
        frames=frames,
        layout=go.Layout(
            title=dict(text=f'REER Levels — {dates[0].strftime("%b %Y")}', font=dict(size=18)),
            xaxis=dict(title='REER Index', range=[50, 160]),
            yaxis=dict(autorange=True),
            plot_bgcolor='#f8f9fa', paper_bgcolor='white',
            height=500, width=900,
            updatemenus=[dict(
                type='buttons', showactive=False, x=0.05, y=1.12,
                buttons=[
                    dict(label='▶ Play', method='animate',
                         args=[None, dict(frame=dict(duration=120, redraw=True),
                                          fromcurrent=True)]),
                    dict(label='⏸ Pause', method='animate',
                         args=[[None], dict(frame=dict(duration=0, redraw=False),
                                            mode='immediate')]),
                ],
            )],
            sliders=[dict(
                active=0, pad=dict(t=50),
                steps=[dict(args=[[f.name], dict(mode='immediate',
                            frame=dict(duration=0, redraw=True))],
                            label=f.name, method='animate')
                       for f in frames],
            )],
        ),
    )

    html_path = os.path.join(ANIM_DIR, 'reer_bar_chart_race.html')
    fig.write_html(html_path, include_plotlyjs=True)
    print(f"  HTML → {html_path}")


# ============================================================================
# VISUAL 2: HEATMAP — REER Deviation from Regime Mean
# ============================================================================
def build_heatmap(reer_df):
    """Export Flourish CSV + Plotly heatmap for regime-adjusted REER deviation."""
    print("\n--- Visual 2: Heatmap (REER Deviation from Regime Mean) ---")

    # Compute regime mean per currency
    regime_means = {}
    for ccy in reer_df.columns:
        rm = get_regime_mean(reer_df[ccy], country=ccy)
        regime_means[ccy] = rm
        print(f"  {ccy}: regime mean = {rm:.1f}")

    # Compute deviation %: (REER - regime_mean) / regime_mean × 100
    deviation = pd.DataFrame(index=reer_df.index)
    for ccy in reer_df.columns:
        deviation[ccy] = ((reer_df[ccy] - regime_means[ccy]) / regime_means[ccy] * 100).round(2)

    # Flourish CSV
    out = deviation.copy()
    out.index = out.index.strftime('%Y-%m')
    out.index.name = 'Date'
    csv_path = os.path.join(DATA_DIR, 'flourish_reer_heatmap.csv')
    out.to_csv(csv_path)
    print(f"  CSV  → {csv_path}")

    # Plotly heatmap
    z = deviation.T.values
    x_labels = [d.strftime('%Y-%m') for d in deviation.index]
    y_labels = list(deviation.columns)

    fig = go.Figure(data=go.Heatmap(
        z=z, x=x_labels, y=y_labels,
        colorscale=[
            [0.0, '#c62828'],   # deep red (overvalued)
            [0.35, '#ef9a9a'],  # light red
            [0.5, '#ffffff'],   # white (fair value)
            [0.65, '#a5d6a7'],  # light green
            [1.0, '#2e7d32'],   # deep green (undervalued)
        ],
        zmid=0,
        colorbar=dict(title='Deviation %', ticksuffix='%'),
        hovertemplate='%{y} | %{x}<br>Deviation: %{z:+.1f}%<extra></extra>',
    ))

    fig.update_layout(
        title=dict(text='REER Deviation from Regime Mean (%)', font=dict(size=18)),
        xaxis=dict(title='', dtick=12, tickangle=-45),
        yaxis=dict(title=''),
        plot_bgcolor='white', paper_bgcolor='white',
        height=450, width=1100,
    )

    html_path = os.path.join(ANIM_DIR, 'reer_deviation_heatmap.html')
    fig.write_html(html_path, include_plotlyjs=True)
    print(f"  HTML → {html_path}")


# ============================================================================
# VISUAL 3: BUMP CHART — Currency Rankings Over Time
# ============================================================================
def build_bump_chart(reer_df):
    """Export Flourish CSV + Plotly bump chart for monthly REER rankings."""
    print("\n--- Visual 3: Bump Chart (Currency Rankings) ---")

    # Compute monthly rank (1 = highest REER = strongest)
    ranks = reer_df.rank(axis=1, ascending=False, method='min').round(0).astype('Int64')

    # Flourish CSV
    out = ranks.copy()
    out.index = out.index.strftime('%Y-%m')
    out.index.name = 'Date'
    csv_path = os.path.join(DATA_DIR, 'flourish_bump_chart.csv')
    out.to_csv(csv_path)
    print(f"  CSV  → {csv_path}")

    # Plotly bump chart (lines with markers, inverted y-axis so rank 1 is on top)
    fig = go.Figure()

    for ccy in ranks.columns:
        series = ranks[ccy].dropna()
        if len(series) == 0:
            continue

        last_rank = series.iloc[-1]
        fig.add_trace(go.Scatter(
            x=series.index, y=series.values,
            mode='lines+markers',
            name=ccy,
            line=dict(color=COLORS.get(ccy, '#888'), width=2.5),
            marker=dict(size=4),
            hovertemplate=f'<b>{ccy}</b><br>%{{x|%b %Y}}<br>Rank: %{{y}}<extra></extra>',
        ))

        # End label
        fig.add_annotation(
            x=series.index[-1], y=last_rank,
            text=f' {ccy}',
            showarrow=False, xanchor='left',
            font=dict(size=11, color=COLORS.get(ccy, '#888'), family='Arial Black'),
        )

    fig.update_layout(
        title=dict(text='Currency Rankings by REER (1 = Strongest)', font=dict(size=18)),
        xaxis=dict(title='', showgrid=True, gridcolor='#e0e0e0'),
        yaxis=dict(
            title='Rank', autorange='reversed',
            dtick=1, showgrid=True, gridcolor='#e0e0e0',
        ),
        plot_bgcolor='#f8f9fa', paper_bgcolor='white',
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
        margin=dict(r=80),
        height=550, width=1100,
    )

    html_path = os.path.join(ANIM_DIR, 'reer_bump_chart.html')
    fig.write_html(html_path, include_plotlyjs=True)
    print(f"  HTML → {html_path}")


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='Flourish Visualization Builder')
    parser.add_argument('--visual', choices=['bar_race', 'heatmap', 'bump', 'all'],
                        default='all', help='Which visual to build (default: all)')
    args = parser.parse_args()

    print("=" * 60)
    print("Flourish Visualization Builder")
    print("=" * 60)

    reer_df = load_reer()
    print(f"  Data: {len(reer_df)} months, {len(reer_df.columns)} currencies")
    print(f"  Period: {reer_df.index[0].strftime('%Y-%m')} to {reer_df.index[-1].strftime('%Y-%m')}")

    os.makedirs(ANIM_DIR, exist_ok=True)

    if args.visual in ('bar_race', 'all'):
        build_bar_chart_race(reer_df)
    if args.visual in ('heatmap', 'all'):
        build_heatmap(reer_df)
    if args.visual in ('bump', 'all'):
        build_bump_chart(reer_df)

    print("\n" + "=" * 60)
    print("Done! Next steps:")
    print("  1. Upload CSVs from data/ to Flourish")
    print("  2. Open HTML files from animations/ in a browser")
    print("=" * 60)


if __name__ == '__main__':
    main()
