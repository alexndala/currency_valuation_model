"""
REER Differentials vs USD (%) — Flourish Line Chart Race + Plotly Preview

Computes percentage differential: (REER(ccy) - REER(USD)) / REER(USD) × 100
for all non-USD currencies and outputs:
  1. Flourish-ready CSV (wide format) → data/reer_differentials_pct_flourish.csv
  2. Interactive Plotly HTML chart   → animations/reer_differentials_pct_vs_usd.html

Usage:
    python reer_differentials_pct_chart.py
"""

import pandas as pd
import plotly.graph_objects as go
import os

# ============================================================================
# CONFIGURATION
# ============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
REER_HIST_CSV = os.path.join(DATA_DIR, 'reer_historical.csv')
OUTPUT_CSV = os.path.join(DATA_DIR, 'reer_differentials_pct_flourish.csv')
OUTPUT_HTML = os.path.join(SCRIPT_DIR, 'animations', 'reer_differentials_pct_vs_usd.html')

CURRENCIES = ['ZAR', 'JPY', 'CNY', 'GBP', 'MXN', 'AUD', 'CAD', 'INR']

COLORS = {
    'ZAR': '#c62828', 'JPY': '#1565c0', 'CNY': '#e65100',
    'GBP': '#2e7d32', 'MXN': '#6a1b9a', 'AUD': '#00838f',
    'CAD': '#ef6c00', 'INR': '#4527a0',
}


# ============================================================================
# DATA LOADING & COMPUTATION
# ============================================================================
def load_and_compute_differentials_pct():
    """Load REER historical data and compute % differentials vs USD."""
    if not os.path.exists(REER_HIST_CSV):
        raise FileNotFoundError(
            f"REER historical data not found at {REER_HIST_CSV}. "
            "Run data_downloader.py first."
        )

    df = pd.read_csv(REER_HIST_CSV, index_col=0, parse_dates=True)
    df.index.name = 'Date'

    df = df.ffill()
    df = df.dropna(subset=['USA'])

    # Compute % differentials: (REER(ccy) - REER(USD)) / REER(USD) × 100
    differentials = pd.DataFrame(index=df.index)
    for ccy in CURRENCIES:
        if ccy in df.columns:
            differentials[ccy] = ((df[ccy] - df['USA']) / df['USA']) * 100

    differentials = differentials.dropna(how='all')
    return differentials


# ============================================================================
# FLOURISH CSV EXPORT
# ============================================================================
def export_flourish_csv(differentials):
    """Export wide-format CSV for Flourish Line Chart Race template."""
    out = differentials.copy()
    out.index = out.index.strftime('%Y-%m')
    out.index.name = 'Date'
    out = out.round(2)
    out.to_csv(OUTPUT_CSV)
    print(f"  Flourish CSV saved → {OUTPUT_CSV}")


# ============================================================================
# PLOTLY INTERACTIVE HTML
# ============================================================================
def build_plotly_chart(differentials):
    """Build an interactive Plotly line chart with end-of-line ranking labels."""
    fig = go.Figure()

    last_vals = {}
    for ccy in CURRENCIES:
        if ccy in differentials.columns:
            series = differentials[ccy].dropna()
            if len(series) > 0:
                last_vals[ccy] = series.iloc[-1]

    ranked = sorted(last_vals.items(), key=lambda x: x[1], reverse=True)

    for ccy in CURRENCIES:
        if ccy not in differentials.columns:
            continue
        series = differentials[ccy].dropna()
        if len(series) == 0:
            continue

        fig.add_trace(go.Scatter(
            x=series.index,
            y=series.values,
            mode='lines',
            name=ccy,
            line=dict(color=COLORS.get(ccy, '#888'), width=2.5),
            hovertemplate=(
                f'<b>{ccy}</b><br>'
                'Date: %{x|%b %Y}<br>'
                'Differential: %{y:+.1f}%<extra></extra>'
            ),
        ))

    for rank, (ccy, val) in enumerate(ranked, 1):
        series = differentials[ccy].dropna()
        fig.add_annotation(
            x=series.index[-1],
            y=val,
            text=f' {rank}. {ccy} ({val:+.1f}%)',
            showarrow=False,
            xanchor='left',
            font=dict(size=11, color=COLORS.get(ccy, '#888'), family='Arial Black'),
        )

    fig.update_layout(
        title=dict(
            text='REER Differentials vs USD (%) — (REER<sub>ccy</sub> − REER<sub>USD</sub>) / REER<sub>USD</sub>',
            font=dict(size=18),
        ),
        xaxis=dict(title='', showgrid=True, gridcolor='#e0e0e0'),
        yaxis=dict(title='REER Differential (%)', showgrid=True, gridcolor='#e0e0e0',
                   ticksuffix='%'),
        plot_bgcolor='#f8f9fa',
        paper_bgcolor='white',
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
        margin=dict(r=140),
        width=1100,
        height=600,
    )

    fig.add_hline(y=0, line_dash='dash', line_color='#999', line_width=1,
                  annotation_text='USD baseline', annotation_position='bottom left')

    os.makedirs(os.path.dirname(OUTPUT_HTML), exist_ok=True)
    fig.write_html(OUTPUT_HTML, include_plotlyjs=True)
    print(f"  Plotly HTML saved  → {OUTPUT_HTML}")


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("=" * 60)
    print("REER Differentials vs USD (%) — Chart Builder")
    print("=" * 60)

    differentials = load_and_compute_differentials_pct()
    print(f"\n  Data: {len(differentials)} months, {len(differentials.columns)} currencies")
    print(f"  Period: {differentials.index[0].strftime('%Y-%m')} to {differentials.index[-1].strftime('%Y-%m')}\n")

    export_flourish_csv(differentials)
    build_plotly_chart(differentials)

    print("\nDone! Next steps:")
    print("  1. Upload reer_differentials_pct_flourish.csv to Flourish (Line Chart Race template)")
    print("  2. Open reer_differentials_pct_vs_usd.html in a browser for local preview")
    print("=" * 60)


if __name__ == '__main__':
    main()
