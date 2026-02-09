"""
Animated Structural Break Analysis Visualization
Progressive time-lapse through REER data with CUSUM + Bai-Perron break detection.

Outputs:
- GIF animation (matplotlib + Pillow)
- Interactive HTML (Plotly)
- Live matplotlib display

Usage:
    python structural_break_animation.py --country JPY --format gif
    python structural_break_animation.py --country all --format html
    python structural_break_animation.py --country ZAR --format live
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import statsmodels.api as sm
from statsmodels.stats.diagnostic import recursive_olsresiduals
import ruptures as rpt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import argparse
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
REER_HIST_CSV = os.path.join(DATA_DIR, 'reer_historical.csv')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'animations')

# Break detection config (same as reer.py)
BREAK_CONFIG = {
    'min_segment_months': 24,
    'bai_perron_penalty': 20,
    'cusum_skip': 10,
}

EM_BREAK_PENALTY = {
    'ZAR': 10, 'MXN': 15, 'INR': 15, 'CNY': 15,
}

# Animation settings
ANIMATION_CONFIG = {
    'fps': 15, # 
    'interval': 67,  # ms between frames
    'skip_frames': 2,  # Process every Nth data point for speed
}

# Professional color palette (matching reer.py)
COLORS = {
    'primary': '#1a5f7a',
    'secondary': '#57837b',
    'accent': '#c38e70',
    'positive': '#2e7d32',
    'negative': '#c62828',
    'cusum': '#0d47a1',
    'bounds': '#ff9800',
    'neutral': '#546e7a',
    'background': '#f8f9fa',
}

# Footnote and author (matching reer.py)
FOOTNOTE = ('Source: Federal Reserve Economic Data (FRED) | '
            'Structural breaks detected using Bai-Perron (PELT) algorithm\n'
            'Author: Mahlatse Ndala')

# Professional matplotlib style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Segoe UI', 'Arial', 'Helvetica'],
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'axes.labelsize': 12,
    'axes.labelweight': 'medium',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.facecolor': 'white',
    'axes.facecolor': '#f8f9fa',
    'grid.color': '#e0e0e0',
    'grid.linestyle': '-',
    'grid.alpha': 0.7,
    'legend.framealpha': 0.95,
    'legend.edgecolor': '#cccccc',
})


def add_footnote(fig, y_pos=0.02):
    """Add footnote to figure."""
    fig.text(0.5, y_pos, FOOTNOTE, ha='center', fontsize=8, style='italic', color='#666666')


# ============================================================================
# STRUCTURAL BREAK DETECTION (from reer.py)
# ============================================================================
def compute_cusum(series):
    """Compute CUSUM values for animation."""
    if len(series) < 30:
        return None, None, None
    
    X = sm.add_constant(np.arange(len(series)))
    try:
        model = sm.OLS(series.values, X).fit()
        skip = max(BREAK_CONFIG['cusum_skip'], int(len(series) * 0.1))
        rols_results = recursive_olsresiduals(model, skip=skip)
        cusum = rols_results[3]
        cusum_bounds = rols_results[4]
        return cusum, cusum_bounds, skip
    except:
        return None, None, None


def detect_breaks_at_point(series, country=None):
    """Detect breaks up to current point in series."""
    if len(series) < BREAK_CONFIG['min_segment_months'] * 2:
        return []
    
    penalty = EM_BREAK_PENALTY.get(country, BREAK_CONFIG['bai_perron_penalty'])
    
    try:
        signal = series.values.reshape(-1, 1)
        model = rpt.Pelt(model="rbf", min_size=BREAK_CONFIG['min_segment_months']).fit(signal)
        breaks = model.predict(pen=penalty)
        breaks = [b for b in breaks if b < len(signal)]
        return breaks
    except:
        return []


# ============================================================================
# MATPLOTLIB ANIMATION
# ============================================================================
def create_matplotlib_animation(country, reer_series, save_path=None, show_live=True):
    """Create progressive reveal animation using matplotlib."""
    
    # Prepare data
    dates = reer_series.index
    values = reer_series.values
    n_points = len(values)
    
    # Compute full CUSUM for bounds reference
    full_cusum, full_bounds, cusum_skip = compute_cusum(reer_series)
    
    # Pre-compute all breaks for the full series
    all_breaks = detect_breaks_at_point(reer_series, country)
    break_indices = set(all_breaks)
    
    # Setup figure with professional styling
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), height_ratios=[2, 1])
    fig.patch.set_facecolor('white')
    
    # REER plot setup
    ax1.set_facecolor(COLORS['background'])
    ax1.set_xlim(dates[0], dates[-1])
    ax1.set_ylim(values.min() * 0.95, values.max() * 1.05)
    ax1.set_ylabel('REER Index', fontsize=11, fontweight='medium')
    ax1.set_title(f'{country} Real Effective Exchange Rate - Structural Break Analysis', 
                  fontsize=14, fontweight='bold', pad=10)
    ax1.axhline(100, color=COLORS['neutral'], linestyle='--', alpha=0.6, linewidth=1.5, 
                label='Baseline (100)')
    
    # CUSUM plot setup
    ax2.set_facecolor(COLORS['background'])
    ax2.set_xlim(dates[0], dates[-1])
    if full_cusum is not None and len(full_cusum) > 0:
        cusum_min = np.nanmin(full_cusum)
        cusum_max = np.nanmax(full_cusum)
        if not np.isnan(cusum_min) and not np.isnan(cusum_max):
            cusum_range = max(abs(cusum_min), abs(cusum_max)) * 1.3
            if cusum_range > 0:
                ax2.set_ylim(-cusum_range, cusum_range)
            else:
                ax2.set_ylim(-5, 5)
        else:
            ax2.set_ylim(-5, 5)
    else:
        ax2.set_ylim(-5, 5)
    ax2.set_ylabel('CUSUM Statistic', fontsize=11, fontweight='medium')
    ax2.set_xlabel('Date', fontsize=11)
    ax2.axhline(0, color=COLORS['neutral'], linestyle='-', alpha=0.3, linewidth=1)
    ax2.set_title('CUSUM Stability Test', fontsize=12, fontweight='bold', pad=8)
    
    # Initialize plot elements with professional colors
    reer_line, = ax1.plot([], [], color=COLORS['primary'], linewidth=2.5, label='REER')
    mean_line, = ax1.plot([], [], color=COLORS['positive'], linewidth=2, 
                          linestyle='--', label='Regime Mean')
    cusum_line, = ax2.plot([], [], color=COLORS['cusum'], linewidth=2, label='CUSUM')
    upper_bound, = ax2.plot([], [], color=COLORS['bounds'], linewidth=1.5, 
                            linestyle='--', alpha=0.7, label='Critical Bounds (5%)')
    lower_bound, = ax2.plot([], [], color=COLORS['bounds'], linewidth=1.5, 
                            linestyle='--', alpha=0.7)
    
    # Storage for break lines and annotations
    break_lines = []
    break_annotations = []
    detected_breaks = set()
    
    # Stats text
    stats_text = ax1.text(0.02, 0.98, '', transform=ax1.transAxes, fontsize=10,
                          verticalalignment='top', fontfamily='monospace',
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    ax1.legend(loc='upper right', fontsize=9)
    ax2.legend(loc='upper right', fontsize=9)
    
    def init():
        reer_line.set_data([], [])
        mean_line.set_data([], [])
        cusum_line.set_data([], [])
        upper_bound.set_data([], [])
        lower_bound.set_data([], [])
        stats_text.set_text('')
        return reer_line, mean_line, cusum_line, upper_bound, lower_bound, stats_text
    
    def animate(frame):
        nonlocal detected_breaks
        
        # Calculate current index (skip frames for speed)
        idx = min(frame * ANIMATION_CONFIG['skip_frames'] + cusum_skip + 10, n_points - 1)
        
        # Current data slice
        current_dates = dates[:idx+1]
        current_values = values[:idx+1]
        current_series = reer_series.iloc[:idx+1]
        
        # Update REER line
        reer_line.set_data(current_dates, current_values)
        
        # Calculate and update regime mean
        current_breaks = detect_breaks_at_point(current_series, country)
        if current_breaks:
            last_break = max(current_breaks)
            regime_mean = current_values[last_break:].mean()
        else:
            regime_mean = current_values.mean()
        mean_line.set_data([current_dates[0], current_dates[-1]], [regime_mean, regime_mean])
        
        # Update CUSUM
        cusum, bounds, skip = compute_cusum(current_series)
        if cusum is not None and len(cusum) > 0:
            cusum_dates = current_dates[skip:skip+len(cusum)]
            if len(cusum_dates) == len(cusum):
                cusum_line.set_data(cusum_dates, cusum)
            
            if bounds is not None and len(bounds) > 0:
                if bounds.ndim > 1:
                    upper_vals = bounds[:, 0]
                else:
                    upper_vals = bounds
                # Ensure matching lengths
                min_len = min(len(cusum_dates), len(upper_vals))
                if min_len > 0:
                    upper_bound.set_data(cusum_dates[:min_len], upper_vals[:min_len])
                    lower_bound.set_data(cusum_dates[:min_len], -upper_vals[:min_len])
        
        # Check for new breaks and add vertical lines
        for brk in current_breaks:
            if brk not in detected_breaks and brk < len(current_dates):
                detected_breaks.add(brk)
                break_date = current_dates[brk]
                
                # Add vertical line to both plots
                vline1 = ax1.axvline(break_date, color=COLORS['negative'], 
                                     linewidth=2.5, alpha=0.8, zorder=5)
                vline2 = ax2.axvline(break_date, color=COLORS['negative'], 
                                     linewidth=2.5, alpha=0.8, zorder=5)
                break_lines.extend([vline1, vline2])
                
                # Add annotation
                ann = ax1.annotate(f'Break\n{break_date.strftime("%Y-%m")}',
                                   xy=(break_date, current_values[brk]),
                                   xytext=(10, 20), textcoords='offset points',
                                   fontsize=9, color=COLORS['negative'], fontweight='bold',
                                   arrowprops=dict(arrowstyle='->', color=COLORS['negative']))
                break_annotations.append(ann)
        
        # Update stats
        full_mean = current_values.mean()
        gap_pct = (regime_mean - current_values[-1]) / current_values[-1] * 100
        stats = (f"Date: {current_dates[-1].strftime('%Y-%m')}\n"
                 f"REER: {current_values[-1]:.1f}\n"
                 f"Regime Mean: {regime_mean:.1f}\n"
                 f"Full Mean: {full_mean:.1f}\n"
                 f"Breaks: {len(detected_breaks)}\n"
                 f"Gap: {gap_pct:+.1f}%")
        stats_text.set_text(stats)
        
        return (reer_line, mean_line, cusum_line, upper_bound, lower_bound, 
                stats_text, *break_lines, *break_annotations)
    
    # Calculate number of frames
    n_frames = (n_points - cusum_skip - 10) // ANIMATION_CONFIG['skip_frames']
    
    anim = animation.FuncAnimation(
        fig, animate, init_func=init, frames=n_frames,
        interval=ANIMATION_CONFIG['interval'], blit=False, repeat=False
    )
    
    # Add footnote
    add_footnote(fig, y_pos=0.01)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.10, top=0.93, hspace=0.25)
    
    if save_path:
        # Save as GIF
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        print(f"Saving animation to {save_path}...")
        writer = animation.PillowWriter(fps=ANIMATION_CONFIG['fps'])
        anim.save(save_path, writer=writer)
        print(f"Saved: {save_path}")
    
    if show_live:
        plt.show()
    else:
        plt.close()
    
    return anim


# ============================================================================
# PLOTLY INTERACTIVE ANIMATION
# ============================================================================
def create_plotly_animation(country, reer_series, save_path=None):
    """Create interactive HTML animation using Plotly."""
    
    dates = reer_series.index
    values = reer_series.values
    n_points = len(values)
    
    # Pre-compute breaks
    all_breaks = detect_breaks_at_point(reer_series, country)
    
    # Pre-compute CUSUM
    full_cusum, full_bounds, cusum_skip = compute_cusum(reer_series)
    
    # Create figure with subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(f'{country} Real Effective Exchange Rate - Structural Breaks', 
                       'CUSUM Stability Test'),
        vertical_spacing=0.15,
        row_heights=[0.65, 0.35]
    )
    
    # Create frames for animation
    frames = []
    skip = ANIMATION_CONFIG['skip_frames']
    
    for i in range(cusum_skip + 20, n_points, skip):
        current_dates = dates[:i+1]
        current_values = values[:i+1]
        current_series = reer_series.iloc[:i+1]
        
        # Calculate regime mean
        current_breaks = detect_breaks_at_point(current_series, country)
        if current_breaks:
            last_break = max(current_breaks)
            regime_mean = current_values[last_break:].mean()
        else:
            regime_mean = current_values.mean()
        
        # CUSUM for current slice
        cusum, bounds, cskip = compute_cusum(current_series)
        
        frame_data = [
            # REER line
            go.Scatter(x=current_dates, y=current_values, mode='lines',
                      line=dict(color=COLORS['primary'], width=2.5), name='REER'),
            # Regime mean
            go.Scatter(x=[current_dates[0], current_dates[-1]], 
                      y=[regime_mean, regime_mean],
                      mode='lines', line=dict(color=COLORS['positive'], width=2, dash='dash'),
                      name='Regime Mean'),
        ]
        
        # Add break lines
        for brk in current_breaks:
            if brk < len(current_dates):
                frame_data.append(
                    go.Scatter(x=[current_dates[brk], current_dates[brk]],
                              y=[values.min() * 0.95, values.max() * 1.05],
                              mode='lines', line=dict(color=COLORS['negative'], width=2.5),
                              name='Break', showlegend=False)
                )
        
        # CUSUM
        if cusum is not None and len(cusum) > 0:
            cusum_dates = current_dates[cskip:cskip+len(cusum)]
            frame_data.append(
                go.Scatter(x=cusum_dates, y=cusum, mode='lines',
                          line=dict(color=COLORS['cusum'], width=2), name='CUSUM',
                          xaxis='x2', yaxis='y2')
            )
            
            if bounds is not None and len(bounds) > 0:
                if bounds.ndim > 1:
                    upper_vals = bounds[:, 0]
                else:
                    upper_vals = bounds
                frame_data.extend([
                    go.Scatter(x=cusum_dates[:len(upper_vals)], y=upper_vals,
                              mode='lines', line=dict(color=COLORS['bounds'], width=1.5, dash='dash'),
                              name='Upper Bound', xaxis='x2', yaxis='y2'),
                    go.Scatter(x=cusum_dates[:len(upper_vals)], y=-upper_vals,
                              mode='lines', line=dict(color=COLORS['bounds'], width=1.5, dash='dash'),
                              name='Lower Bound', xaxis='x2', yaxis='y2'),
                ])
        
        frames.append(go.Frame(data=frame_data, name=str(i)))
    
    # Initial data (first frame)
    initial_idx = cusum_skip + 20
    initial_dates = dates[:initial_idx]
    initial_values = values[:initial_idx]
    
    # Add traces for initial state
    fig.add_trace(go.Scatter(x=initial_dates, y=initial_values, mode='lines',
                            line=dict(color=COLORS['primary'], width=2.5), name='REER'), row=1, col=1)
    fig.add_trace(go.Scatter(x=[initial_dates[0], initial_dates[-1]], 
                            y=[initial_values.mean(), initial_values.mean()],
                            mode='lines', line=dict(color=COLORS['positive'], width=2, dash='dash'),
                            name='Regime Mean'), row=1, col=1)
    
    # Empty CUSUM trace
    fig.add_trace(go.Scatter(x=[], y=[], mode='lines',
                            line=dict(color=COLORS['cusum'], width=2), name='CUSUM'), row=2, col=1)
    
    fig.frames = frames
    
    # Animation controls
    fig.update_layout(
        updatemenus=[
            dict(
                type='buttons',
                showactive=False,
                y=1.15,
                x=0.5,
                xanchor='center',
                buttons=[
                    dict(label='▶ Play',
                         method='animate',
                         args=[None, dict(frame=dict(duration=ANIMATION_CONFIG['interval'], 
                                                     redraw=True),
                                         fromcurrent=True,
                                         transition=dict(duration=0))]),
                    dict(label='⏸ Pause',
                         method='animate',
                         args=[[None], dict(frame=dict(duration=0, redraw=False),
                                           mode='immediate',
                                           transition=dict(duration=0))])
                ]
            )
        ],
        sliders=[dict(
            active=0,
            steps=[dict(method='animate',
                       args=[[f.name], dict(mode='immediate',
                                           frame=dict(duration=ANIMATION_CONFIG['interval'], redraw=True),
                                           transition=dict(duration=0))],
                       label=dates[int(f.name)].strftime('%Y-%m') if int(f.name) < len(dates) else '')
                   for f in frames[::5]],  # Show every 5th frame in slider
            x=0.1, len=0.8,
            xanchor='left',
            y=-0.05,
            currentvalue=dict(prefix='Date: ', visible=True, xanchor='center'),
            transition=dict(duration=0)
        )]
    )
    
    # Layout styling with professional theme and footnote
    fig.update_layout(
        title=dict(text=f'{country} Real Effective Exchange Rate - Structural Break Analysis',
                  font=dict(size=16, family='Segoe UI, Arial, sans-serif')),
        height=750,
        template='plotly_white',
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        # Add footnote as annotation
        annotations=[
            dict(
                text=('Source: Federal Reserve Economic Data (FRED) | '
                      'Structural breaks detected using Bai-Perron (PELT) algorithm<br>'
                      '<b>Author: Mahlatse Ndala</b>'),
                xref='paper', yref='paper',
                x=0.5, y=-0.12,
                showarrow=False,
                font=dict(size=10, color='#666666', family='Segoe UI, Arial, sans-serif'),
                align='center'
            )
        ],
        margin=dict(b=100)
    )
    
    fig.update_xaxes(title_text='Date', row=2, col=1)
    fig.update_yaxes(title_text='REER Index', row=1, col=1)
    fig.update_yaxes(title_text='CUSUM Statistic', row=2, col=1)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.write_html(save_path)
        print(f"Saved: {save_path}")
    
    return fig


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='Animated Structural Break Analysis')
    parser.add_argument('--country', type=str, default='JPY',
                       help='Country code (e.g., JPY, ZAR, USA) or "all"')
    parser.add_argument('--format', type=str, default='live',
                       choices=['gif', 'html', 'live', 'all'],
                       help='Output format')
    args = parser.parse_args()
    
    # Load data
    if not os.path.exists(REER_HIST_CSV):
        print(f"Error: {REER_HIST_CSV} not found. Run data_downloader.py first.")
        return
    
    reer_hist = pd.read_csv(REER_HIST_CSV, index_col=0, parse_dates=True)
    
    # Determine countries to process
    if args.country.lower() == 'all':
        countries = list(reer_hist.columns)
    else:
        countries = [args.country.upper()]
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for country in countries:
        if country not in reer_hist.columns:
            print(f"Warning: {country} not found in data, skipping.")
            continue
        
        print(f"\nProcessing {country}...")
        series = reer_hist[country].dropna()
        
        if len(series) < 50:
            print(f"  Insufficient data for {country}, skipping.")
            continue
        
        if args.format in ['gif', 'all']:
            gif_path = os.path.join(OUTPUT_DIR, f'{country}_structural_breaks.gif')
            create_matplotlib_animation(country, series, save_path=gif_path, show_live=False)
        
        if args.format in ['html', 'all']:
            html_path = os.path.join(OUTPUT_DIR, f'{country}_structural_breaks.html')
            fig = create_plotly_animation(country, series, save_path=html_path)
        
        if args.format == 'live':
            create_matplotlib_animation(country, series, save_path=None, show_live=True)
    
    print("\nDone!")


if __name__ == '__main__':
    main()
