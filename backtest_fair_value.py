"""
Backtest: Fair Value REER Model
Expanding-window out-of-sample evaluation of REER = a + b * Real_Rate_Diff_vs_USD
Evaluates directional accuracy and simulated returns at 12-month horizon.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.getcwd()
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
PANEL_DATA_CSV = os.path.join(DATA_DIR, 'panel_data.csv')

OOS_START = '2010-01-01'
FORWARD_HORIZON = 12  # months
MIN_TRAIN_OBS = 200

# ============================================================================
# THEME (matches reer.py)
# ============================================================================
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

COLORS = {
    'primary': '#1a5f7a',
    'secondary': '#57837b',
    'accent': '#c38e70',
    'positive': '#2e7d32',
    'negative': '#c62828',
    'real_rate': '#1b5e20',
    'inflation': '#bf360c',
    'term_premium': '#0d47a1',
    'currency_risk': '#6a1b9a',
    'neutral': '#546e7a'
}

footnote = ('Backtest: Expanding-window pooled OLS | REER = a + b × Real Rate Diff vs USD | '
            f'{FORWARD_HORIZON}M forward horizon | OOS from {OOS_START[:7]}')

def add_footnote(fig, y_pos=0.02):
    fig.text(0.5, y_pos, footnote, ha='center', fontsize=8, style='italic', color='#666666')

# ============================================================================
# DATA LOADING
# ============================================================================
print("=" * 100)
print("FAIR VALUE REER BACKTEST — Expanding Window Out-of-Sample Evaluation")
print("=" * 100)

if not os.path.exists(PANEL_DATA_CSV):
    raise FileNotFoundError(f"Panel data not found: {PANEL_DATA_CSV}\nRun reer.py first to generate data.")

panel_df = pd.read_csv(PANEL_DATA_CSV, parse_dates=['Date'])
panel_df['real_rate'] = panel_df['bond10y'] - panel_df['cpi']

# US real rate for differentials
us_panel = panel_df[panel_df['Country'] == 'USA'][['Date', 'real_rate']].copy()
us_panel.columns = ['Date', 'us_real_rate']
panel_df = panel_df.merge(us_panel, on='Date', how='left')
panel_df['real_rate_diff_vs_usd'] = panel_df['real_rate'] - panel_df['us_real_rate']

# Drop USA and rows with missing key fields
backtest_df = panel_df[panel_df['Country'] != 'USA'].copy()
backtest_df = backtest_df.dropna(subset=['reer', 'real_rate_diff_vs_usd'])

# Pre-compute 12M forward REER change for each country-date
backtest_df = backtest_df.sort_values(['Country', 'Date'])
backtest_df['reer_fwd_12m'] = backtest_df.groupby('Country')['reer'].shift(-FORWARD_HORIZON)
backtest_df['reer_change_12m'] = (backtest_df['reer_fwd_12m'] - backtest_df['reer']) / backtest_df['reer'] * 100

dates = sorted(backtest_df['Date'].unique())
oos_dates = [d for d in dates if d >= pd.Timestamp(OOS_START)]
# Last usable signal date: must have 12M of forward data
max_signal_date = backtest_df.dropna(subset=['reer_change_12m'])['Date'].max()
oos_dates = [d for d in oos_dates if d <= max_signal_date]

print(f"Panel data: {len(panel_df):,} obs | {panel_df['Country'].nunique()} countries")
print(f"Date range: {panel_df['Date'].min().strftime('%Y-%m')} to {panel_df['Date'].max().strftime('%Y-%m')}")
print(f"OOS period: {pd.Timestamp(OOS_START).strftime('%Y-%m')} to {pd.Timestamp(max_signal_date).strftime('%Y-%m')} ({len(oos_dates)} months)")
print(f"Forward horizon: {FORWARD_HORIZON} months")
print("-" * 100)

# ============================================================================
# EXPANDING-WINDOW BACKTEST
# ============================================================================
results = []
coef_history = []

for t in oos_dates:
    # Train on all data up to and including t
    train = backtest_df[backtest_df['Date'] <= t].dropna(subset=['reer', 'real_rate_diff_vs_usd'])

    if len(train) < MIN_TRAIN_OBS:
        continue

    X_train = sm.add_constant(train['real_rate_diff_vs_usd'])
    y_train = train['reer']

    try:
        model = sm.OLS(y_train, X_train).fit()
    except Exception:
        continue

    coef_history.append({
        'Date': t,
        'intercept': model.params['const'],
        'beta': model.params['real_rate_diff_vs_usd'],
        'r_squared': model.rsquared,
        'n_obs': len(train),
    })

    # Score current-month signals for each country
    current = backtest_df[backtest_df['Date'] == t].copy()
    if current.empty:
        continue

    X_current = sm.add_constant(current['real_rate_diff_vs_usd'])
    current['fair_value'] = model.predict(X_current)
    current['deviation_pct'] = (current['reer'] - current['fair_value']) / current['fair_value'] * 100

    for _, row in current.iterrows():
        if np.isnan(row['reer_change_12m']):
            continue

        dev = row['deviation_pct']
        fwd = row['reer_change_12m']
        # Directional hit: overvalued (dev>0) should depreciate (fwd<0), and vice versa
        hit = 1 if (dev > 0 and fwd < 0) or (dev < 0 and fwd > 0) else 0
        # Simulated return: proportional to signal strength and outcome
        # Long undervalued (dev<0 -> expect appreciation -> gain if fwd>0)
        # Short overvalued (dev>0 -> expect depreciation -> gain if fwd<0)
        sim_return = -dev * fwd / 100  # scale down

        results.append({
            'Date': t,
            'Country': row['Country'],
            'reer': row['reer'],
            'fair_value': row['fair_value'],
            'deviation_pct': dev,
            'reer_change_12m': fwd,
            'hit': hit,
            'sim_return': sim_return,
        })

results_df = pd.DataFrame(results)
coef_df = pd.DataFrame(coef_history)

if results_df.empty:
    print("ERROR: No backtest results generated. Check data coverage.")
    exit(1)

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================
print("\n" + "=" * 100)
print("BACKTEST RESULTS SUMMARY")
print("=" * 100)

# Overall stats
overall_hits = results_df['hit'].mean() * 100
total_signals = len(results_df)
avg_sim = results_df['sim_return'].mean()

print(f"Total signals: {total_signals:,}")
print(f"Overall directional accuracy: {overall_hits:.1f}%")
print(f"Average simulated return per signal: {avg_sim:+.3f}")
print("-" * 100)

# By country
print(f"\n{'Country':<8} {'Signals':>8} {'Hit Rate':>10} {'Avg Dev%':>10} {'Avg Fwd%':>10} {'Avg SimRet':>12}")
print("-" * 60)
for country in sorted(results_df['Country'].unique()):
    c = results_df[results_df['Country'] == country]
    print(f"{country:<8} {len(c):>8} {c['hit'].mean()*100:>9.1f}% {c['deviation_pct'].mean():>+10.1f} "
          f"{c['reer_change_12m'].mean():>+10.1f} {c['sim_return'].mean():>+12.3f}")

print("=" * 100)

# By quintile of deviation
results_df['dev_quintile'] = pd.qcut(results_df['deviation_pct'], 5, labels=['Q1\n(Most Under)', 'Q2', 'Q3', 'Q4', 'Q5\n(Most Over)'])
print(f"\n{'Quintile':<16} {'Signals':>8} {'Hit Rate':>10} {'Avg Dev%':>10} {'Avg Fwd%':>10} {'Avg SimRet':>12}")
print("-" * 70)
for q in ['Q1\n(Most Under)', 'Q2', 'Q3', 'Q4', 'Q5\n(Most Over)']:
    qdf = results_df[results_df['dev_quintile'] == q]
    q_label = q.replace('\n', ' ')
    print(f"{q_label:<16} {len(qdf):>8} {qdf['hit'].mean()*100:>9.1f}% {qdf['deviation_pct'].mean():>+10.1f} "
          f"{qdf['reer_change_12m'].mean():>+10.1f} {qdf['sim_return'].mean():>+12.3f}")

print("=" * 100)

# Coefficient stability
print(f"\nCoefficient Stability (expanding window):")
print(f"  Intercept (a): {coef_df['intercept'].mean():.2f} ± {coef_df['intercept'].std():.2f} "
      f"[{coef_df['intercept'].min():.2f}, {coef_df['intercept'].max():.2f}]")
print(f"  Beta (b):      {coef_df['beta'].mean():.2f} ± {coef_df['beta'].std():.2f} "
      f"[{coef_df['beta'].min():.2f}, {coef_df['beta'].max():.2f}]")
print(f"  R-squared:     {coef_df['r_squared'].mean():.4f} ± {coef_df['r_squared'].std():.4f}")
print("=" * 100)

# ============================================================================
# VISUALIZATIONS
# ============================================================================

# Plot 1: Hit Rate by Country
fig1, ax1 = plt.subplots(figsize=(11, 6))
country_hits = results_df.groupby('Country')['hit'].mean().sort_values() * 100
bars = ax1.bar(country_hits.index, country_hits.values, color=COLORS['primary'], edgecolor='white', linewidth=1.5)
ax1.axhline(50, color=COLORS['negative'], linestyle='--', linewidth=2, alpha=0.8, label='Random (50%)')
ax1.axhline(overall_hits, color=COLORS['accent'], linestyle='--', linewidth=2, alpha=0.8,
            label=f'Overall ({overall_hits:.1f}%)')
ax1.set_ylabel('Directional Accuracy (%)')
ax1.set_title(f'Fair Value REER Backtest: Directional Accuracy by Country ({FORWARD_HORIZON}M Horizon)')
ax1.legend(loc='upper left', frameon=True)
for bar, val in zip(bars, country_hits.values):
    ax1.annotate(f'{val:.1f}%', (bar.get_x() + bar.get_width()/2, val + 1),
                 ha='center', fontsize=10, fontweight='bold', color='#333333')
add_footnote(fig1)
plt.tight_layout()
plt.subplots_adjust(bottom=0.15)
plt.show()

# Plot 2: Cumulative Simulated Return Over Time
fig2, ax2 = plt.subplots(figsize=(12, 6))
monthly_return = results_df.groupby('Date')['sim_return'].mean().sort_index()
cum_return = monthly_return.cumsum()
ax2.plot(cum_return.index, cum_return.values, color=COLORS['primary'], linewidth=2.5)
ax2.fill_between(cum_return.index, 0, cum_return.values,
                 where=cum_return.values >= 0, color=COLORS['positive'], alpha=0.15)
ax2.fill_between(cum_return.index, 0, cum_return.values,
                 where=cum_return.values < 0, color=COLORS['negative'], alpha=0.15)
ax2.axhline(0, color='#333333', linewidth=1)
ax2.set_ylabel('Cumulative Simulated Return')
ax2.set_title('Fair Value REER Backtest: Cumulative Signal Return (Long Undervalued / Short Overvalued)')
ax2.set_xlabel('Date')
add_footnote(fig2)
plt.tight_layout()
plt.subplots_adjust(bottom=0.15)
plt.show()

# Plot 3: Scatter — Deviation % vs 12M Forward REER Change
fig3, ax3 = plt.subplots(figsize=(11, 7))
ax3.scatter(results_df['deviation_pct'], results_df['reer_change_12m'],
            c=COLORS['primary'], alpha=0.25, s=20, edgecolors='none')

# Regression line
X_scatter = sm.add_constant(results_df['deviation_pct'])
scatter_model = sm.OLS(results_df['reer_change_12m'], X_scatter).fit()
x_range = np.linspace(results_df['deviation_pct'].min(), results_df['deviation_pct'].max(), 100)
y_pred = scatter_model.params['const'] + scatter_model.params['deviation_pct'] * x_range
ax3.plot(x_range, y_pred, color=COLORS['accent'], linewidth=2.5,
         label=f'Fitted (β={scatter_model.params["deviation_pct"]:.3f}, R²={scatter_model.rsquared:.3f})')

ax3.axhline(0, color='#333333', linewidth=0.8)
ax3.axvline(0, color='#333333', linewidth=0.8)
ax3.set_xlabel('Deviation from Fair Value (%)')
ax3.set_ylabel(f'{FORWARD_HORIZON}M Forward REER Change (%)')
ax3.set_title(f'Signal Quality: Valuation Deviation vs {FORWARD_HORIZON}M Forward REER Change')
ax3.legend(loc='upper right', frameon=True)

# Quadrant labels
xlim = ax3.get_xlim()
ylim = ax3.get_ylim()
ax3.text(xlim[0] * 0.7, ylim[1] * 0.85, 'Undervalued\n→ Appreciated ✓', fontsize=9,
         color=COLORS['positive'], fontweight='bold', ha='center')
ax3.text(xlim[1] * 0.7, ylim[0] * 0.85, 'Overvalued\n→ Depreciated ✓', fontsize=9,
         color=COLORS['positive'], fontweight='bold', ha='center')
ax3.text(xlim[1] * 0.7, ylim[1] * 0.85, 'Overvalued\n→ Appreciated ✗', fontsize=9,
         color=COLORS['negative'], fontweight='bold', ha='center')
ax3.text(xlim[0] * 0.7, ylim[0] * 0.85, 'Undervalued\n→ Depreciated ✗', fontsize=9,
         color=COLORS['negative'], fontweight='bold', ha='center')

add_footnote(fig3)
plt.tight_layout()
plt.subplots_adjust(bottom=0.12)
plt.show()

# Plot 4: Coefficient Stability Over Time
fig4, (ax4a, ax4b) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

ax4a.plot(coef_df['Date'], coef_df['intercept'], color=COLORS['primary'], linewidth=2, label='Intercept (a)')
ax4a.set_ylabel('Intercept (a)')
ax4a.set_title('Expanding-Window Coefficient Stability')
ax4a.legend(loc='upper right', frameon=True)
ax4a.axhline(coef_df['intercept'].mean(), color=COLORS['neutral'], linestyle='--', linewidth=1, alpha=0.7)

ax4b.plot(coef_df['Date'], coef_df['beta'], color=COLORS['accent'], linewidth=2, label='Beta (b)')
ax4b.set_ylabel('Beta (b) — Real Rate Diff')
ax4b.set_xlabel('Date')
ax4b.legend(loc='upper right', frameon=True)
ax4b.axhline(coef_df['beta'].mean(), color=COLORS['neutral'], linestyle='--', linewidth=1, alpha=0.7)

# Add R² on secondary axis
ax4c = ax4b.twinx()
ax4c.plot(coef_df['Date'], coef_df['r_squared'], color=COLORS['secondary'], linewidth=1.5,
          linestyle=':', alpha=0.7, label='R²')
ax4c.set_ylabel('R²', color=COLORS['secondary'])
ax4c.legend(loc='lower right', frameon=True)

add_footnote(fig4)
plt.tight_layout()
plt.subplots_adjust(bottom=0.1)
plt.show()

print("\nBacktest complete.")
