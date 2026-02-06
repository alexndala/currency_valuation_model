import pandas as pd
import numpy as np
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import squarify
import statsmodels.api as sm
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PROFESSIONAL THEME SETUP
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

# Professional color palette
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

# ============================================================================
# DATA CONFIGURATION
# ============================================================================
# Multi-tenor bond yields for term structure analysis
data_map = {
    'USA': {
        'reer': 'RBUSBIS', 'rate': 'FEDFUNDS', 'cpi': 'CPALTT01USM657N',
        'bond5y': 'DGS5', 'bond10y': 'DGS10', 'bond20y': 'DGS20', 'bond30y': 'DGS30'
    },
    'ZAR': {
        'reer': 'RBZABIS', 'rate': 'IR3TIB01ZAM156N', 'cpi': 'CPALTT01ZAM659N',
        'bond10y': 'IRLTLT01ZAM156N'
    },
    'JPY': {
        'reer': 'RBJPBIS', 'rate': 'IR3TIB01JPM156N', 'cpi': 'FPCPITOTLZGJPN',
        'bond10y': 'IRLTLT01JPM156N'
    },
    'CNY': {
        'reer': 'RBCNBIS', 'rate': 'INTDSRCNM193N', 'cpi': 'CPALTT01CNM659N',
        'bond10y': 'IRLTLT01CNM156N'
    },
    'GBP': {
        'reer': 'RBGBBIS', 'rate': 'IR3TIB01GBM156N', 'cpi': 'CPALTT01GBM659N',
        'bond10y': 'IRLTLT01GBM156N'
    },
    'MXN': {
        'reer': 'RBMXBIS', 'rate': 'IR3TIB01MXM156N', 'cpi': 'CPALTT01MXM659N',
        'bond10y': 'IRLTLT01MXM156N'
    },
    'AUD': {
        'reer': 'RBAUBIS', 'rate': 'IR3TIB01AUM156N', 'cpi': 'FPCPITOTLZGAUS',
        'bond10y': 'IRLTLT01AUM156N'
    },
    'CAD': {
        'reer': 'RBCABIS', 'rate': 'IR3TIB01CAM156N', 'cpi': 'CPALTT01CAM659N',
        'bond10y': 'IRLTLT01CAM156N'
    },
    'INR': {
        'reer': 'RBINBIS', 'rate': 'IRSTCI01INM156N', 'cpi': 'CPALTT01INM659N',
        'bond10y': 'IRLTLT01INM156N'
    }
}

TENORS = ['5Y', '10Y', '15Y', '20Y']
start = datetime(2022, 1, 1)
end = datetime.now()

# ============================================================================
# DATA FETCHING FUNCTIONS
# ============================================================================
def fetch_single_series(series_id):
    """Fetch a single FRED series and return latest value."""
    try:
        df = web.DataReader(series_id, 'fred', start, end)
        return df.iloc[-1, 0]
    except:
        return np.nan


def interpolate_yield(y5, y10, y20, y30, tenor):
    """Interpolate yields for missing tenors using linear interpolation."""
    tenors_available = {5: y5, 10: y10, 20: y20, 30: y30}
    tenors_available = {k: v for k, v in tenors_available.items() if not np.isnan(v)}
    
    if tenor in tenors_available:
        return tenors_available[tenor]
    
    if len(tenors_available) < 2:
        return np.nan
    
    sorted_tenors = sorted(tenors_available.keys())
    for i in range(len(sorted_tenors) - 1):
        if sorted_tenors[i] < tenor < sorted_tenors[i + 1]:
            t1, t2 = sorted_tenors[i], sorted_tenors[i + 1]
            y1, y2 = tenors_available[t1], tenors_available[t2]
            return y1 + (y2 - y1) * (tenor - t1) / (t2 - t1)
    
    return np.nan


def estimate_tenor_yield(country_data, base_10y, tenor_years):
    """Estimate yield for a given tenor based on available data."""
    if country_data.get('bond5y') and country_data.get('bond20y'):
        y5 = fetch_single_series(country_data['bond5y'])
        y10 = base_10y
        y20 = fetch_single_series(country_data['bond20y'])
        y30 = fetch_single_series(country_data.get('bond30y', '')) if country_data.get('bond30y') else np.nan
        return interpolate_yield(y5, y10, y20, y30, tenor_years)
    else:
        # Estimate using typical term structure slope
        slope_per_year = 0.05  # ~5bp per year typical slope
        return base_10y + (tenor_years - 10) * slope_per_year


def fetch_macro_data(mapping):
    results = []
    print("Fetching data from FRED...")
    print("-" * 80)
    
    for country, ids in mapping.items():
        try:
            # Download core series
            reer_val = fetch_single_series(ids['reer'])
            rate_val = fetch_single_series(ids['rate'])
            cpi_val = fetch_single_series(ids['cpi'])
            bond10y_val = fetch_single_series(ids['bond10y'])
            
            # Calculate derived values
            real_rate = rate_val - cpi_val
            term_premium_10y = bond10y_val - rate_val
            
            # Estimate yields for all tenors
            bond5y_val = estimate_tenor_yield(ids, bond10y_val, 5)
            bond15y_val = estimate_tenor_yield(ids, bond10y_val, 15)
            bond20y_val = estimate_tenor_yield(ids, bond10y_val, 20)
            
            result = {
                'Country': country,
                'REER': reer_val,
                'Nominal_Rate': rate_val,
                'Inflation': cpi_val,
                'Real_Rate': real_rate,
                'Bond_5Y': bond5y_val,
                'Bond_10Y': bond10y_val,
                'Bond_15Y': bond15y_val,
                'Bond_20Y': bond20y_val,
                'Term_Premium_5Y': bond5y_val - rate_val,
                'Term_Premium_10Y': term_premium_10y,
                'Term_Premium_15Y': bond15y_val - rate_val,
                'Term_Premium_20Y': bond20y_val - rate_val
            }
            results.append(result)
            
            print(f"  {country:4s}: 5Y={bond5y_val:5.2f}% | 10Y={bond10y_val:5.2f}% | "
                  f"15Y={bond15y_val:5.2f}% | 20Y={bond20y_val:5.2f}%")
            
        except Exception as e:
            print(f"  {country}: FAILED - {str(e)[:50]}")
    
    return pd.DataFrame(results)


# ============================================================================
# MAIN EXECUTION
# ============================================================================
print("=" * 80)
print("Multi-Tenor Currency Valuation Analysis")
print("=" * 80)
df_final = fetch_macro_data(data_map)

# Calculate Differentials vs USD
usd_row = df_final.loc[df_final['Country'] == 'USA'].iloc[0]
usd_real_rate = usd_row['Real_Rate']
usd_inflation = usd_row['Inflation']

df_final['Diff_vs_USD'] = df_final['Real_Rate'] - usd_real_rate
df_final['Inflation_Diff'] = df_final['Inflation'] - usd_inflation

# Calculate Currency Risk Premium for each tenor
for tenor in ['5Y', '10Y', '15Y', '20Y']:
    bond_col = f'Bond_{tenor}'
    term_col = f'Term_Premium_{tenor}'
    df_final[f'Currency_Risk_Premium_{tenor}'] = (
        df_final[bond_col] - usd_real_rate - df_final['Inflation'] - df_final[term_col]
    )

# Linear Regression (REER ~ Differential)
df_final = df_final.dropna(subset=['REER', 'Diff_vs_USD'])
X = sm.add_constant(df_final['Diff_vs_USD'])
model = sm.OLS(df_final['REER'], X).fit()
df_final['Fitted_REER'] = model.predict(X)

# ============================================================================
# VISUALIZATION
# ============================================================================
footnote = ('Source: Federal Reserve Economic Data (FRED) | Data may have reporting lags of 1-3 months. '
            'Non-US tenors estimated via interpolation where direct data unavailable.\n'
            'Author: Mahlatse Ndala')


def add_footnote(fig, y_pos=0.02):
    fig.text(0.5, y_pos, footnote, ha='center', fontsize=8, style='italic', color='#666666')


# Plot 1: Bar Graph of Current REER
fig1, ax1 = plt.subplots(figsize=(11, 6))
df_sorted = df_final.sort_values('REER')
bars = ax1.bar(df_sorted['Country'], df_sorted['REER'], color=COLORS['primary'], 
               edgecolor='white', linewidth=1.5)
ax1.axhline(100, color=COLORS['negative'], linestyle='--', linewidth=2, label='Baseline (100)', alpha=0.8)
ax1.set_title('Real Effective Exchange Rate (REER) by Currency')
ax1.set_ylabel('REER Index')
ax1.set_xlabel('Currency')
ax1.legend(loc='upper left', frameon=True)

for bar, val in zip(bars, df_sorted['REER']):
    ax1.annotate(f'{val:.1f}', (bar.get_x() + bar.get_width()/2, bar.get_height() + 1),
                 ha='center', va='bottom', fontsize=9, fontweight='bold', color='#333333')

add_footnote(fig1)
plt.tight_layout()
plt.subplots_adjust(bottom=0.18)
plt.show()

# Plot 2: Scatter Plot with Regression Line
fig2, ax2 = plt.subplots(figsize=(11, 6))
ax2.scatter(df_final['Diff_vs_USD'], df_final['REER'], marker='o', s=120, 
            c=COLORS['primary'], edgecolors='white', linewidth=2, zorder=5)

for i, row in df_final.iterrows():
    ax2.annotate(row['Country'], (row['Diff_vs_USD'], row['REER'] + 1.5),
                 ha='center', fontsize=10, fontweight='bold', color='#333333')

x_range = np.linspace(df_final['Diff_vs_USD'].min() - 1, df_final['Diff_vs_USD'].max() + 1, 100)
ax2.plot(x_range, model.params[0] + model.params[1] * x_range, 
         color=COLORS['accent'], linewidth=2.5, label=f'Fitted Model (R²={model.rsquared:.2f})')

ax2.set_xlabel('Real Rate Differential vs USD (%)')
ax2.set_ylabel('REER Index')
ax2.set_title('REER vs Real Interest Rate Differential')
ax2.legend(loc='upper left', frameon=True)
add_footnote(fig2)
plt.tight_layout()
plt.subplots_adjust(bottom=0.18)
plt.show()

# Plot 3: Bar Chart of Nominal Yields (Short-term rates)
fig3, ax3 = plt.subplots(figsize=(11, 6))
df_nom_sorted = df_final.sort_values('Nominal_Rate')
colors = [COLORS['positive'] if x >= 0 else COLORS['negative'] for x in df_nom_sorted['Nominal_Rate']]
bars = ax3.bar(df_nom_sorted['Country'], df_nom_sorted['Nominal_Rate'], color=colors,
               edgecolor='white', linewidth=1.5)
ax3.axhline(0, color='#333333', linestyle='-', linewidth=1)
ax3.set_title('Short-Term Nominal Interest Rates by Currency')
ax3.set_ylabel('Nominal Rate (%)')
ax3.set_xlabel('Currency')
ax3.tick_params(axis='x', rotation=0)

for bar, val in zip(bars, df_nom_sorted['Nominal_Rate']):
    y_pos = bar.get_height() + 0.2 if val >= 0 else bar.get_height() - 0.4
    ax3.annotate(f'{val:.1f}%', (bar.get_x() + bar.get_width()/2, y_pos),
                 ha='center', va='bottom' if val >= 0 else 'top', fontsize=9, fontweight='bold')

add_footnote(fig3)
plt.tight_layout()
plt.subplots_adjust(bottom=0.18)
plt.show()

# Plot 4: Multi-Tenor Yield Curve Comparison
fig4, ax4 = plt.subplots(figsize=(12, 7))
tenor_labels = ['5Y', '10Y', '15Y', '20Y']
tenor_nums = [5, 10, 15, 20]

cmap = plt.cm.get_cmap('tab10')
for idx, (_, row) in enumerate(df_final.iterrows()):
    yields = [row['Bond_5Y'], row['Bond_10Y'], row['Bond_15Y'], row['Bond_20Y']]
    ax4.plot(tenor_nums, yields, marker='o', markersize=8, linewidth=2.5,
             label=row['Country'], color=cmap(idx), alpha=0.85)

ax4.set_xlabel('Tenor (Years)')
ax4.set_ylabel('Yield (%)')
ax4.set_title('Yield Curve Comparison Across Currencies')
ax4.set_xticks(tenor_nums)
ax4.set_xticklabels(tenor_labels)
ax4.legend(loc='upper left', frameon=True, ncol=3)
add_footnote(fig4)
plt.tight_layout()
plt.subplots_adjust(bottom=0.15)
plt.show()

# Plot 5: Treemap - 10Y Yield Decomposition
fig5, axes = plt.subplots(3, 3, figsize=(14, 12))
axes = axes.flatten()

colors_decomp = {
    'Real Rate (USD)': COLORS['real_rate'],
    'Inflation': COLORS['inflation'],
    'Term Premium': COLORS['term_premium'],
    'Currency Risk': COLORS['currency_risk']
}

for idx, (_, row) in enumerate(df_final.iterrows()):
    if idx >= 9:
        break
    ax = axes[idx]
    
    components = {
        'Real Rate (USD)': usd_real_rate,
        'Inflation': row['Inflation'],
        'Term Premium': row['Term_Premium_10Y'],
        'Currency Risk': row['Currency_Risk_Premium_10Y']
    }
    
    sizes = [max(abs(v), 0.1) for v in components.values()]
    labels = [f"{k}\n{v:+.1f}%" for k, v in components.items()]
    colors = [colors_decomp[k] for k in components.keys()]
    
    squarify.plot(sizes=sizes, label=labels, color=colors, alpha=0.85, ax=ax,
                  text_kwargs={'fontsize': 8, 'fontweight': 'bold', 'color': 'white'})
    ax.set_title(f"{row['Country']} (10Y: {row['Bond_10Y']:.1f}%)", fontweight='bold', fontsize=11)
    ax.axis('off')

fig5.suptitle('10-Year Yield Decomposition by Currency', fontsize=16, fontweight='bold', y=0.98)
add_footnote(fig5, y_pos=0.01)
plt.tight_layout()
plt.subplots_adjust(bottom=0.06, top=0.93)
plt.show()

# Plot 6: Stacked Bar - Multi-Tenor Decomposition (10Y focus)
fig6, ax6 = plt.subplots(figsize=(13, 7))
df_plot = df_final.sort_values('Bond_10Y')

x = np.arange(len(df_plot))
width = 0.65

bottom = np.zeros(len(df_plot))
components = [
    ('Real Rate (USD)', [usd_real_rate] * len(df_plot), COLORS['real_rate']),
    ('Inflation', df_plot['Inflation'].values, COLORS['inflation']),
    ('Term Premium', df_plot['Term_Premium_10Y'].values, COLORS['term_premium']),
    ('Currency Risk Premium', df_plot['Currency_Risk_Premium_10Y'].values, COLORS['currency_risk'])
]

for label, values, color in components:
    ax6.bar(x, values, width, label=label, bottom=bottom, color=color, 
            edgecolor='white', linewidth=0.5)
    bottom += np.array(values)

ax6.set_xlabel('Currency')
ax6.set_ylabel('Yield Component (%)')
ax6.set_title('10-Year Yield Decomposition: Real Rate + Inflation + Term Premium + Currency Risk Premium')
ax6.set_xticks(x)
ax6.set_xticklabels(df_plot['Country'])
ax6.legend(loc='upper left', frameon=True)
ax6.axhline(0, color='#333333', linewidth=1)

for i, (_, row) in enumerate(df_plot.iterrows()):
    ax6.annotate(f"{row['Bond_10Y']:.1f}%", (i, row['Bond_10Y'] + 0.3),
                 ha='center', fontsize=10, fontweight='bold', color='#333333')

add_footnote(fig6)
plt.tight_layout()
plt.subplots_adjust(bottom=0.12)
plt.show()

# Plot 7: Multi-Tenor Currency Risk Premium Comparison
fig7, ax7 = plt.subplots(figsize=(13, 7))
df_plot = df_final[df_final['Country'] != 'USA'].sort_values('Currency_Risk_Premium_10Y')

x = np.arange(len(df_plot))
width = 0.2

tenors_data = [
    ('5Y', df_plot['Currency_Risk_Premium_5Y'].values, '#7cb342'),
    ('10Y', df_plot['Currency_Risk_Premium_10Y'].values, '#1976d2'),
    ('15Y', df_plot['Currency_Risk_Premium_15Y'].values, '#f57c00'),
    ('20Y', df_plot['Currency_Risk_Premium_20Y'].values, '#7b1fa2')
]

for i, (label, values, color) in enumerate(tenors_data):
    offset = (i - 1.5) * width
    bars = ax7.bar(x + offset, values, width, label=label, color=color, 
                   edgecolor='white', linewidth=0.5)

ax7.set_xlabel('Currency')
ax7.set_ylabel('Currency Risk Premium (%)')
ax7.set_title('Currency Risk Premium by Tenor (vs USD)')
ax7.set_xticks(x)
ax7.set_xticklabels(df_plot['Country'])
ax7.legend(title='Tenor', loc='upper left', frameon=True)
ax7.axhline(0, color='#333333', linewidth=1)

add_footnote(fig7)
plt.tight_layout()
plt.subplots_adjust(bottom=0.12)
plt.show()

# ============================================================================
# SUMMARY TABLE
# ============================================================================
print("\n" + "=" * 100)
print("Multi-Tenor Yield Decomposition Summary")
print("=" * 100)
summary_cols = ['Country', 'Bond_5Y', 'Bond_10Y', 'Bond_15Y', 'Bond_20Y', 
                'Currency_Risk_Premium_5Y', 'Currency_Risk_Premium_10Y', 
                'Currency_Risk_Premium_15Y', 'Currency_Risk_Premium_20Y']
print(df_final[summary_cols].to_string(index=False, float_format='%.2f'))
print("=" * 100)
