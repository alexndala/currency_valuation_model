import pandas as pd
import numpy as np
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import statsmodels.api as sm
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 1. Define countries with verified working FRED Series IDs
data_map = {
    'USA':  {'reer': 'RBUSBIS', 'rate': 'FEDFUNDS',       'cpi': 'CPALTT01USM657N'},
    'ZAR':  {'reer': 'RBZABIS', 'rate': 'IR3TIB01ZAM156N', 'cpi': 'CPALTT01ZAM659N'},
    'JPY':  {'reer': 'RBJPBIS', 'rate': 'IR3TIB01JPM156N', 'cpi': 'FPCPITOTLZGJPN'},
    'CNY':  {'reer': 'RBCNBIS', 'rate': 'INTDSRCNM193N',   'cpi': 'CPALTT01CNM659N'},
    # 'EURO': {'reer': 'RBEEBIS', 'rate': 'ECBDFR',          'cpi': 'EA19CPALTT01GYM'},
    'GBP':  {'reer': 'RBGBBIS', 'rate': 'IR3TIB01GBM156N', 'cpi': 'CPALTT01GBM659N'},
    # 'BRL':  {'reer': 'RBBRBIS', 'rate': 'INTDSRBRM193N',   'cpi': 'CPALTT01BRM659N'},
    'MXN':  {'reer': 'RBMXBIS', 'rate': 'IR3TIB01MXM156N', 'cpi': 'CPALTT01MXM659N'},
    'AUD':  {'reer': 'RBAUBIS', 'rate': 'IR3TIB01AUM156N', 'cpi': 'FPCPITOTLZGAUS'},
    'CAD':  {'reer': 'RBCABIS', 'rate': 'IR3TIB01CAM156N', 'cpi': 'CPALTT01CAM659N'},
    # 'CHF':  {'reer': 'RBCHBIS', 'rate': 'IR3TIB01CHM156N', 'cpi': 'CPALTT01CHM659N'},
    'INR':  {'reer': 'RBINBIS', 'rate': 'IRSTCI01INM156N', 'cpi': 'CPALTT01INM659N'}
}

# 2. Fetch Data
start = datetime(2022, 1, 1)
end = datetime.now()

def fetch_macro_data(mapping):
    results = []
    print("Fetching data from FRED...")
    for country, ids in mapping.items():
        try:
            # Download each series separately to handle different frequencies
            reer_df = web.DataReader(ids['reer'], 'fred', start, end)
            rate_df = web.DataReader(ids['rate'], 'fred', start, end)
            cpi_df = web.DataReader(ids['cpi'], 'fred', start, end)
            
            # Get latest values
            latest_reer = reer_df.iloc[-1, 0]
            latest_nom_rate = rate_df.iloc[-1, 0]
            latest_inflation = cpi_df.iloc[-1, 0]
            
            # Calculate real rate
            real_rate = latest_nom_rate - latest_inflation
            
            results.append({
                'Country': country, 
                'REER': latest_reer, 
                'Nominal_Rate': latest_nom_rate,
                'Inflation': latest_inflation,
                'Real_Rate': real_rate
            })
            print(f"  {country}: REER={latest_reer:.2f}, Rate={latest_nom_rate:.2f}%, Inflation={latest_inflation:.2f}%, Real Rate={real_rate:.2f}%")
        except Exception as e:
            print(f"  {country}: FAILED - {str(e)[:50]}")
    return pd.DataFrame(results)

print("=" * 60)
print("Real Effective Exchange Rate Analysis")
print("=" * 60)
df_final = fetch_macro_data(data_map)

# 3. Calculate Differentials vs USD
usd_real_rate = df_final.loc[df_final['Country'] == 'USA', 'Real_Rate'].values[0]
df_final['Diff_vs_USD'] = df_final['Real_Rate'] - usd_real_rate

# 4. Linear Regression (REER ~ Differential)
df_final = df_final.dropna()
X = sm.add_constant(df_final['Diff_vs_USD'])
model = sm.OLS(df_final['REER'], X).fit()
df_final['Fitted_REER'] = model.predict(X)

# --- Visualization ---
footnote = ('Source: Federal Reserve Economic Data (FRED) | Data may have reporting lags of 1-3 months\n'
            'Author: Mahlatse Ndala')

# Plot 1: Bar Graph of Current REER
fig1, ax1 = plt.subplots(figsize=(10, 6))
df_sorted = df_final.sort_values('REER')
ax1.bar(df_sorted['Country'], df_sorted['REER'], color='steelblue')
ax1.axhline(100, color='red', linestyle='--', label='Baseline (100)')
ax1.set_title('Current Real Effective Exchange Rate (REER)')
ax1.set_ylabel('REER')
ax1.set_xlabel('Currency')
ax1.legend()
fig1.text(0.5, 0.02, footnote, ha='center', fontsize=9, style='italic')
plt.tight_layout()
plt.subplots_adjust(bottom=0.22)
plt.show()

# Plot 2: Scatter Plot with Regression Line
fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.scatter(df_final['Diff_vs_USD'], df_final['REER'], marker='x', s=100)
for i, txt in enumerate(df_final['Country']):
    ax2.annotate(txt, (df_final['Diff_vs_USD'].iloc[i], df_final['REER'].iloc[i] + 1))

x_range = np.linspace(df_final['Diff_vs_USD'].min() - 0.02, df_final['Diff_vs_USD'].max() + 0.02, 100)
ax2.plot(x_range, model.params[0] + model.params[1] * x_range, color='black', label='Fitted REER Model')

ax2.set_xlabel('Real Rate Differential vs USD (%)')
ax2.set_ylabel('REER')
ax2.set_title('REER vs Real Rate Differential')
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.legend()
fig2.text(0.5, 0.02, footnote, ha='center', fontsize=9, style='italic')
plt.tight_layout()
plt.subplots_adjust(bottom=0.22)
plt.show()

# Plot 3: Bar Chart of Nominal Yields
fig3, ax3 = plt.subplots(figsize=(10, 6))
df_nom_sorted = df_final.sort_values('Nominal_Rate')
colors = ['green' if x >= 0 else 'red' for x in df_nom_sorted['Nominal_Rate']]
ax3.bar(df_nom_sorted['Country'], df_nom_sorted['Nominal_Rate'], color=colors)
ax3.axhline(0, color='black', linestyle='-', linewidth=0.5)
ax3.set_title('Nominal Interest Rates by Currency')
ax3.set_ylabel('Nominal Rate (%)')
ax3.set_xlabel('Currency')
ax3.tick_params(axis='x', rotation=45)
ax3.grid(True, linestyle=':', alpha=0.6)
fig3.text(0.5, 0.02, footnote, ha='center', fontsize=9, style='italic')
plt.tight_layout()
plt.subplots_adjust(bottom=0.22)
plt.show()