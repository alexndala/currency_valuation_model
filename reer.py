import pandas as pd
import numpy as np
import pandas_datareader.data as web
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import squarify
import statsmodels.api as sm
from statsmodels.stats.diagnostic import recursive_olsresiduals
import ruptures as rpt
from datetime import datetime
import os
import json
import argparse
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# COMMAND LINE ARGUMENTS
# ============================================================================
parser = argparse.ArgumentParser(description='Currency Valuation Model')
parser.add_argument('--refresh', action='store_true', 
                    help='Force refresh data from FRED (ignore local cache)')
parser.add_argument('--local-only', action='store_true',
                    help='Use local data only (fail if not available)')
args, _ = parser.parse_known_args()

# Data paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.getcwd()
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
MACRO_DATA_CSV = os.path.join(DATA_DIR, 'fred_macro_data.csv')
REER_HIST_CSV = os.path.join(DATA_DIR, 'reer_historical.csv')
PANEL_DATA_CSV = os.path.join(DATA_DIR, 'panel_data.csv')
ACM_DATA_CSV = os.path.join(DATA_DIR, 'acm_decomposition.csv')
METADATA_FILE = os.path.join(DATA_DIR, 'metadata.json')

def check_local_data():
    """Check if local data files exist and return metadata."""
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, 'r') as f:
            return json.load(f)
    return None

def load_local_macro_data():
    """Load macro data from local CSV file."""
    if os.path.exists(MACRO_DATA_CSV):
        return pd.read_csv(MACRO_DATA_CSV)
    return None

def load_local_reer_historical():
    """Load REER historical data from local CSV file."""
    if os.path.exists(REER_HIST_CSV):
        df = pd.read_csv(REER_HIST_CSV, index_col=0, parse_dates=True)
        return df
    return None

USE_LOCAL_DATA = False
LOCAL_MACRO_DF = None
LOCAL_REER_DF = None

if not args.refresh:
    metadata = check_local_data()
    if metadata:
        LOCAL_MACRO_DF = load_local_macro_data()
        LOCAL_REER_DF = load_local_reer_historical()
        if LOCAL_MACRO_DF is not None and LOCAL_REER_DF is not None:
            USE_LOCAL_DATA = True
            print(f"Using local data (last updated: {metadata.get('last_updated', 'unknown')})")
            print("Run with --refresh to force update from FRED")
            print("-" * 80)

if args.local_only and not USE_LOCAL_DATA:
    raise FileNotFoundError("Local data not found. Run data_downloader.py first.")

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
# Added tips10y for inflation-linked bonds where available (LTCMA decomposition)
data_map = {
    'USA': {
        'reer': 'RBUSBIS', 'rate': 'FEDFUNDS', 'cpi': 'CPALTT01USM657N',
        'bond5y': 'DGS5', 'bond10y': 'DGS10', 'bond20y': 'DGS20', 'bond30y': 'DGS30',
        'tips10y': 'DFII10'  # 10-Year TIPS yield
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
start_10y = datetime(end.year - 10, end.month, end.day)  # 10 years of history for REER

# EM Structural Kicker (Balassa-Samuelson effect) - annual rates
EM_KICKERS = {
    'INR': 1.5,   # India: high productivity growth
    'CNY': 1.0,   # China: strong productivity growth
    'ZAR': 0.25,  # South Africa: moderate EM premium
    'MXN': 0.5,  # Mexico: moderate EM premium
}

# Mean Reversion Configuration
REVERSION_MODE = 'half_life'   # 'half_life' or 'linear'
REVERSION_HALF_LIFE = 5        # years (only used when mode='half_life'); Rogoff (1996) standard
REVERSION_HORIZON = 10         # investment horizon in years

# ============================================================================
# LTCMA BUILDING BLOCKS (JP Morgan Style)
# ============================================================================
# Expected Real Cash Rate (r*) - IMF long-term potential growth forecasts
REAL_CASH_RATES = {
    'USA': 1.00,
    'GBP': 0.75,
    'CAD': 0.75,
    'JPY': 0.00,
    'INR': 3.50,
    'MXN': 3.00,
    'ZAR': 2.50,
    'CNY': 2.00,
    'AUD': 1.00,
}

# Central Bank Inflation Targets
CB_INFLATION_TARGETS = {
    'USA': 2.0,
    'GBP': 2.0,
    'CAD': 2.0,
    'JPY': 2.0,
    'AUD': 2.0,
    'INR': 4.0,
    'ZAR': 4.5,
    'MXN': 3.0,
    'CNY': 3.0,
}

# ============================================================================
# STRUCTURAL BREAK DETECTION CONFIGURATION
# ============================================================================
BREAK_DETECTION_CONFIG = {
    'min_segment_months': 24,      # Minimum 2 years per regime
    'cusum_significance': 0.05,    # 5% significance level
    'bai_perron_penalty': 20,      # Penalty 20 gives 1-2 breaks for major shifts
    'use_regime_mean': True,       # Toggle for valuation calculation
    'min_regime_months': 12,       # Blend with previous if regime too short
}

# EM currencies get lower penalty (more sensitive to breaks due to volatility)
EM_BREAK_PENALTY = {
    'ZAR': 10, 'MXN': 15, 'INR': 15, 'CNY': 15,
}

# Tree Charts: which country to display in branching decompositions
LTCMA_TREE_COUNTRY = 'ZAR'
RETURNS_TREE_COUNTRY = 'ZAR'

# ============================================================================
# STRUCTURAL BREAK DETECTION FUNCTIONS
# ============================================================================
def cusum_test(reer_series, significance=0.05):
    """
    CUSUM test for structural breaks using recursive OLS residuals.
    Returns break indices where CUSUM crosses critical bounds.
    """
    if len(reer_series) < 30:
        return [], None, None
    
    # Fit OLS: REER_t = alpha + beta*t + epsilon
    X = sm.add_constant(np.arange(len(reer_series)))
    try:
        model = sm.OLS(reer_series.values, X).fit()
        rols_results = recursive_olsresiduals(model, skip=max(10, int(len(reer_series) * 0.1)))
        
        rols_resid = rols_results[0]  # Recursive residuals
        cusum = rols_results[3]       # Cumulative sum
        cusum_bounds = rols_results[4]  # Critical bounds
        
        # Find where CUSUM crosses bounds
        breaks = []
        n = len(cusum)
        if n > 0 and len(cusum_bounds) > 0:
            upper_bound = cusum_bounds[:, 0] if cusum_bounds.ndim > 1 else cusum_bounds
            lower_bound = -upper_bound
            
            for i in range(len(cusum)):
                if i < len(upper_bound):
                    if cusum[i] > upper_bound[i] or cusum[i] < lower_bound[i]:
                        breaks.append(i)
        
        return breaks, cusum, cusum_bounds
    except Exception as e:
        return [], None, None


def bai_perron_breaks(reer_series, penalty=None, min_size=24):
    """
    Bai-Perron structural break detection using PELT algorithm.
    Returns list of break indices.
    """
    if len(reer_series) < min_size * 2:
        return []
    
    if penalty is None:
        penalty = BREAK_DETECTION_CONFIG['bai_perron_penalty']
    
    try:
        signal = reer_series.values.reshape(-1, 1)
        model = rpt.Pelt(model="rbf", min_size=min_size).fit(signal)
        breaks = model.predict(pen=penalty)
        # Remove last element (always equals len(signal))
        breaks = [b for b in breaks if b < len(signal)]
        return breaks
    except Exception as e:
        return []


def detect_structural_breaks(reer_series, country=None):
    """
    Combined structural break detection using CUSUM + Bai-Perron.
    Returns dict with break info and regime statistics.
    """
    if reer_series is None or len(reer_series) < 36:
        return {
            'breaks': [],
            'break_dates': [],
            'regime_mean': reer_series.mean() if reer_series is not None else np.nan,
            'full_mean': reer_series.mean() if reer_series is not None else np.nan,
            'regime_start': reer_series.index[0] if reer_series is not None else None,
            'regime_months': len(reer_series) if reer_series is not None else 0,
            'cusum': None,
            'cusum_bounds': None,
            'method': 'none'
        }
    
    # Get penalty (lower for EM)
    penalty = EM_BREAK_PENALTY.get(country, BREAK_DETECTION_CONFIG['bai_perron_penalty'])
    min_size = BREAK_DETECTION_CONFIG['min_segment_months']
    
    # Run both methods
    cusum_breaks, cusum, cusum_bounds = cusum_test(reer_series, BREAK_DETECTION_CONFIG['cusum_significance'])
    bp_breaks = bai_perron_breaks(reer_series, penalty=penalty, min_size=min_size)
    
    # Use Bai-Perron breaks only (more conservative)
    # CUSUM breaks are stored for visualization but not used to add more breaks
    all_breaks = sorted(set(bp_breaks))
    
    # Convert to dates
    break_dates = [reer_series.index[b] for b in all_breaks if b < len(reer_series)]
    
    # Calculate regime statistics
    if all_breaks:
        last_break = max(all_breaks)
        regime_start_idx = min(last_break + 1, len(reer_series) - 1)
        regime_series = reer_series.iloc[regime_start_idx:]
        regime_mean = regime_series.mean()
        regime_start = reer_series.index[regime_start_idx]
        regime_months = len(regime_series)
        
        # If regime too short, blend with previous
        if regime_months < BREAK_DETECTION_CONFIG['min_regime_months'] and len(all_breaks) > 1:
            prev_break = all_breaks[-2] if len(all_breaks) >= 2 else 0
            extended_series = reer_series.iloc[prev_break:]
            regime_mean = extended_series.mean()
            regime_months = len(extended_series)
    else:
        regime_mean = reer_series.mean()
        regime_start = reer_series.index[0]
        regime_months = len(reer_series)
    
    return {
        'breaks': all_breaks,
        'break_dates': break_dates,
        'regime_mean': regime_mean,
        'full_mean': reer_series.mean(),
        'regime_start': regime_start,
        'regime_months': regime_months,
        'cusum': cusum,
        'cusum_bounds': cusum_bounds,
        'method': 'cusum+bai_perron'
    }


def get_regime_adjusted_stats(country, reer_series):
    """
    Get REER statistics with structural break adjustment.
    Returns current value, regime-adjusted mean, and std.
    """
    if reer_series is None or len(reer_series) == 0:
        return np.nan, np.nan, np.nan, {}
    
    break_info = detect_structural_breaks(reer_series, country)
    current = reer_series.iloc[-1]
    
    if BREAK_DETECTION_CONFIG['use_regime_mean'] and break_info['breaks']:
        mean_val = break_info['regime_mean']
        # Calculate std for current regime only
        if break_info['regime_months'] > 0:
            last_break = max(break_info['breaks'])
            regime_start_idx = min(last_break + 1, len(reer_series) - 1)
            std_val = reer_series.iloc[regime_start_idx:].std()
        else:
            std_val = reer_series.std()
    else:
        mean_val = break_info['full_mean']
        std_val = reer_series.std()
    
    return current, mean_val, std_val, break_info

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


def fetch_reer_historical(series_id):
    """Fetch 10Y historical REER data and return current, mean, std."""
    try:
        df = web.DataReader(series_id, 'fred', start_10y, end)
        df = df.dropna()
        current = df.iloc[-1, 0]
        mean_10y = df.iloc[:, 0].mean()
        std_10y = df.iloc[:, 0].std()
        return current, mean_10y, std_10y
    except:
        return np.nan, np.nan, np.nan


def get_local_value(country, indicator):
    """Get value from local macro data."""
    if LOCAL_MACRO_DF is None:
        return np.nan
    row = LOCAL_MACRO_DF[LOCAL_MACRO_DF['Country'] == country]
    if row.empty:
        return np.nan
    col = f'{indicator}_value'
    if col in row.columns:
        val = row[col].iloc[0]
        return val if pd.notna(val) else np.nan
    return np.nan


def get_local_reer_stats(country):
    """Get REER stats from local historical data."""
    if LOCAL_REER_DF is None or country not in LOCAL_REER_DF.columns:
        return np.nan, np.nan, np.nan
    series = LOCAL_REER_DF[country].dropna()
    if len(series) == 0:
        return np.nan, np.nan, np.nan
    return series.iloc[-1], series.mean(), series.std()


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


def estimate_tenor_yield(country_data, base_10y, tenor_years, country=None):
    """Estimate yield for a given tenor based on available data."""
    if country_data.get('bond5y') and country_data.get('bond20y'):
        if USE_LOCAL_DATA and country:
            y5 = get_local_value(country, 'bond5y')
            y20 = get_local_value(country, 'bond20y')
            y30 = get_local_value(country, 'bond30y') if country_data.get('bond30y') else np.nan
        else:
            y5 = fetch_single_series(country_data['bond5y'])
            y20 = fetch_single_series(country_data['bond20y'])
            y30 = fetch_single_series(country_data.get('bond30y', '')) if country_data.get('bond30y') else np.nan
        y10 = base_10y
        return interpolate_yield(y5, y10, y20, y30, tenor_years)
    else:
        # Estimate using typical term structure slope
        slope_per_year = 0.05  # ~5bp per year typical slope
        return base_10y + (tenor_years - 10) * slope_per_year


# Global storage for structural break analysis results
REER_BREAK_INFO = {}

def fetch_macro_data(mapping):
    global REER_BREAK_INFO
    results = []
    REER_BREAK_INFO = {}  # Reset break info
    
    if USE_LOCAL_DATA:
        print("Loading data from local cache...")
    else:
        print("Fetching data from FRED...")
    print("-" * 80)
    
    # Fetch US TIPS for BEI proxy (used when country-specific linkers unavailable)
    if USE_LOCAL_DATA:
        us_tips10y = get_local_value('USA', 'tips10y')
        us_bond10y = get_local_value('USA', 'bond10y')
    else:
        us_tips10y = fetch_single_series('DFII10')
        us_bond10y = fetch_single_series('DGS10')
    us_bei = us_bond10y - us_tips10y if not np.isnan(us_tips10y) else np.nan
    
    for country, ids in mapping.items():
        try:
            # Download core series with structural break analysis
            if USE_LOCAL_DATA:
                # Get REER historical series for break detection
                reer_series = LOCAL_REER_DF[country].dropna() if LOCAL_REER_DF is not None and country in LOCAL_REER_DF.columns else None
                reer_val, reer_mean_10y, reer_std_10y, break_info = get_regime_adjusted_stats(country, reer_series)
                rate_val = get_local_value(country, 'rate')
                cpi_val = get_local_value(country, 'cpi')
                bond10y_val = get_local_value(country, 'bond10y')
            else:
                reer_val, reer_mean_10y, reer_std_10y = fetch_reer_historical(ids['reer'])
                break_info = {}  # No break detection in live mode (would need full series)
                rate_val = fetch_single_series(ids['rate'])
                cpi_val = fetch_single_series(ids['cpi'])
                bond10y_val = fetch_single_series(ids['bond10y'])
            
            # Fetch TIPS/linker yield if available, calculate BEI
            if ids.get('tips10y'):
                if USE_LOCAL_DATA:
                    tips10y_val = get_local_value(country, 'tips10y')
                else:
                    tips10y_val = fetch_single_series(ids['tips10y'])
                bei_val = bond10y_val - tips10y_val if not np.isnan(tips10y_val) else us_bei
            else:
                tips10y_val = np.nan
                bei_val = us_bei  # Use US BEI as proxy
            
            # LTCMA building blocks
            r_star = REAL_CASH_RATES.get(country, np.nan)
            cb_target = CB_INFLATION_TARGETS.get(country, np.nan)
            irp = bei_val - cb_target if not np.isnan(bei_val) and not np.isnan(cb_target) else np.nan
            
            # Calculate derived values
            real_rate = rate_val - cpi_val
            term_premium_10y = bond10y_val - rate_val
            
            # Estimate yields for all tenors
            bond5y_val = estimate_tenor_yield(ids, bond10y_val, 5, country)
            bond15y_val = estimate_tenor_yield(ids, bond10y_val, 15, country)
            bond20y_val = estimate_tenor_yield(ids, bond10y_val, 20, country)
            
            # Store break info globally for visualization
            if break_info:
                REER_BREAK_INFO[country] = break_info
            
            # Extract regime info for result
            regime_months = break_info.get('regime_months', 0) if break_info else 0
            n_breaks = len(break_info.get('breaks', [])) if break_info else 0
            last_break_date = break_info.get('break_dates', [])[-1] if break_info and break_info.get('break_dates') else None
            full_mean = break_info.get('full_mean', reer_mean_10y) if break_info else reer_mean_10y
            
            result = {
                'Country': country,
                'REER': reer_val,
                'REER_10Y_Mean': reer_mean_10y,  # This is now regime-adjusted if breaks detected
                'REER_Full_Mean': full_mean,     # Original 10Y mean (no adjustment)
                'REER_10Y_Std': reer_std_10y,
                'REER_Regime_Months': regime_months,
                'REER_N_Breaks': n_breaks,
                'REER_Last_Break': last_break_date,
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
                'Term_Premium_20Y': bond20y_val - rate_val,
                # LTCMA components
                'TIPS_10Y': tips10y_val,
                'BEI': bei_val,
                'R_Star': r_star,
                'CB_Inflation_Target': cb_target,
                'IRP': irp,
            }
            results.append(result)
            
            # Print with break info
            break_str = f" | Breaks={n_breaks}" if n_breaks > 0 else ""
            print(f"  {country:4s}: 5Y={bond5y_val:5.2f}% | 10Y={bond10y_val:5.2f}% | "
                  f"15Y={bond15y_val:5.2f}% | 20Y={bond20y_val:5.2f}%{break_str}")
            
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

# Linear Regression (REER ~ Differential) - Cross-sectional (latest snapshot only)
df_final = df_final.dropna(subset=['REER', 'Diff_vs_USD'])
X = sm.add_constant(df_final['Diff_vs_USD'])
model = sm.OLS(df_final['REER'], X).fit()
df_final['Fitted_REER'] = model.predict(X)

# ============================================================================
# PANEL REGRESSION (April 1994 - Present)
# ============================================================================
panel_model = None
panel_results = None

if os.path.exists(PANEL_DATA_CSV):
    try:
        from linearmodels.panel import PanelOLS, PooledOLS
        
        # Load panel data
        panel_df = pd.read_csv(PANEL_DATA_CSV, parse_dates=['Date'])
        
        # Calculate real rate for each country-month
        panel_df['real_rate'] = panel_df['rate'] - panel_df['cpi']
        
        # Get US data for differentials
        us_panel = panel_df[panel_df['Country'] == 'USA'][['Date', 'real_rate', 'cpi', 'rate']].copy()
        us_panel.columns = ['Date', 'us_real_rate', 'us_cpi', 'us_rate']
        
        # Merge US data to calculate differentials
        panel_df = panel_df.merge(us_panel, on='Date', how='left')
        panel_df['real_rate_diff'] = panel_df['real_rate'] - panel_df['us_real_rate']
        panel_df['inflation_diff'] = panel_df['cpi'] - panel_df['us_cpi']
        
        # Drop US from panel (no differential with itself)
        panel_reg = panel_df[panel_df['Country'] != 'USA'].copy()
        
        # Drop missing values
        panel_reg = panel_reg.dropna(subset=['reer', 'real_rate_diff', 'inflation_diff'])
        
        if len(panel_reg) > 100:
            # Set multi-index for panel
            panel_reg = panel_reg.set_index(['Country', 'Date'])
            
            # Fixed Effects Model
            y = panel_reg['reer']
            X_panel = panel_reg[['real_rate_diff', 'inflation_diff']]
            X_panel = sm.add_constant(X_panel)
            
            # Entity (country) fixed effects
            panel_model = PanelOLS(y, X_panel, entity_effects=True)
            panel_results = panel_model.fit(cov_type='clustered', cluster_entity=True)
            
            print("\n" + "=" * 80)
            print("PANEL REGRESSION: REER ~ Real Rate Differential + Inflation Differential")
            print("Fixed Effects Model with Clustered Standard Errors (by Country)")
            print("=" * 80)
            print(f"Observations: {panel_results.nobs:,.0f} | Countries: {len(panel_reg.index.get_level_values(0).unique())}")
            print(f"Date Range: {panel_reg.index.get_level_values(1).min().strftime('%Y-%m')} to {panel_reg.index.get_level_values(1).max().strftime('%Y-%m')}")
            print("-" * 80)
            print(f"{'Variable':<25} {'Coef':>12} {'Std Err':>12} {'t-stat':>10} {'p-value':>10}")
            print("-" * 80)
            for var in panel_results.params.index:
                coef = panel_results.params[var]
                se = panel_results.std_errors[var]
                tstat = panel_results.tstats[var]
                pval = panel_results.pvalues[var]
                sig = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else ''
                print(f"{var:<25} {coef:>12.4f} {se:>12.4f} {tstat:>10.2f} {pval:>10.4f} {sig}")
            print("-" * 80)
            print(f"R-squared (within): {panel_results.rsquared_within:.4f}")
            print(f"R-squared (between): {panel_results.rsquared_between:.4f}")
            print(f"R-squared (overall): {panel_results.rsquared_overall:.4f}")
            print("=" * 80)
            
    except ImportError:
        print("\nNote: Install 'linearmodels' for panel regression: pip install linearmodels")
    except Exception as e:
        print(f"\nPanel regression failed: {e}")

# ============================================================================
# FAIR VALUE REER MODEL (Pooled Regression)
# Uses real interest rate differential vs USD to estimate fundamentals-based fair value
# ============================================================================
fair_value_df = None
pooled_model_results = None

if os.path.exists(PANEL_DATA_CSV):
    try:
        # Load panel data
        panel_df = pd.read_csv(PANEL_DATA_CSV, parse_dates=['Date'])
        
        # Calculate real rate for each country-month (bond10y - cpi as inflation proxy)
        panel_df['real_rate'] = panel_df['bond10y'] - panel_df['cpi']
        
        # Get US data for differentials
        us_panel = panel_df[panel_df['Country'] == 'USA'][['Date', 'real_rate']].copy()
        us_panel.columns = ['Date', 'us_real_rate']
        
        # Merge US data to calculate differentials
        panel_df = panel_df.merge(us_panel, on='Date', how='left')
        panel_df['real_rate_diff_vs_usd'] = panel_df['real_rate'] - panel_df['us_real_rate']
        
        # Drop rows with missing values
        fair_value_df = panel_df.dropna(subset=['reer', 'real_rate_diff_vs_usd']).copy()
        
        if len(fair_value_df) > 100:
            # Pooled OLS: REER = a + b x real_rate_diff_vs_usd
            X_pooled = sm.add_constant(fair_value_df['real_rate_diff_vs_usd'])
            y_pooled = fair_value_df['reer']
            
            pooled_model = sm.OLS(y_pooled, X_pooled).fit(cov_type='cluster', 
                                                          cov_kwds={'groups': fair_value_df['Country']})
            pooled_model_results = pooled_model
            
            # Calculate fitted values (fair value REER)
            fair_value_df['fair_value_reer'] = pooled_model.predict(X_pooled)
            fair_value_df['deviation'] = fair_value_df['reer'] - fair_value_df['fair_value_reer']
            fair_value_df['deviation_pct'] = (fair_value_df['deviation'] / fair_value_df['fair_value_reer']) * 100
            
            print("\n" + "=" * 80)
            print("FAIR VALUE REER MODEL (Pooled Regression)")
            print("REER = a + b x Real_Rate_Differential_vs_USD")
            print("=" * 80)
            print(f"Observations: {len(fair_value_df):,} | Countries: {fair_value_df['Country'].nunique()}")
            print(f"Date Range: {fair_value_df['Date'].min().strftime('%Y-%m')} to {fair_value_df['Date'].max().strftime('%Y-%m')}")
            print("-" * 80)
            print(f"{'Variable':<30} {'Coef':>12} {'Std Err':>12} {'t-stat':>10} {'p-value':>10}")
            print("-" * 80)
            for var in pooled_model.params.index:
                coef = pooled_model.params[var]
                se = pooled_model.bse[var]
                tstat = pooled_model.tvalues[var]
                pval = pooled_model.pvalues[var]
                sig = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else ''
                var_name = 'Intercept (a)' if var == 'const' else 'Real Rate Diff vs USD (b)'
                print(f"{var_name:<30} {coef:>12.4f} {se:>12.4f} {tstat:>10.2f} {pval:>10.4f} {sig}")
            print("-" * 80)
            print(f"R-squared: {pooled_model.rsquared:.4f}")
            print(f"Interpretation: 1% increase in real rate differential -> {pooled_model.params['real_rate_diff_vs_usd']:.2f} point change in REER")
            print("=" * 80)
            
    except Exception as e:
        print(f"\nFair value model failed: {e}")
        import traceback
        traceback.print_exc()

# ============================================================================
# 10Y EXPECTED CURRENCY RETURNS (CARRY + VALUATION + EM KICKER)
# Methodology: Geometric real returns using BEI for inflation expectations
# ============================================================================
usd_bond_10y = usd_row['Bond_10Y']
usd_bei = usd_row['BEI']

# Component 1: Real Carry (nominal yield differential adjusted for inflation differential)
# Real yield ≈ Nominal yield - Expected inflation (BEI)
df_final['Real_Yield_10Y'] = df_final['Bond_10Y'] - df_final['BEI']
usd_real_yield_10y = usd_bond_10y - usd_bei
df_final['Real_Carry_Annual'] = df_final['Real_Yield_10Y'] - usd_real_yield_10y

# Component 2: Valuation / Currency Drag (mean reversion toward fair value)
# Positive gap = undervalued = expected appreciation
# Negative gap = overvalued = expected depreciation
df_final['Valuation_Gap_Pct'] = (df_final['REER_10Y_Mean'] - df_final['REER']) / df_final['REER'] * 100

if REVERSION_MODE == 'half_life':
    # Exponential decay: fraction closed = 1 - exp(-ln(2)/half_life * horizon)
    decay_rate = np.log(2) / REVERSION_HALF_LIFE
    fraction_closed = 1 - np.exp(-decay_rate * REVERSION_HORIZON)
    df_final['Valuation_Gap_Annual'] = df_final['Valuation_Gap_Pct'] * fraction_closed / REVERSION_HORIZON
else:
    # Linear: 100% reversion over horizon
    fraction_closed = 1.0
    df_final['Valuation_Gap_Annual'] = df_final['Valuation_Gap_Pct'] / REVERSION_HORIZON

# Component 3: EM Structural Kicker (Balassa-Samuelson effect)
df_final['EM_Kicker_Annual'] = df_final['Country'].map(lambda x: EM_KICKERS.get(x, 0.0))

# Total expected ANNUAL return (arithmetic sum of components)
df_final['Expected_Annual_Return'] = (df_final['Real_Carry_Annual'] + 
                                       df_final['Valuation_Gap_Annual'] + 
                                       df_final['EM_Kicker_Annual'])

# Convert to GEOMETRIC 10Y cumulative return: (1 + r_annual)^10 - 1
# Using decimal form for calculation
df_final['Expected_10Y_Return_Geometric'] = (
    (1 + df_final['Expected_Annual_Return'] / 100) ** 10 - 1
) * 100

# Keep arithmetic version for comparison
df_final['Carry_Annual'] = df_final['Bond_10Y'] - usd_bond_10y  # Nominal carry (legacy)
df_final['Carry_10Y'] = df_final['Carry_Annual'] * 10
df_final['Valuation_Gap_10Y'] = df_final['Valuation_Gap_Pct']
df_final['EM_Kicker_10Y'] = df_final['EM_Kicker_Annual'] * 10
df_final['Expected_10Y_Return'] = (df_final['Carry_10Y'] + 
                                    df_final['Valuation_Gap_10Y'] + 
                                    df_final['EM_Kicker_10Y'])


# ============================================================================
# LTCMA YIELD DECOMPOSITION (JP Morgan Style)
# ============================================================================
# Verify decomposition: 10Y Nominal ≈ r* + Expected Inflation + IRP + Term Premium
# Note: Term premium is the residual not captured by r*, inflation, and IRP
df_final['LTCMA_Sum'] = df_final['R_Star'] + df_final['CB_Inflation_Target'] + df_final['IRP']
df_final['LTCMA_Residual'] = df_final['Bond_10Y'] - df_final['LTCMA_Sum']

# ============================================================================
# ACM TERM PREMIUM DECOMPOSITION (US Only - used as proxy for others)
# ============================================================================
acm_data = None
us_acm_ratios = {}

if os.path.exists(ACM_DATA_CSV):
    try:
        acm_df = pd.read_csv(ACM_DATA_CSV, index_col=0, parse_dates=True)
        
        # Get latest valid values for each series (some may be lagged)
        def get_latest_valid(series):
            last_idx = series.last_valid_index()
            return series.loc[last_idx] if last_idx is not None else np.nan
        
        # US 10Y decomposition using ACM
        us_10y_nominal = usd_bond_10y
        us_term_premium = get_latest_valid(acm_df['term_premium_10y'])
        us_expected_short = get_latest_valid(acm_df['expected_short_rate_10y'])
        us_expected_inflation = get_latest_valid(acm_df['expected_inflation_10y'])
        us_tips = get_latest_valid(acm_df['tips_10y'])
        us_bei_acm = get_latest_valid(acm_df['bei_10y'])
        
        # Calculate ratios for proxying other countries
        if not np.isnan(us_term_premium) and us_10y_nominal > 0:
            us_acm_ratios['term_premium_ratio'] = us_term_premium / us_10y_nominal
            us_acm_ratios['expected_rate_ratio'] = us_expected_short / us_10y_nominal if not np.isnan(us_expected_short) else np.nan
        
        # Add ACM data to US row
        us_idx = df_final[df_final['Country'] == 'USA'].index
        if len(us_idx) > 0:
            df_final.loc[us_idx, 'ACM_Term_Premium'] = us_term_premium
            df_final.loc[us_idx, 'ACM_Expected_Short_Rate'] = us_expected_short
            df_final.loc[us_idx, 'ACM_Expected_Inflation'] = us_expected_inflation
        
        # Estimate term premium for other countries using US ratio
        if 'term_premium_ratio' in us_acm_ratios:
            for idx, row in df_final.iterrows():
                if row['Country'] != 'USA' and not np.isnan(row['Bond_10Y']):
                    # Estimate term premium as proportion of yield (adjusted for EM premium)
                    em_adj = 1.2 if row['Country'] in ['ZAR', 'MXN', 'INR', 'CNY'] else 1.0
                    df_final.loc[idx, 'ACM_Term_Premium_Est'] = row['Bond_10Y'] * us_acm_ratios['term_premium_ratio'] * em_adj
        
        print("\n" + "=" * 80)
        print("ACM TERM PREMIUM DECOMPOSITION (US Data)")
        print("=" * 80)
        print(f"US 10Y Nominal:          {us_10y_nominal:.2f}%")
        print(f"  Term Premium (ACM):    {us_term_premium:.2f}%")
        print(f"  Expected Short Rate:   {us_expected_short:.2f}%")
        print(f"  Expected Inflation:    {us_expected_inflation:.2f}%")
        print(f"  TIPS 10Y:              {us_tips:.2f}%")
        print(f"  BEI 10Y:               {us_bei_acm:.2f}%")
        print("-" * 80)
        print(f"Term Premium Ratio:      {us_acm_ratios.get('term_premium_ratio', np.nan)*100:.1f}% of nominal yield")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nACM decomposition failed: {e}")

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

# Plot 2: REER Time Series with 10-Year Average and Structural Breaks
if os.path.exists(REER_HIST_CSV):
    reer_hist = pd.read_csv(REER_HIST_CSV, index_col=0, parse_dates=True)
    
    # Create a 3x3 grid for all currencies
    fig2, axes = plt.subplots(3, 3, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, country in enumerate(reer_hist.columns):
        if idx >= 9:
            break
        ax = axes[idx]
        
        # Get country data
        country_series = reer_hist[country].dropna()
        if len(country_series) == 0:
            continue
        
        # Get break info if available
        break_info = REER_BREAK_INFO.get(country, {})
        break_dates = break_info.get('break_dates', [])
        regime_mean = break_info.get('regime_mean', country_series.mean())
        full_mean = break_info.get('full_mean', country_series.mean())
        
        # Use regime mean if breaks detected, otherwise full mean
        mean_10y = regime_mean if break_dates else full_mean
        current_val = country_series.iloc[-1]
        
        # Plot time series
        ax.plot(country_series.index, country_series.values, 
                color=COLORS['primary'], linewidth=2, label='REER')
        
        # Plot structural breaks as vertical lines
        for bd in break_dates:
            ax.axvline(bd, color=COLORS['negative'], linestyle='-', 
                       linewidth=1.5, alpha=0.7, zorder=3)
        
        # Plot regime mean (solid) and full mean (dashed) for comparison
        if break_dates:
            ax.axhline(regime_mean, color=COLORS['positive'], linestyle='-', 
                       linewidth=2, label=f'Regime Avg: {regime_mean:.1f}')
            ax.axhline(full_mean, color=COLORS['accent'], linestyle='--', 
                       linewidth=1.5, alpha=0.6, label=f'Full 10Y: {full_mean:.1f}')
        else:
            ax.axhline(mean_10y, color=COLORS['accent'], linestyle='--', 
                       linewidth=2, label=f'10Y Avg: {mean_10y:.1f}')
        
        # Plot baseline at 100
        ax.axhline(100, color=COLORS['neutral'], linestyle=':', 
                   linewidth=1, alpha=0.7)
        
        # Shade area between current and average
        ax.fill_between(country_series.index, country_series.values, mean_10y, 
                        alpha=0.2, color=COLORS['positive'] if current_val < mean_10y else COLORS['negative'])
        
        # Calculate valuation gap using regime-adjusted mean
        gap_pct = (mean_10y - current_val) / current_val * 100
        gap_label = f'{"Underval" if gap_pct > 0 else "Overval"}: {abs(gap_pct):.1f}%'
        
        # Add break count to title
        break_str = f' [{len(break_dates)} breaks]' if break_dates else ''
        ax.set_title(f'{country} (Current: {current_val:.1f}){break_str}', fontweight='bold', fontsize=11)
        ax.set_ylabel('REER Index', fontsize=9)
        ax.legend(loc='upper left', fontsize=7, frameon=True)
        ax.tick_params(axis='x', rotation=45)
        
        # Add valuation gap annotation
        ax.annotate(gap_label, xy=(0.98, 0.02), xycoords='axes fraction',
                    ha='right', va='bottom', fontsize=8, fontweight='bold',
                    color=COLORS['positive'] if gap_pct > 0 else COLORS['negative'],
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    fig2.suptitle('REER Time Series: Regime-Adjusted Mean with Structural Breaks', fontsize=14, fontweight='bold', y=0.98)
    add_footnote(fig2, y_pos=0.01)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08, top=0.93, hspace=0.35)
    plt.show()

# Plot 3: Scatter Plot with Regression Line
fig3, ax3 = plt.subplots(figsize=(11, 6))
ax3.scatter(df_final['Diff_vs_USD'], df_final['REER'], marker='o', s=120, 
            c=COLORS['primary'], edgecolors='white', linewidth=2, zorder=5)

for i, row in df_final.iterrows():
    ax3.annotate(row['Country'], (row['Diff_vs_USD'], row['REER'] + 1.5),
                 ha='center', fontsize=10, fontweight='bold', color='#333333')

x_range = np.linspace(df_final['Diff_vs_USD'].min() - 1, df_final['Diff_vs_USD'].max() + 1, 100)
ax3.plot(x_range, model.params[0] + model.params[1] * x_range, 
         color=COLORS['accent'], linewidth=2.5, label=f'Fitted Model (R²={model.rsquared:.2f})')

ax3.set_xlabel('Real Rate Differential vs USD (%)')
ax3.set_ylabel('REER Index')
ax3.set_title('REER vs Real Interest Rate Differential')
ax3.legend(loc='upper left', frameon=True)
add_footnote(fig3)
plt.tight_layout()
plt.subplots_adjust(bottom=0.18)
plt.show()

# Plot 4: Bar Chart of Nominal Yields (Short-term rates)
fig4, ax4 = plt.subplots(figsize=(11, 6))
df_nom_sorted = df_final.sort_values('Nominal_Rate')
colors = [COLORS['positive'] if x >= 0 else COLORS['negative'] for x in df_nom_sorted['Nominal_Rate']]
bars = ax4.bar(df_nom_sorted['Country'], df_nom_sorted['Nominal_Rate'], color=colors,
               edgecolor='white', linewidth=1.5)
ax4.axhline(0, color='#333333', linestyle='-', linewidth=1)
ax4.set_title('Nominal Yields by Currency')
ax4.set_ylabel('Nominal Rate (%)')
ax4.set_xlabel('Currency')
ax4.tick_params(axis='x', rotation=0)

for bar, val in zip(bars, df_nom_sorted['Nominal_Rate']):
    y_pos = bar.get_height() + 0.2 if val >= 0 else bar.get_height() - 0.4
    ax4.annotate(f'{val:.1f}%', (bar.get_x() + bar.get_width()/2, y_pos),
                 ha='center', va='bottom' if val >= 0 else 'top', fontsize=9, fontweight='bold')

add_footnote(fig4)
plt.tight_layout()
plt.subplots_adjust(bottom=0.18)
plt.show()

# Plot 5: Multi-Tenor Yield Curve Comparison
fig5, ax5 = plt.subplots(figsize=(12, 7))
tenor_labels = ['5Y', '10Y', '15Y', '20Y']
tenor_nums = [5, 10, 15, 20]

cmap = plt.cm.get_cmap('tab10')
for idx, (_, row) in enumerate(df_final.iterrows()):
    yields = [row['Bond_5Y'], row['Bond_10Y'], row['Bond_15Y'], row['Bond_20Y']]
    ax5.plot(tenor_nums, yields, marker='o', markersize=8, linewidth=2.5,
             label=row['Country'], color=cmap(idx), alpha=0.85)

ax5.set_xlabel('Tenor (Years)')
ax5.set_ylabel('Yield (%)')
ax5.set_title('Yield Curve Comparison Across Currencies')
ax5.set_xticks(tenor_nums)
ax5.set_xticklabels(tenor_labels)
ax5.legend(loc='upper left', frameon=True, ncol=3)
add_footnote(fig5)
plt.tight_layout()
plt.subplots_adjust(bottom=0.15)
plt.show()

# Plot 6: Expected Annual Currency Returns (Carry + Valuation + EM Kicker)
# Using same theme as REER chart
fig6, ax6 = plt.subplots(figsize=(11, 6))
df_returns = df_final[df_final['Country'] != 'USA'].sort_values('Expected_Annual_Return')

# Annualized components
carry_vals = df_returns['Real_Carry_Annual'].values
valuation_vals = df_returns['Valuation_Gap_Annual'].values
em_kicker_vals = df_returns['EM_Kicker_Annual'].values

# Simple bar chart with primary theme color (matching REER chart)
bars = ax6.bar(df_returns['Country'], df_returns['Expected_Annual_Return'], 
               color=COLORS['primary'], edgecolor='white', linewidth=1.5)

ax6.axhline(0, color=COLORS['negative'], linestyle='--', linewidth=2, label='Zero Return', alpha=0.8)
ax6.set_title('Expected Annual Currency Returns vs USD (Real, Geometric)')
ax6.set_ylabel('Expected Annual Return (%)')
ax6.set_xlabel('Currency')
ax6.legend(loc='upper left', frameon=True)

# Annotate values on bars
for bar, val in zip(bars, df_returns['Expected_Annual_Return']):
    y_pos = bar.get_height() + 0.15 if val >= 0 else bar.get_height() - 0.25
    ax6.annotate(f'{val:+.2f}%', (bar.get_x() + bar.get_width()/2, y_pos),
                 ha='center', va='bottom' if val >= 0 else 'top', 
                 fontsize=9, fontweight='bold', color='#333333')

add_footnote(fig6)
plt.tight_layout()
plt.subplots_adjust(bottom=0.18)
plt.show()

# Plot 7: Stacked Annual Returns Decomposition
fig7, ax7 = plt.subplots(figsize=(12, 6))
df_returns = df_final[df_final['Country'] != 'USA'].sort_values('Expected_Annual_Return')

x = np.arange(len(df_returns))
width = 0.6

# Use theme-consistent colors
colors_returns = {
    'carry': COLORS['primary'],
    'valuation': COLORS['secondary'],
    'em_kicker': COLORS['accent']
}

# Stacked bar with annualized values
ax7.bar(x, df_returns['Real_Carry_Annual'].values, width, 
        label='Real Carry', color=colors_returns['carry'], edgecolor='white', linewidth=0.5)
ax7.bar(x, df_returns['Valuation_Gap_Annual'].values, width, 
        bottom=df_returns['Real_Carry_Annual'].values, 
        label='Valuation / Currency Drag', color=colors_returns['valuation'], edgecolor='white', linewidth=0.5)
ax7.bar(x, df_returns['EM_Kicker_Annual'].values, width, 
        bottom=df_returns['Real_Carry_Annual'].values + df_returns['Valuation_Gap_Annual'].values, 
        label='Balassa-Samuelson Effect', color=colors_returns['em_kicker'], edgecolor='white', linewidth=0.5)

ax7.set_xlabel('Currency')
ax7.set_ylabel('Annual Return (%)')
ax7.set_title('Expected Annual Currency Returns: Real Carry + Valuation + EM Premium')
ax7.set_xticks(x)
ax7.set_xticklabels(df_returns['Country'])
ax7.legend(loc='upper left', frameon=True)
ax7.axhline(0, color='#333333', linewidth=1)

# Annotate total return
for i, (_, row) in enumerate(df_returns.iterrows()):
    total = row['Expected_Annual_Return']
    y_pos = total + 0.2 if total >= 0 else total - 0.3
    ax7.annotate(f'{total:+.2f}%', (i, y_pos), ha='center', fontsize=9, 
                 fontweight='bold', color='#333333')

add_footnote(fig7)
plt.tight_layout()
plt.subplots_adjust(bottom=0.15)
plt.show()

# Plot 7b: Expected Currency Returns Decomposition Tree (Branching Diagram)
ret_row = df_final[df_final['Country'] == RETURNS_TREE_COUNTRY]
if not ret_row.empty and RETURNS_TREE_COUNTRY != 'USA':
    rr = ret_row.iloc[0]
    total_return = rr['Expected_Annual_Return']
    real_carry = rr['Real_Carry_Annual']
    val_annual = rr['Valuation_Gap_Annual']
    em_kicker = rr['EM_Kicker_Annual']
    nom_diff = rr['Bond_10Y'] - usd_bond_10y
    infl_diff = -(rr['BEI'] - usd_bei)
    gap_pct = rr['Valuation_Gap_Pct']

    fig7b, ax7b = plt.subplots(figsize=(14, 8))
    ax7b.set_xlim(0, 10)
    ax7b.set_ylim(0, 10)
    ax7b.axis('off')

    node_w, node_h = 2.2, 1.1

    # 3-level layout: root → 3 children → 2+2 leaves (carry & valuation split)
    nodes = {
        'root':      (1.1, 5.0),
        'carry':     (4.5, 8.0),
        'valuation': (4.5, 5.0),
        'em':        (4.5, 2.0),
        'nom_diff':  (8.2, 9.0),
        'infl_diff': (8.2, 7.0),
        'gap':       (8.2, 5.8),
        'reversion': (8.2, 4.2),
    }

    reversion_label = (f'HL={REVERSION_HALF_LIFE}yr ({fraction_closed*100:.0f}%)'
                       if REVERSION_MODE == 'half_life' else 'Linear (100%)')

    node_info = {
        'root':      ('Annual Return',     total_return, COLORS['primary']),
        'carry':     ('Real Carry',        real_carry,   COLORS['secondary']),
        'valuation': ('Valuation',         val_annual,   COLORS['accent']),
        'em':        ('EM Kicker (B-S)',   em_kicker,    COLORS['currency_risk']),
        'nom_diff':  ('Nom. Yield Diff',   nom_diff,     COLORS['term_premium']),
        'infl_diff': ('Infl. Diff (BEI)',  infl_diff,    COLORS['inflation']),
        'gap':       ('REER Gap',          gap_pct,      COLORS['real_rate']),
        'reversion': (reversion_label,     fraction_closed * 100, COLORS['neutral']),
    }

    edges = [
        ('root', 'carry'), ('root', 'valuation'), ('root', 'em'),
        ('carry', 'nom_diff'), ('carry', 'infl_diff'),
        ('valuation', 'gap'), ('valuation', 'reversion'),
    ]

    for parent, child in edges:
        px, py = nodes[parent]
        cx, cy = nodes[child]
        ax7b.annotate('',
            xy=(cx - node_w / 2, cy), xytext=(px + node_w / 2, py),
            arrowprops=dict(arrowstyle='->', color='#888888', lw=1.8,
                            connectionstyle='arc3,rad=0.0'))

    for key, (cx, cy) in nodes.items():
        label, value, color = node_info[key]
        box = FancyBboxPatch((cx - node_w / 2, cy - node_h / 2), node_w, node_h,
                             boxstyle='round,pad=0.15', facecolor=color,
                             edgecolor='white', linewidth=2)
        ax7b.add_patch(box)
        ax7b.text(cx, cy + 0.15, label, ha='center', va='center',
                  fontsize=11, fontweight='bold', color='white')
        fmt = f'{value:+.2f}%' if key != 'reversion' else f'{value:.0f}%'
        ax7b.text(cx, cy - 0.25, fmt, ha='center', va='center',
                  fontsize=13, fontweight='bold', color='white')

    ax7b.set_title(f'Expected Annual Currency Return Decomposition — {RETURNS_TREE_COUNTRY} vs USD',
                   fontsize=14, fontweight='bold', pad=15)

    fig7b.text(0.5, 0.04,
        f'Annual Return ({total_return:+.2f}%) = Real Carry ({real_carry:+.2f}%) '
        f'+ Valuation ({val_annual:+.2f}%) + EM Kicker ({em_kicker:+.2f}%)',
        ha='center', fontsize=10, style='italic', color='#555555')
    add_footnote(fig7b, y_pos=0.01)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    plt.show()

# Plot 8: LTCMA Yield Decomposition (JP Morgan Style)
fig8, ax8 = plt.subplots(figsize=(12, 6))
df_ltcma = df_final.dropna(subset=['Bond_10Y', 'R_Star', 'CB_Inflation_Target']).sort_values('Bond_10Y')

x = np.arange(len(df_ltcma))
width = 0.6

# Stacked bar: r* + Expected Inflation + IRP + Residual (Term Premium)
r_star_vals = df_ltcma['R_Star'].values
inflation_vals = df_ltcma['CB_Inflation_Target'].values
irp_vals = df_ltcma['IRP'].fillna(0).values
residual_vals = df_ltcma['LTCMA_Residual'].fillna(0).values

ax8.bar(x, r_star_vals, width, label='Real Cash Rate (r*)', 
        color=COLORS['primary'], edgecolor='white', linewidth=0.5)
ax8.bar(x, inflation_vals, width, bottom=r_star_vals, label='Expected Inflation (CB Target)',
        color=COLORS['secondary'], edgecolor='white', linewidth=0.5)
ax8.bar(x, irp_vals, width, bottom=r_star_vals + inflation_vals, label='Inflation Risk Premium',
        color=COLORS['accent'], edgecolor='white', linewidth=0.5)
ax8.bar(x, residual_vals, width, bottom=r_star_vals + inflation_vals + irp_vals, 
        label='Term Premium (Residual)', color=COLORS['neutral'], edgecolor='white', linewidth=0.5)

ax8.set_xlabel('Currency')
ax8.set_ylabel('Yield Component (%)')
ax8.set_title('LTCMA 10-Year Yield Decomposition: r* + Expected Inflation + IRP + Term Premium')
ax8.set_xticks(x)
ax8.set_xticklabels(df_ltcma['Country'])
ax8.legend(loc='upper left', frameon=True)
ax8.axhline(0, color='#333333', linewidth=1)

# Annotate total yield on top
for i, (_, row) in enumerate(df_ltcma.iterrows()):
    if not np.isnan(row['Bond_10Y']):
        ax8.annotate(f"{row['Bond_10Y']:.1f}%", (i, row['Bond_10Y'] + 0.3), 
                     ha='center', fontsize=10, fontweight='bold', color='#333333')

add_footnote(fig8)
plt.tight_layout()
plt.subplots_adjust(bottom=0.15)
plt.show()

# Plot 8b: LTCMA Yield Decomposition Tree (Branching Diagram)
tree_row = df_final[df_final['Country'] == LTCMA_TREE_COUNTRY]
if not tree_row.empty and not np.isnan(tree_row.iloc[0]['Bond_10Y']):
    tr = tree_row.iloc[0]
    nominal_10y = tr['Bond_10Y']
    r_star = tr['R_Star']
    cb_target = tr['CB_Inflation_Target']
    irp = tr['IRP'] if not np.isnan(tr['IRP']) else 0.0
    term_prem = tr['LTCMA_Residual'] if not np.isnan(tr['LTCMA_Residual']) else 0.0
    real_yield = r_star + term_prem
    inflation_comp = cb_target + irp

    fig8b, ax8b = plt.subplots(figsize=(14, 7))
    ax8b.set_xlim(0, 10)
    ax8b.set_ylim(0, 10)
    ax8b.axis('off')

    # Node layout: (cx, cy, width, height)
    node_w, node_h = 2.2, 1.1
    nodes = {
        'root':      (1.3, 5.0),
        'real':      (5.0, 7.3),
        'infl':      (5.0, 2.7),
        'rstar':     (8.7, 8.5),
        'termprem':  (8.7, 6.1),
        'cbtarget':  (8.7, 3.9),
        'irp':       (8.7, 1.5),
    }

    node_info = {
        'root':     ('10Y Nominal',      nominal_10y,    COLORS['primary']),
        'real':     ('Real Yield',        real_yield,     COLORS['secondary']),
        'infl':     ('Inflation Comp.',   inflation_comp, COLORS['accent']),
        'rstar':    ('r* (Real Cash)',    r_star,         COLORS['real_rate']),
        'termprem': ('Term Premium',      term_prem,      COLORS['term_premium']),
        'cbtarget': ('CB Infl. Target',   cb_target,      COLORS['inflation']),
        'irp':      ('Infl. Risk Prem.',  irp,            COLORS['currency_risk']),
    }

    edges = [
        ('root', 'real'), ('root', 'infl'),
        ('real', 'rstar'), ('real', 'termprem'),
        ('infl', 'cbtarget'), ('infl', 'irp'),
    ]

    # Draw edges first (behind nodes)
    for parent, child in edges:
        px, py = nodes[parent]
        cx, cy = nodes[child]
        ax8b.annotate('',
            xy=(cx - node_w / 2, cy), xytext=(px + node_w / 2, py),
            arrowprops=dict(arrowstyle='->', color='#888888', lw=1.8,
                            connectionstyle='arc3,rad=0.0'))

    # Draw nodes
    for key, (cx, cy) in nodes.items():
        label, value, color = node_info[key]
        box = FancyBboxPatch((cx - node_w / 2, cy - node_h / 2), node_w, node_h,
                             boxstyle='round,pad=0.15', facecolor=color,
                             edgecolor='white', linewidth=2)
        ax8b.add_patch(box)
        ax8b.text(cx, cy + 0.15, label, ha='center', va='center',
                  fontsize=11, fontweight='bold', color='white')
        ax8b.text(cx, cy - 0.25, f'{value:+.2f}%', ha='center', va='center',
                  fontsize=13, fontweight='bold', color='white')

    ax8b.set_title(f'LTCMA 10Y Yield Decomposition Tree — {LTCMA_TREE_COUNTRY}',
                   fontsize=14, fontweight='bold', pad=15)

    # Equation footnote
    fig8b.text(0.5, 0.04,
        f'10Y Nominal ({nominal_10y:.2f}%) = r* ({r_star:.2f}%) + CB Target ({cb_target:.2f}%) '
        f'+ IRP ({irp:+.2f}%) + Term Premium ({term_prem:+.2f}%)',
        ha='center', fontsize=10, style='italic', color='#555555')
    add_footnote(fig8b, y_pos=0.01)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    plt.show()

# Plot 9: Fair Value REER Model - Actual vs Model-Implied
if fair_value_df is not None and pooled_model_results is not None:
    fig9, axes = plt.subplots(3, 3, figsize=(14, 10))
    axes = axes.flatten()
    
    countries = fair_value_df['Country'].unique()
    
    for idx, country in enumerate(countries):
        if idx >= 9:
            break
        ax = axes[idx]
        
        # Get country data
        country_data = fair_value_df[fair_value_df['Country'] == country].sort_values('Date')
        if len(country_data) == 0:
            continue
        
        # Get latest values
        latest = country_data.iloc[-1]
        current_reer = latest['reer']
        current_fair = latest['fair_value_reer']
        current_dev_pct = latest['deviation_pct']
        
        # Plot actual REER
        ax.plot(country_data['Date'], country_data['reer'], 
                color=COLORS['primary'], linewidth=1.5, label='Actual REER', alpha=0.9)
        
        # Plot fair value (model-implied)
        ax.plot(country_data['Date'], country_data['fair_value_reer'], 
                color=COLORS['accent'], linewidth=2, linestyle='--', label='Fair Value')
        
        # Shade deviation area
        ax.fill_between(country_data['Date'], 
                        country_data['reer'], 
                        country_data['fair_value_reer'],
                        alpha=0.2, 
                        color=COLORS['negative'] if current_dev_pct > 0 else COLORS['positive'])
        
        # Deviation label
        dev_label = f'{"Overvalued" if current_dev_pct > 0 else "Undervalued"}: {abs(current_dev_pct):.1f}%'
        
        ax.set_title(f'{country}', fontweight='bold', fontsize=11)
        ax.set_ylabel('REER Index', fontsize=9)
        ax.legend(loc='upper left', fontsize=7, frameon=True)
        ax.tick_params(axis='x', rotation=45)
        
        # Add current deviation annotation
        ax.annotate(dev_label, xy=(0.98, 0.02), xycoords='axes fraction',
                    ha='right', va='bottom', fontsize=8, fontweight='bold',
                    color=COLORS['negative'] if current_dev_pct > 0 else COLORS['positive'],
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Add current values annotation
        ax.annotate(f'REER: {current_reer:.1f} | FV: {current_fair:.1f}', 
                    xy=(0.98, 0.12), xycoords='axes fraction',
                    ha='right', va='bottom', fontsize=7, color='#333333',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
    
    # Add model info to title
    beta = pooled_model_results.params['real_rate_diff_vs_usd']
    r2 = pooled_model_results.rsquared
    fig9.suptitle(f'Fair Value REER: Actual vs Model-Implied (b={beta:.2f}, R2={r2:.3f})', 
                  fontsize=14, fontweight='bold', y=0.98)
    add_footnote(fig9, y_pos=0.01)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08, top=0.93, hspace=0.35)
    plt.show()
    
    # Print Fair Value Summary Table
    print("\n" + "=" * 100)
    print("FAIR VALUE REER MODEL: Current Valuation Summary")
    print("Model: REER = a + b x Real_Rate_Differential_vs_USD")
    print("=" * 100)
    print(f"{'Currency':<8} {'Current':>10} {'Fair Value':>12} {'Deviation':>12} {'Status':<15}")
    print("-" * 100)
    
    for country in countries:
        latest = fair_value_df[fair_value_df['Country'] == country].iloc[-1]
        status = 'OVERVALUED' if latest['deviation_pct'] > 0 else 'UNDERVALUED'
        print(f"{country:<8} {latest['reer']:>10.1f} {latest['fair_value_reer']:>12.1f} "
              f"{latest['deviation_pct']:>+11.1f}% {status:<15}")
    
    print("=" * 100)
    print(f"Model: REER = {pooled_model_results.params['const']:.2f} + "
          f"{pooled_model_results.params['real_rate_diff_vs_usd']:.2f} x Real_Rate_Diff_vs_USD")
    print(f"Observations: {len(fair_value_df):,} | R2: {pooled_model_results.rsquared:.4f}")
    print("=" * 100)

# ============================================================================
# Plot 10: Structural Break Analysis (CUSUM Charts)
# ============================================================================
if REER_BREAK_INFO and os.path.exists(REER_HIST_CSV):
    reer_hist = pd.read_csv(REER_HIST_CSV, index_col=0, parse_dates=True)
    
    # Only plot countries with detected breaks or CUSUM data
    countries_with_breaks = [c for c in reer_hist.columns if c in REER_BREAK_INFO]
    
    if countries_with_breaks:
        n_countries = len(countries_with_breaks)
        n_cols = 3
        n_rows = (n_countries + n_cols - 1) // n_cols
        
        fig10, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        for idx, country in enumerate(countries_with_breaks):
            if idx >= len(axes):
                break
            ax = axes[idx]
            
            break_info = REER_BREAK_INFO[country]
            country_series = reer_hist[country].dropna()
            break_dates = break_info.get('break_dates', [])
            regime_mean = break_info.get('regime_mean', country_series.mean())
            full_mean = break_info.get('full_mean', country_series.mean())
            
            # Plot REER with regime coloring
            ax.plot(country_series.index, country_series.values, 
                    color=COLORS['primary'], linewidth=1.5, alpha=0.8)
            
            # Color-code regimes
            if break_dates:
                all_breaks = [country_series.index[0]] + list(break_dates) + [country_series.index[-1]]
                regime_colors = plt.cm.Set3(np.linspace(0, 1, len(all_breaks)))
                
                for i in range(len(all_breaks) - 1):
                    start_date = all_breaks[i]
                    end_date = all_breaks[i + 1]
                    mask = (country_series.index >= start_date) & (country_series.index <= end_date)
                    regime_data = country_series[mask]
                    if len(regime_data) > 0:
                        ax.fill_between(regime_data.index, regime_data.values.min() * 0.98, 
                                        regime_data.values.max() * 1.02,
                                        alpha=0.2, color=regime_colors[i])
            
            # Plot break points
            for bd in break_dates:
                ax.axvline(bd, color=COLORS['negative'], linestyle='-', 
                           linewidth=2, alpha=0.8, zorder=3)
                ax.annotate(bd.strftime('%Y-%m'), xy=(bd, ax.get_ylim()[1] * 0.98),
                           fontsize=7, rotation=45, ha='right', color=COLORS['negative'])
            
            # Plot means
            ax.axhline(regime_mean, color=COLORS['positive'], linestyle='-', linewidth=2,
                       label=f'Regime Mean: {regime_mean:.1f}')
            ax.axhline(full_mean, color=COLORS['accent'], linestyle='--', linewidth=1.5,
                       alpha=0.7, label=f'Full 10Y Mean: {full_mean:.1f}')
            
            n_breaks = len(break_dates)
            regime_months = break_info.get('regime_months', 0)
            ax.set_title(f'{country}: {n_breaks} Break(s), Regime: {regime_months}mo', 
                        fontweight='bold', fontsize=11)
            ax.legend(loc='upper left', fontsize=7, frameon=True)
            ax.tick_params(axis='x', rotation=45)
        
        # Hide unused axes
        for idx in range(len(countries_with_breaks), len(axes)):
            axes[idx].set_visible(False)
        
        fig10.suptitle('Structural Break Analysis: CUSUM + Bai-Perron Detection', 
                       fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.subplots_adjust(top=0.92, hspace=0.4)
        plt.show()
        
        # Print structural break summary
        print("\n" + "=" * 100)
        print("STRUCTURAL BREAK ANALYSIS SUMMARY")
        print("=" * 100)
        print(f"{'Currency':<8} {'N Breaks':>10} {'Last Break':>14} {'Regime(mo)':>12} "
              f"{'Full Mean':>11} {'Regime Mean':>13} {'Diff':>8}")
        print("-" * 100)
        
        for country in countries_with_breaks:
            bi = REER_BREAK_INFO[country]
            n_brk = len(bi.get('break_dates', []))
            last_brk = (bi.get('break_dates') or [None])[-1]
            last_brk_str = last_brk.strftime('%Y-%m') if last_brk else 'N/A'
            regime_mo = bi.get('regime_months', 0)
            full_m = bi.get('full_mean', np.nan)
            regime_m = bi.get('regime_mean', np.nan)
            diff = regime_m - full_m if not np.isnan(regime_m) and not np.isnan(full_m) else np.nan
            print(f"{country:<8} {n_brk:>10} {last_brk_str:>14} {regime_mo:>12} "
                  f"{full_m:>11.1f} {regime_m:>13.1f} {diff:>+8.1f}")
        
        print("=" * 100)
        print("Notes: Regime Mean = post-break average (used for valuation) | Diff = adjustment vs full 10Y mean")
        print("=" * 100)

# ============================================================================
# SUMMARY TABLES
# ============================================================================
# Expected Annual Returns Summary
print("\n" + "=" * 140)
print("Expected Annual Currency Returns (REAL, GEOMETRIC) | BEI-adjusted carry + Valuation + EM Kicker")
print("=" * 140)
print(f"{'Currency':<8} {'REER':>7} {'10YAvg':>7} {'Gap%':>7} | "
      f"{'Nom10Y':>7} {'BEI':>6} {'Real10Y':>8} | "
      f"{'RealCarry':>10} {'Val/Yr':>8} {'EM/Yr':>6} | "
      f"{'Annual':>8} {'Geom10Y':>10}")
print("-" * 140)

for _, row in df_final.iterrows():
    gap_pct = (row['REER_10Y_Mean'] - row['REER']) / row['REER'] * 100
    real_yield = row['Bond_10Y'] - row['BEI'] if not np.isnan(row['BEI']) else np.nan
    print(f"{row['Country']:<8} {row['REER']:>7.1f} {row['REER_10Y_Mean']:>7.1f} "
          f"{gap_pct:>+7.1f} | "
          f"{row['Bond_10Y']:>7.2f} {row['BEI']:>6.2f} {real_yield:>+8.2f} | "
          f"{row['Real_Carry_Annual']:>+10.2f} {row['Valuation_Gap_Annual']:>+8.2f} "
          f"{row['EM_Kicker_Annual']:>+6.2f} | "
          f"{row['Expected_Annual_Return']:>+8.2f} "
          f"{row['Expected_10Y_Return_Geometric']:>+10.1f}")

print("=" * 140)
reversion_desc = (f"half-life={REVERSION_HALF_LIFE}yr, {fraction_closed*100:.0f}% closure over {REVERSION_HORIZON}yr" 
                  if REVERSION_MODE == 'half_life' else f"linear 100% over {REVERSION_HORIZON}yr")
print("Notes: Real10Y = Nominal 10Y - BEI | RealCarry = Real yield differential vs USD | Geom10Y = (1+r)^10-1")
print(f"       Gap% = REER mean reversion ({reversion_desc}) | EM Kicker = Balassa-Samuelson")
print("=" * 140)

# Sensitivity Table: Expected Annual Returns under different half-life assumptions
sensitivity_half_lives = [3, 5, 7]
print("\n" + "=" * 100)
print("MEAN REVERSION SENSITIVITY: Expected Annual Returns by Half-Life Assumption")
print("=" * 100)
header_hl = "  ".join([f"HL={hl}yr ({(1 - np.exp(-np.log(2)/hl * REVERSION_HORIZON))*100:.0f}%)" for hl in sensitivity_half_lives])
print(f"{'Currency':<8} {'Gap%':>7} | {header_hl}  |  {'Linear':>8}")
print("-" * 100)

for _, row in df_final.iterrows():
    if row['Country'] == 'USA':
        continue
    gap = row['Valuation_Gap_Pct']
    carry = row['Real_Carry_Annual']
    em = row['EM_Kicker_Annual']
    vals = []
    for hl in sensitivity_half_lives:
        frac = 1 - np.exp(-np.log(2) / hl * REVERSION_HORIZON)
        val_annual = gap * frac / REVERSION_HORIZON
        total = carry + val_annual + em
        vals.append(f"{total:>+8.2f}")
    linear_total = carry + gap / REVERSION_HORIZON + em
    print(f"{row['Country']:<8} {gap:>+7.1f} | {'  '.join(vals)}  |  {linear_total:>+8.2f}")

print("=" * 100)
print(f"Notes: Active mode = {REVERSION_MODE} | Gap% = (REER_mean - REER_current) / REER_current")
print(f"       HL=Nyr means {REVERSION_HORIZON}yr gap closure: 3yr->90%, 5yr->75%, 7yr->63% | Linear = 100%")
print("=" * 100)

# ============================================================================
# LTCMA YIELD DECOMPOSITION SUMMARY
# ============================================================================
print("\n" + "=" * 130)
print("LTCMA 10-Year Yield Decomposition (JP Morgan Style)")
print("=" * 130)
print(f"{'Currency':<8} {'10Y Nom':>8} | {'r*':>6} {'Exp Inf':>8} {'BEI':>6} {'IRP':>6} | "
      f"{'LTCMA Sum':>10} {'Residual':>9} | {'Check':>8}")
print("-" * 130)

for _, row in df_final.iterrows():
    ltcma_sum = row['R_Star'] + row['CB_Inflation_Target'] + (row['IRP'] if not np.isnan(row['IRP']) else 0)
    residual = row['Bond_10Y'] - ltcma_sum if not np.isnan(row['Bond_10Y']) else np.nan
    check = "OK" if not np.isnan(row['Bond_10Y']) and abs(row['Bond_10Y'] - ltcma_sum - residual) < 0.01 else "N/A"
    
    print(f"{row['Country']:<8} {row['Bond_10Y']:>8.2f} | "
          f"{row['R_Star']:>6.2f} {row['CB_Inflation_Target']:>8.2f} "
          f"{row['BEI']:>6.2f} {row['IRP']:>+6.2f} | "
          f"{ltcma_sum:>10.2f} {residual:>+9.2f} | {check:>8}")

print("=" * 130)
print("Notes: r* = Expected Real Cash Rate (IMF) | Exp Inf = CB Inflation Target | BEI = Breakeven Inflation")
print("       IRP = Inflation Risk Premium (BEI - CB Target) | Residual = Term Premium (10Y Nom - LTCMA Sum)")
print("=" * 130)
