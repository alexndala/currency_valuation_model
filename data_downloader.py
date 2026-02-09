"""
FRED Data Downloader for Currency Valuation Model
Downloads all required FRED series and saves locally for offline analysis.
Run monthly to refresh data.

Supports:
- Latest snapshot data (for quick analysis)
- Full panel dataset from April 1994 (for regression analysis)
"""
import pandas as pd
import pandas_datareader.data as web
from datetime import datetime
import os
import json
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# DATA CONFIGURATION (same as reer.py)
# ============================================================================
DATA_MAP = {
    'USA': {
        'reer': 'RBUSBIS', 'rate': 'FEDFUNDS', 'cpi': 'CPALTT01USM659N',
        'bond5y': 'DGS5', 'bond10y': 'DGS10', 'bond20y': 'DGS20', 'bond30y': 'DGS30',
        'tips10y': 'DFII10'
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

# US ACM Term Premium series (for yield decomposition)
ACM_SERIES = {
    'term_premium_10y': 'THREEFYTP10',
    'term_premium_5y': 'THREEFYTP5', 
    'expected_short_rate_10y': 'THREEFF10',
    'tips_10y': 'DFII10',
    'bei_10y': 'T10YIE',
    'expected_inflation_10y': 'EXPINF10YR',
}

# Panel data configuration
PANEL_START = datetime(1994, 4, 1)  # April 1994

# Output paths
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
MACRO_DATA_CSV = os.path.join(DATA_DIR, 'fred_macro_data.csv')
MACRO_DATA_XLSX = os.path.join(DATA_DIR, 'fred_macro_data.xlsx')
REER_HIST_CSV = os.path.join(DATA_DIR, 'reer_historical.csv')
REER_HIST_XLSX = os.path.join(DATA_DIR, 'reer_historical.xlsx')
PANEL_DATA_CSV = os.path.join(DATA_DIR, 'panel_data.csv')
PANEL_DATA_XLSX = os.path.join(DATA_DIR, 'panel_data.xlsx')
ACM_DATA_CSV = os.path.join(DATA_DIR, 'acm_decomposition.csv')
METADATA_FILE = os.path.join(DATA_DIR, 'metadata.json')


def ensure_data_dir():
    """Create data directory if it doesn't exist."""
    os.makedirs(DATA_DIR, exist_ok=True)


def fetch_series(series_id, start, end, retries=3):
    """Fetch a single FRED series with retry logic."""
    for attempt in range(retries):
        try:
            df = web.DataReader(series_id, 'fred', start, end)
            return df
        except Exception as e:
            if attempt == retries - 1:
                print(f"    WARNING: Failed to fetch {series_id} after {retries} attempts: {e}")
                return None
    return None


def download_macro_data():
    """Download current values for all macro indicators."""
    print("=" * 70)
    print("Downloading Macro Data from FRED")
    print("=" * 70)
    
    end = datetime.now()
    start = datetime(2022, 1, 1)  # Recent data for current values
    
    results = []
    
    for country, ids in DATA_MAP.items():
        print(f"\n{country}:")
        row = {'Country': country}
        
        for indicator, series_id in ids.items():
            print(f"  Fetching {indicator} ({series_id})...", end=" ")
            df = fetch_series(series_id, start, end)
            
            if df is not None and len(df) > 0:
                latest_value = df.iloc[-1, 0]
                latest_date = df.index[-1].strftime('%Y-%m-%d')
                row[f'{indicator}_value'] = latest_value
                row[f'{indicator}_date'] = latest_date
                row[f'{indicator}_series'] = series_id
                print(f"OK ({latest_value:.2f} as of {latest_date})")
            else:
                row[f'{indicator}_value'] = None
                row[f'{indicator}_date'] = None
                row[f'{indicator}_series'] = series_id
                print("FAILED")
        
        results.append(row)
    
    return pd.DataFrame(results)


def download_reer_historical():
    """Download 10-year historical REER data for all countries."""
    print("\n" + "=" * 70)
    print("Downloading 10-Year REER Historical Data")
    print("=" * 70)
    
    end = datetime.now()
    start = datetime(end.year - 10, end.month, end.day)
    
    reer_data = {}
    
    for country, ids in DATA_MAP.items():
        reer_series = ids.get('reer')
        if reer_series:
            print(f"  Fetching {country} REER ({reer_series})...", end=" ")
            df = fetch_series(reer_series, start, end)
            
            if df is not None and len(df) > 0:
                reer_data[country] = df.iloc[:, 0]
                print(f"OK ({len(df)} observations)")
            else:
                print("FAILED")
    
    if reer_data:
        return pd.DataFrame(reer_data)
    return pd.DataFrame()


def download_panel_data():
    """Download full panel dataset from April 1994 to present."""
    print("\n" + "=" * 70)
    print("Downloading Panel Data (April 1994 - Present)")
    print("=" * 70)
    
    end = datetime.now()
    start = PANEL_START
    
    all_data = []
    
    for country, ids in DATA_MAP.items():
        print(f"\n{country}:")
        country_data = {}
        
        # Core indicators for panel regression
        core_indicators = ['reer', 'rate', 'cpi', 'bond10y']
        
        for indicator in core_indicators:
            if indicator not in ids:
                continue
            series_id = ids[indicator]
            print(f"  Fetching {indicator} ({series_id})...", end=" ")
            
            df = fetch_series(series_id, start, end)
            
            if df is not None and len(df) > 0:
                # Resample to monthly frequency (end of month)
                df_monthly = df.resample('ME').last()
                country_data[indicator] = df_monthly.iloc[:, 0]
                start_date = df_monthly.index[0].strftime('%Y-%m')
                end_date = df_monthly.index[-1].strftime('%Y-%m')
                print(f"OK ({len(df_monthly)} months, {start_date} to {end_date})")
            else:
                print("FAILED")
        
        # Combine into a DataFrame for this country
        if country_data:
            df_country = pd.DataFrame(country_data)
            df_country['Country'] = country
            df_country.index.name = 'Date'
            all_data.append(df_country.reset_index())
    
    # Combine all countries
    if all_data:
        panel_df = pd.concat(all_data, ignore_index=True)
        
        # Reorder columns
        cols = ['Country', 'Date', 'reer', 'rate', 'cpi', 'bond10y']
        cols = [c for c in cols if c in panel_df.columns]
        panel_df = panel_df[cols]
        
        return panel_df
    
    return pd.DataFrame()


def download_acm_data():
    """Download US ACM term premium decomposition data."""
    print("\n" + "=" * 70)
    print("Downloading US ACM Term Premium Data")
    print("=" * 70)
    
    end = datetime.now()
    start = PANEL_START
    
    acm_data = {}
    
    for name, series_id in ACM_SERIES.items():
        print(f"  Fetching {name} ({series_id})...", end=" ")
        df = fetch_series(series_id, start, end)
        
        if df is not None and len(df) > 0:
            # Resample to monthly
            df_monthly = df.resample('ME').last()
            acm_data[name] = df_monthly.iloc[:, 0]
            start_date = df_monthly.index[0].strftime('%Y-%m')
            print(f"OK ({len(df_monthly)} months from {start_date})")
        else:
            print("FAILED or not available")
    
    if acm_data:
        return pd.DataFrame(acm_data)
    return pd.DataFrame()


def save_data(macro_df, reer_df, panel_df=None, acm_df=None):
    """Save data to CSV and XLSX formats."""
    print("\n" + "=" * 70)
    print("Saving Data Files")
    print("=" * 70)
    
    ensure_data_dir()
    
    # Save macro data (latest snapshot)
    if not macro_df.empty:
        macro_df.to_csv(MACRO_DATA_CSV, index=False)
        macro_df.to_excel(MACRO_DATA_XLSX, index=False, engine='openpyxl')
        print(f"  Macro data saved to:")
        print(f"    - {MACRO_DATA_CSV}")
        print(f"    - {MACRO_DATA_XLSX}")
    
    # Save REER historical data (10-year)
    if not reer_df.empty:
        reer_df.to_csv(REER_HIST_CSV)
        reer_df.to_excel(REER_HIST_XLSX, engine='openpyxl')
        print(f"  REER historical data saved to:")
        print(f"    - {REER_HIST_CSV}")
        print(f"    - {REER_HIST_XLSX}")
    
    # Save panel data (full history from 1994)
    if panel_df is not None and not panel_df.empty:
        panel_df.to_csv(PANEL_DATA_CSV, index=False)
        panel_df.to_excel(PANEL_DATA_XLSX, index=False, engine='openpyxl')
        print(f"  Panel data saved to:")
        print(f"    - {PANEL_DATA_CSV}")
        print(f"    - {PANEL_DATA_XLSX}")
        
        # Print panel summary
        countries = panel_df['Country'].nunique()
        dates = panel_df['Date'].nunique()
        obs = len(panel_df)
        print(f"    ({countries} countries, {dates} months, {obs} observations)")
    
    # Save ACM decomposition data
    if acm_df is not None and not acm_df.empty:
        acm_df.to_csv(ACM_DATA_CSV)
        print(f"  ACM decomposition data saved to:")
        print(f"    - {ACM_DATA_CSV}")
    
    # Save metadata
    metadata = {
        'last_updated': datetime.now().isoformat(),
        'panel_start': PANEL_START.isoformat(),
        'countries': list(DATA_MAP.keys()),
        'files': {
            'macro_data': 'fred_macro_data.csv',
            'reer_historical': 'reer_historical.csv',
            'panel_data': 'panel_data.csv',
            'acm_decomposition': 'acm_decomposition.csv'
        }
    }
    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata saved to: {METADATA_FILE}")


def main():
    """Main entry point for data download."""
    print("\n" + "=" * 70)
    print("FRED Data Downloader - Currency Valuation Model")
    print(f"Run Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Panel Start: {PANEL_START.strftime('%Y-%m-%d')}")
    print("=" * 70)
    
    # Download all data
    macro_df = download_macro_data()
    reer_df = download_reer_historical()
    panel_df = download_panel_data()
    acm_df = download_acm_data()
    
    # Save to files
    save_data(macro_df, reer_df, panel_df, acm_df)
    
    print("\n" + "=" * 70)
    print("Download Complete!")
    print("=" * 70)
    print("\nData files created:")
    print("  - fred_macro_data.csv/xlsx    (latest snapshot)")
    print("  - reer_historical.csv/xlsx    (10-year REER)")
    print("  - panel_data.csv/xlsx         (full panel from 1994)")
    print("  - acm_decomposition.csv       (US term premium)")
    print("\nNext steps:")
    print("  1. Run 'python reer.py' to analyze using local data")
    print("  2. Re-run this script monthly to refresh data")
    print("=" * 70)


if __name__ == '__main__':
    main()
