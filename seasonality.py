############################################################
# • Dynamic seasonal return chart using matplotlib.
# • Green/red bars and black diamonds.
############################################################

import datetime as dt
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from matplotlib.ticker import MultipleLocator, PercentFormatter

try:
    from pandas_datareader import data as pdr
except ImportError:
    pdr = None

FALLBACK_MAP = {
    '^GSPC': 'SP500',
    '^DJI':  'DJIA',
    '^IXIC': 'NASDAQCOM',
}

MONTH_LABELS = [
    'Jan','Feb','Mar','Apr','May','Jun',
    'Jul','Aug','Sep','Oct','Nov','Dec',
]

st.set_page_config(page_title="Seasonality Dashboard", layout="wide")
st.title("📈 Monthly Seasonality Analysis")


def seasonal_stats(prices: pd.Series) -> pd.DataFrame:
    monthly = prices.resample('ME').last().pct_change().dropna()
    monthly.index = monthly.index.to_period('M')
    grouped = monthly.groupby(monthly.index.month)
    median_ret = grouped.median() * 100
    hit_rate = grouped.apply(lambda x: x.gt(0).mean() * 100)
    counts = grouped.size()

    idx = pd.Index(range(1,13), name='month')
    stats = pd.DataFrame(index=idx)
    stats['median_ret'] = median_ret
    stats['hit_rate'] = hit_rate
    stats['count'] = counts
    stats['label'] = MONTH_LABELS
    return stats


def plot_seasonality(stats: pd.DataFrame, title: str) -> None:
    plot_df = stats.dropna(subset=['median_ret','hit_rate'], how='all')
    labels = plot_df['label'].tolist()
    median = plot_df['median_ret'].to_numpy(dtype=float)
    hit = plot_df['hit_rate'].to_numpy(dtype=float)

    med_min, med_max = np.nanmin(median), np.nanmax(median)
    y1_bottom = min(0.0, med_min - 1.0)
    y1_top    = med_max + 1.0

    hr_min, hr_max = np.nanmin(hit), np.nanmax(hit)
    y2_bottom = max(0.0, hr_min - 5.0)
    y2_top    = min(100.0, hr_max + 5.0)

    fig, ax1 = plt.subplots(figsize=(10,6))
    ax2 = ax1.twinx()

    bar_colors = ['mediumseagreen' if val >= 0 else 'indianred' for val in median]
    edge_colors = ['darkgreen' if val >= 0 else 'darkred' for val in median]
    ax1.bar(
        labels, median, width=0.8,
        color=bar_colors, edgecolor=edge_colors, linewidth=1.2,
        zorder=2
    )
    ax1.set_ylabel('Median return', weight='bold')
    ax1.yaxis.set_major_formatter(PercentFormatter())
    ax1.yaxis.set_major_locator(MultipleLocator(1))
    ax1.set_ylim(bottom=y1_bottom, top=y1_top)
    ax1.set_axisbelow(True)
    ax1.grid(axis='y', linestyle='--', linewidth=0.5, color='lightgrey', alpha=0.7, zorder=1)

    ax2.scatter(
        labels, hit, marker='D', s=80,
        facecolors='black', edgecolors='black', linewidths=0.8,
        zorder=3
    )
    ax2.set_ylabel('Hit rate of positive returns', weight='bold')
    ax2.yaxis.set_major_formatter(PercentFormatter())
    ax2.yaxis.set_major_locator(MultipleLocator(5))
    ax2.set_ylim(bottom=y2_bottom, top=y2_top)

    fig.suptitle(title, fontsize=14, weight='bold')
    fig.tight_layout(pad=2)
    st.pyplot(fig)


# ── Streamlit UI ─────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    symbol = st.text_input("Ticker symbol", value="^GSPC")
with col2:
    start_year = st.number_input("Start year", value=1950, min_value=1900, max_value=dt.datetime.today().year)

start_date = f"{start_year}-01-01"

# ── Data Fetch ───────────────────────────────────────────
warnings.filterwarnings('ignore', category=FutureWarning, module='yfinance')
try:
    start_dt = pd.to_datetime(start_date)
    sym_up = symbol.upper()

    if pdr and sym_up in FALLBACK_MAP and start_dt.year < 1950:
        fred_key = FALLBACK_MAP[sym_up]
        st.info(f"Using FRED fallback: {fred_key} from {start_date}")
        df_fred = pdr.DataReader(fred_key, 'fred', start_dt, dt.date.today())
        prices = df_fred[fred_key].rename('Close')
    else:
        df = yf.download(symbol, start=start_date, auto_adjust=True, progress=False)
        if df.empty:
            st.error(f"No data found for '{symbol}'")
            st.stop()
        prices = df['Close']

    stats = seasonal_stats(prices)
    first_year = prices.index[0].year
    plot_seasonality(stats, f"{symbol} seasonality (since {first_year})")

    st.markdown("### Monthly Stats Table")
    stats_display = stats[['label', 'median_ret', 'hit_rate', 'count']].copy()
    stats_display.columns = ['Month', 'Median Return (%)', 'Hit Rate (%)', 'Years Observed']
    st.dataframe(stats_display.set_index('Month').style.format("{:.2f}"))

except Exception as e:
    st.error(f"Error: {e}")
