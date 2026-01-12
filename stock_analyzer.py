"""
Tadawul Stock Analysis Tool - Fixed for MultiIndex columns
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from scipy import stats
import warnings
import os

warnings.filterwarnings('ignore')

# For better visualization
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# ====================== CORE FUNCTIONS ======================

def load_stock_data(ticker, period='1y', start_date=None, end_date=None):
    """Load stock data from Yahoo Finance - FIXED VERSION"""
    try:
        if start_date and end_date:
            stock = yf.download(ticker, start=start_date, end=end_date)
        else:
            stock = yf.download(ticker, period=period)

        if stock.empty:
            print(f"⚠️ No data available for {ticker}")
            return None

        print(f"✅ Successfully loaded {ticker}")
        print(f"   Period: {stock.index[0].date()} to {stock.index[-1].date()}")
        print(f"   Days: {len(stock)}")

        # Debug: Check columns structure
        print(f"   Columns type: {type(stock.columns)}")

        # Handle MultiIndex columns
        if isinstance(stock.columns, pd.MultiIndex):
            print("   ⚠️ MultiIndex columns detected - flattening...")
            # Flatten MultiIndex to simple columns
            stock.columns = ['_'.join(col).strip() for col in stock.columns.values]
            print(f"   New columns: {stock.columns.tolist()}")

        return stock

    except Exception as e:
        print(f"❌ Error loading {ticker}: {e}")
        return None


def normalize_column_names(df):
    """Normalize column names for Saudi stocks - FIXED VERSION"""
    data = df.copy()

    print(f"   Input columns: {data.columns.tolist()}")

    # Map for renaming flattened columns to standard names
    column_mapping = {}

    for col in data.columns:
        # Check for Close column (any variant)
        if 'close' in col.lower() and 'adj' not in col.lower():
            column_mapping[col] = 'Adj_Close'  # We'll use Close as Adj_Close
            print(f"   ✓ Using '{col}' as 'Adj_Close'")

        # Map other columns
        elif 'open' in col.lower():
            column_mapping[col] = 'Open'
        elif 'high' in col.lower():
            column_mapping[col] = 'High'
        elif 'low' in col.lower():
            column_mapping[col] = 'Low'
        elif 'volume' in col.lower():
            column_mapping[col] = 'Volume'

    # Apply the mapping
    data.rename(columns=column_mapping, inplace=True)

    # If we still don't have Adj_Close, try to find Close
    if 'Adj_Close' not in data.columns and 'Close' in data.columns:
        data['Adj_Close'] = data['Close']
        print(f"   ✓ Using 'Close' as 'Adj_Close'")
    elif 'Adj_Close' not in data.columns:
        # Create Adj_Close from first available price column
        for price_col in ['Close', 'Open', 'High', 'Low']:
            if price_col in data.columns:
                data['Adj_Close'] = data[price_col]
                print(f"   ✓ Using '{price_col}' as 'Adj_Close'")
                break

    print(f"   Final columns: {data.columns.tolist()}")
    return data


def calculate_technical_indicators(df):
    """Calculate technical indicators - FIXED VERSION"""
    # First normalize column names
    data = normalize_column_names(df)

    # Check if we have the necessary column
    if 'Adj_Close' not in data.columns:
        print("❌ Critical Error: 'Adj_Close' column not created!")
        print(f"   Available columns: {data.columns.tolist()}")
        # Try to use Close as fallback
        if 'Close' in data.columns:
            data['Adj_Close'] = data['Close']
            print("   ⚠️ Using 'Close' as fallback for 'Adj_Close'")
        else:
            print("   ❌ No price column found!")
            return data

    print(f"   Using 'Adj_Close' for calculations")

    # 1. Daily Returns
    data['Daily_Return'] = data['Adj_Close'].pct_change() * 100

    # 2. Moving Averages (only if we have enough data)
    min_data_points = max(200, len(data) // 2)
    if len(data) >= 20:
        data['MA_20'] = data['Adj_Close'].rolling(window=min(20, len(data))).mean()
    if len(data) >= 50:
        data['MA_50'] = data['Adj_Close'].rolling(window=min(50, len(data))).mean()
    if len(data) >= 200:
        data['MA_200'] = data['Adj_Close'].rolling(window=min(200, len(data))).mean()

    # 3. Volatility
    if len(data) >= 20:
        data['Volatility_20D'] = data['Daily_Return'].rolling(window=min(20, len(data))).std()

    # 4. RSI (only if we have enough data)
    if len(data) >= 14:
        delta = data['Adj_Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
    else:
        data['RSI'] = np.nan

    # 5. Bollinger Bands
    if len(data) >= 20:
        data['BB_Middle'] = data['Adj_Close'].rolling(window=20).mean()
        bb_std = data['Adj_Close'].rolling(window=20).std()
        data['BB_Upper'] = data['BB_Middle'] + (bb_std * 2)
        data['BB_Lower'] = data['BB_Middle'] - (bb_std * 2)

    # 6. Volume MA
    if 'Volume' in data.columns and len(data) >= 20:
        data['Volume_MA_20'] = data['Volume'].rolling(window=min(20, len(data))).mean()

    return data


def calculate_performance_metrics(data, ticker_name):
    """Calculate performance metrics - FIXED VERSION"""
    if data is None or len(data) < 10:
        print("⚠️ Insufficient data for analysis")
        return None

    # Check for required column
    if 'Adj_Close' not in data.columns:
        print("❌ Error: 'Adj_Close' column not found!")
        return None

    latest_price = data['Adj_Close'].iloc[-1]

    # Calculate returns based on available data
    if len(data) >= 50:
        price_50d_ago = data['Adj_Close'].iloc[-50]
        returns_50d = ((latest_price / price_50d_ago) - 1) * 100
    else:
        price_start = data['Adj_Close'].iloc[0]
        returns_50d = ((latest_price / price_start) - 1) * 100

    # Get High and Low
    if 'High' in data.columns:
        high_52w = data['High'].max()
    else:
        high_52w = data['Adj_Close'].max()

    if 'Low' in data.columns:
        low_52w = data['Low'].min()
    else:
        low_52w = data['Adj_Close'].min()

    # Calculate metrics
    if 'Daily_Return' in data.columns:
        daily_returns = data['Daily_Return'].dropna()
        if len(daily_returns) > 0:
            sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(
                252) if daily_returns.std() != 0 else 0
            max_loss = daily_returns.min()
            avg_return = daily_returns.mean()
            volatility = daily_returns.std()
        else:
            sharpe_ratio = 0
            max_loss = 0
            avg_return = 0
            volatility = 0
    else:
        sharpe_ratio = 0
        max_loss = 0
        avg_return = 0
        volatility = 0

    # Get RSI if available
    current_rsi = data['RSI'].iloc[-1] if 'RSI' in data.columns and not pd.isna(data['RSI'].iloc[-1]) else None

    metrics = {
        'Stock': ticker_name,
        'Last_Price': round(latest_price, 2),
        '52W_High': round(high_52w, 2),
        '52W_Low': round(low_52w, 2),
        'Return_%': round(returns_50d, 2),
        'Avg_Daily_Return_%': round(avg_return, 3),
        'Volatility_%': round(volatility, 3),
        'Sharpe_Ratio': round(sharpe_ratio, 2),
        'Max_Daily_Loss_%': round(max_loss, 2),
        'Current_RSI': round(current_rsi, 1) if current_rsi else None,
        'Days_Analyzed': len(data)
    }

    return metrics


def plot_stock_price_with_ma(data, ticker_name):
    """Plot stock price with moving averages - FIXED VERSION"""
    if 'Adj_Close' not in data.columns:
        print(f"❌ Cannot plot: 'Adj_Close' not found for {ticker_name}")
        return

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})

    # Top plot: Price and MAs
    axes[0].plot(data.index, data['Adj_Close'], label='Price', linewidth=2, color='blue', alpha=0.7)

    # Add moving averages if they exist
    if 'MA_20' in data.columns and not data['MA_20'].isna().all():
        axes[0].plot(data.index, data['MA_20'], label='MA 20 Days', linewidth=1.5, color='orange', alpha=0.8)
    if 'MA_50' in data.columns and not data['MA_50'].isna().all():
        axes[0].plot(data.index, data['MA_50'], label='MA 50 Days', linewidth=1.5, color='red', alpha=0.8)

    # Bollinger Bands if available
    if 'BB_Lower' in data.columns and 'BB_Upper' in data.columns:
        axes[0].fill_between(data.index, data['BB_Lower'], data['BB_Upper'],
                             alpha=0.2, color='gray', label='Bollinger Band')

    axes[0].set_title(f'{ticker_name} Stock Analysis', fontsize=16, fontweight='bold')
    axes[0].set_ylabel('Price (SAR)', fontsize=12)
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3)

    # Bottom plot: Volume
    if 'Volume' in data.columns:
        axes[1].bar(data.index, data['Volume'], color='purple', alpha=0.6, label='Volume')
        if 'Volume_MA_20' in data.columns:
            axes[1].plot(data.index, data['Volume_MA_20'], color='red', linewidth=2, label='20-Day Volume MA')

        axes[1].set_ylabel('Volume', fontsize=12)
        axes[1].legend(loc='best')
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, 'Volume data not available',
                     ha='center', va='center', transform=axes[1].transAxes, fontsize=12)
        axes[1].set_ylabel('N/A')

    plt.tight_layout()

    # Save the chart
    try:
        safe_name = ticker_name.replace(' ', '_').replace('/', '_')
        plt.savefig(f'{safe_name}_price_analysis.png', dpi=300, bbox_inches='tight')
        print(f"   💾 Chart saved as: {safe_name}_price_analysis.png")
    except Exception as e:
        print(f"   ⚠️ Could not save chart: {e}")

    plt.show()


def comprehensive_stock_analysis(ticker_symbol, ticker_name, period='1y'):
    """Perform comprehensive stock analysis - FIXED VERSION"""
    print(f"\n{'=' * 60}")
    print(f"🔍 Analyzing: {ticker_name} ({ticker_symbol})")
    print(f"{'=' * 60}")

    # Load data
    print("\n📥 Loading data...")
    stock_data = load_stock_data(ticker_symbol, period=period)

    if stock_data is None or stock_data.empty:
        print(f"❌ Failed to load {ticker_name}")
        return None, None

    # Calculate indicators
    print("\n📊 Calculating technical indicators...")
    stock_data = calculate_technical_indicators(stock_data)

    # Calculate metrics
    print("\n🧮 Calculating performance metrics...")
    metrics = calculate_performance_metrics(stock_data, ticker_name)

    if metrics:
        print(f"\n📈 {ticker_name} Performance Metrics:")
        print("-" * 50)
        for key, value in metrics.items():
            if value is not None:
                print(f"{key:25} : {value}")

    # Plot
    print(f"\n🎨 Generating charts...")
    plot_stock_price_with_ma(stock_data, ticker_name)

    return stock_data, metrics


# ====================== SIMPLE TEST FUNCTION ======================

def simple_test():
    """Simple test to verify the fix"""
    print("🧪 Simple Test - Verifying Fix")
    print("=" * 50)

    # Test with Aramco
    ticker = '2222.SR'
    print(f"\nTesting {ticker}...")

    # Load data
    data = yf.download(ticker, period='1mo')
    print(f"Raw data shape: {data.shape}")
    print(f"Raw columns: {data.columns.tolist()}")

    # Flatten if MultiIndex
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = ['_'.join(col).strip() for col in data.columns.values]
        print(f"Flattened columns: {data.columns.tolist()}")

    # Test normalization
    print("\nTesting normalization...")
    normalized = normalize_column_names(data)
    print(f"Normalized columns: {normalized.columns.tolist()}")

    # Check if Adj_Close exists
    if 'Adj_Close' in normalized.columns:
        print("✅ SUCCESS: 'Adj_Close' column created!")
        print(f"Sample Adj_Close values:")
        print(normalized['Adj_Close'].head())
    else:
        print("❌ FAILED: 'Adj_Close' not created")

    return normalized


if __name__ == "__main__":
    print("🧪 Running stock_analyzer test...")
    simple_test()
