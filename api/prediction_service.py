import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
import io
import base64

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import MinMaxScaler

import yfinance as yf

warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =========================
# Attention Layer
# =========================
class AttentionLayer(nn.Module):
    def __init__(self, hidden_size, attention_size=64):
        super().__init__()
        self.W = nn.Linear(hidden_size, attention_size)
        self.U = nn.Linear(attention_size, 1, bias=False)

    def forward(self, lstm_output):
        scores = torch.tanh(self.W(lstm_output))
        scores = self.U(scores).squeeze(-1)
        weights = F.softmax(scores, dim=1)
        context = torch.sum(lstm_output * weights.unsqueeze(-1), dim=1)
        return context, weights


# =========================
# LSTM Model
# =========================
class MultivariateLSTMWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size1=128, hidden_size2=64, dropout=0.2):
        super().__init__()

        self.lstm1 = nn.LSTM(input_size, hidden_size1, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)

        self.lstm2 = nn.LSTM(hidden_size1, hidden_size2, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)

        self.attention = AttentionLayer(hidden_size2)

        self.fc1 = nn.Linear(hidden_size2, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout1(x)

        x, _ = self.lstm2(x)
        x = self.dropout2(x)

        context, _ = self.attention(x)

        x = F.relu(self.fc1(context))
        return self.fc2(x)


# =========================
# Technical Indicators
# =========================
def calculate_rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = -delta.clip(upper=0).rolling(window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line, macd - signal_line


def calculate_bollinger(series, window=20, std=2):
    sma = series.rolling(window).mean()
    stddev = series.rolling(window).std()
    return sma + std*stddev, sma - std*stddev, sma


# =========================
# Sequence Builder
# =========================
def prepare_sequences(data, features, lookback=60):
    X, y = [], []
    close_idx = features.index('Close')

    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i])
        y.append(data[i, close_idx])

    return np.array(X), np.array(y)


# =========================
# Fetch Data from Yahoo Finance
# =========================
def fetch_stock_data_yahoo(ticker):
    """
    Historical stock data from Yahoo Finance
    """
    # yfinance uses 'Ticker' string, or download directly.
    # download returns MultiIndex if multiple tickers, but here it's one.
    # auto_adjust=True matches Close to Adj Close, but let's stick to standard and use 'Close'.
    
    # Check if ticker acts like an index or crypto? Users might pass simple symbols.
    # We'll just pass it through.
    
    # Calculate start date (5 years ago)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)
    
    # df = yf.download(ticker, start=start_date, end=end_date)
    # yfinance download prints progress, to silence: progress=False
    suffixes = ['', '.NS', '.BO']
    for suffix in suffixes:
        try_ticker = f"{ticker}{suffix}"
        print(f"Attempting to download data for: {try_ticker}")
        df = yf.download(try_ticker, start=start_date, end=end_date, progress=False, multi_level_index=False)
        
        if not df.empty:
            print(f"Successfully found data for {try_ticker}")
            ticker = try_ticker # Update ticker for consistent logging
            break
    
    if df.empty:
        raise Exception(f"No data from Yahoo Finance for {ticker} (tried suffixes: {suffixes})")

    df = df.reset_index()
    # yfinance columns are Date (index), Open, High, Low, Close, Adj Close, Volume
    # We need to ensure columns match what we expect: 'Date', 'Open', 'High', 'Low', 'Close', 'Volume'
    
    # Rename columns just in case, though yf usually gives title case
    # If the df has multi-level columns (happens sometimes), we flattened it via multi_level_index=False if updated yfinance
    # With version < 0.2.x it might differ, but I installed latest.
    
    # Standardize column names
    df.columns = [c.capitalize() for c in df.columns] 
    
    # Ensure Date is datetime
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
       # Sometimes 'Volume' might be missing for indices etc.
       # But for now raise error or handle.
       pass

    return df


# =========================
# Main Prediction Function
# =========================
def get_stock_predictions(ticker, lookback=60, epochs=15):

    df = fetch_stock_data_yahoo(ticker)

    # Indicators
    df['MA_50'] = df['Close'].rolling(50).mean()
    df['MA_100'] = df['Close'].rolling(100).mean()
    df['MA_200'] = df['Close'].rolling(200).mean()
    df['RSI'] = calculate_rsi(df['Close'])
    df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = calculate_macd(df['Close'])
    df['BB_Upper'], df['BB_Lower'], df['BB_Middle'] = calculate_bollinger(df['Close'])
    df['Pct_Change'] = df['Close'].pct_change()
    df['Volume_MA'] = df['Volume'].rolling(20).mean()
    df['Price_Range'] = (df['High'] - df['Low']) / df['Close']

    df.dropna(inplace=True)

    features = [
        'Open','High','Low','Close','Volume',
        'MA_50','MA_100','MA_200','RSI',
        'MACD','MACD_Signal','MACD_Hist',
        'BB_Upper','BB_Lower','BB_Middle',
        'Pct_Change','Volume_MA','Price_Range'
    ]

    data = df[features].values

    split = int(len(data) * 0.8)
    train, test = data[:split], data

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train)
    test_scaled = scaler.transform(test)

    X_train, y_train = prepare_sequences(train_scaled, features, lookback)
    X_test, y_test = prepare_sequences(test_scaled, features, lookback)

    X_train = torch.FloatTensor(X_train).to(device)
    y_train = torch.FloatTensor(y_train).to(device)
    X_test = torch.FloatTensor(X_test).to(device)

    model = MultivariateLSTMWithAttention(len(features)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)

    model.train()
    for _ in range(epochs):
        for x, y in loader:
            loss = criterion(model(x).squeeze(), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = model(X_test).cpu().numpy().flatten()

    close_scaler = MinMaxScaler()
    close_scaler.fit(train[:, features.index('Close')].reshape(-1,1))

    preds_inv = close_scaler.inverse_transform(preds.reshape(-1,1)).flatten()
    y_inv = close_scaler.inverse_transform(y_test.reshape(-1,1)).flatten()

    # Calculate all metrics
    rmse = np.sqrt(np.mean((preds_inv - y_inv)**2))
    mae = np.mean(np.abs(preds_inv - y_inv))
    mape = np.mean(np.abs((y_inv - preds_inv) / y_inv)) * 100

    # Directional accuracy
    y_test_prev = test_scaled[lookback-1:-1, features.index('Close')]
    y_test_prev_inv = close_scaler.inverse_transform(y_test_prev.reshape(-1,1)).flatten()
    
    actual_direction = np.sign(y_inv - y_test_prev_inv)
    pred_direction = np.sign(preds_inv - y_test_prev_inv)
    directional_accuracy = np.mean(actual_direction == pred_direction) * 100

    # Generate plot
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(14, 7))
    
    dates = df['Date'].iloc[lookback:lookback+len(y_inv)]
    
    ax.plot(dates, y_inv, label='Actual Price', color='#22c55e', linewidth=2, alpha=0.8)
    ax.plot(dates, preds_inv, label='Predicted Price', color='#3b82f6', linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Date', fontsize=12, color='white')
    ax.set_ylabel('Price (₹)', fontsize=12, color='white')
    ax.set_title(f'{ticker} Stock Price Prediction', fontsize=16, fontweight='bold', color='white')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.2)
    ax.set_facecolor('#0a0a0a')
    fig.patch.set_facecolor('#0a0a0a')
    
    # FUTURE FORECASTING
    future_days = 30
    future_predictions = []
    
    # Make sure we have the last sequence correctly
    last_sequence = data[-lookback:] # raw data
    current_seq_np = scaler.transform(last_sequence) # scaled
    current_seq = torch.FloatTensor(current_seq_np).unsqueeze(0).to(device)
    
    close_idx = features.index('Close')
    open_idx = features.index('Open')
    high_idx = features.index('High')
    low_idx = features.index('Low')
    
    for _ in range(future_days):
        with torch.no_grad():
            pred = model(current_seq).item()
        future_predictions.append(pred)
        
        # Construct next step based on last step
        next_step = current_seq[:, -1, :].clone()
        next_step[0, close_idx] = pred
        next_step[0, open_idx] = pred
        next_step[0, high_idx] = pred
        next_step[0, low_idx] = pred
        # For other features, we carry forward the last value (naive assumption)
        
        next_step = next_step.unsqueeze(1)
        current_seq = torch.cat((current_seq[:, 1:, :], next_step), dim=1)
        
    future_preds_inv = close_scaler.inverse_transform(np.array(future_predictions).reshape(-1,1)).flatten()
    
    last_date = df['Date'].iloc[-1]
    future_dates_dt = [last_date + timedelta(days=i) for i in range(1, future_days+1)]
    future_dates_str = [d.strftime('%Y-%m-%d') for d in future_dates_dt]
    
    # Plot Future
    ax.plot(future_dates_dt, future_preds_inv, label='30-Day Forecast', color='#f59e0b', linestyle='--', linewidth=2)
    ax.legend(loc='best', fontsize=11)

    plt.tight_layout()
    
    # Convert plot to base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=100, facecolor='#0a0a0a')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode()
    plt.close()

    # Prepare recent predictions for table
    recent_predictions = []
    for i in range(min(10, len(y_inv))):
        idx = -(i+1)
        recent_predictions.append({
            'date': dates.iloc[idx].strftime('%Y-%m-%d'),
            'actual': float(y_inv[idx]),
            'predicted': float(preds_inv[idx]),
            'difference': float(y_inv[idx] - preds_inv[idx])
        })

    return {
        "ticker": ticker,
        "metrics": {
            "rmse": float(rmse),
            "mae": float(mae),
            "mape": float(mape),
            "directional_accuracy": float(directional_accuracy)
        },
        "plot": image_base64,
        "predictions": recent_predictions,
        "last_actual_price": float(y_inv[-1]),
        "last_predicted_price": float(preds_inv[-1]),
        "forecast_30_days": future_preds_inv.tolist(),
        "forecast_dates": future_dates_str
    }
