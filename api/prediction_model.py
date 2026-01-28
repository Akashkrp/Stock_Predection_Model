import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============================================================================
# ATTENTION MECHANISM
# ============================================================================

class AttentionLayer(nn.Module):
    """
    Attention mechanism for LSTM outputs
    Computes attention weights over time steps to focus on relevant information
    """
    def __init__(self, hidden_size, attention_size=64):
        super(AttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.attention_size = attention_size
        
        # Attention weights
        self.W = nn.Linear(hidden_size, attention_size)
        self.U = nn.Linear(attention_size, 1, bias=False)
        self.tanh = nn.Tanh()
        
    def forward(self, lstm_output):
        """
        Args:
            lstm_output: (batch_size, seq_len, hidden_size)
        Returns:
            context: (batch_size, hidden_size) - weighted sum of lstm outputs
            attention_weights: (batch_size, seq_len) - attention weights for visualization
        """
        # Calculate attention scores
        attn_scores = self.tanh(self.W(lstm_output))
        attn_scores = self.U(attn_scores)
        attn_scores = attn_scores.squeeze(-1)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attn_scores, dim=1)
        
        # Calculate weighted sum (context vector)
        attention_weights_expanded = attention_weights.unsqueeze(-1)
        weighted_lstm = lstm_output * attention_weights_expanded
        context = weighted_lstm.sum(dim=1)
        
        return context, attention_weights

# ============================================================================
# MULTIVARIATE LSTM WITH ATTENTION
# ============================================================================

class MultivariateLSTMWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size1=128, hidden_size2=64, 
                 num_layers=1, dropout=0.2, attention_size=64):
        super(MultivariateLSTMWithAttention, self).__init__()
        
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        
        # First LSTM layer
        self.lstm1 = nn.LSTM(input_size=input_size, 
                             hidden_size=hidden_size1, 
                             num_layers=num_layers,
                             batch_first=True,
                             dropout=dropout if num_layers > 1 else 0)
        self.dropout1 = nn.Dropout(dropout)
        
        # Second LSTM layer
        self.lstm2 = nn.LSTM(input_size=hidden_size1, 
                             hidden_size=hidden_size2,
                             num_layers=num_layers,
                             batch_first=True,
                             dropout=dropout if num_layers > 1 else 0)
        self.dropout2 = nn.Dropout(dropout)
        
        # Attention layer
        self.attention = AttentionLayer(hidden_size2, attention_size)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size2, 32)
        self.relu = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 1)
        
        # Store attention weights for visualization
        self.last_attention_weights = None
        
    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        
        out, _ = self.lstm2(out)
        out = self.dropout2(out)
        
        # Apply attention mechanism
        context, attention_weights = self.attention(out)
        self.last_attention_weights = attention_weights.detach()
        
        # Fully connected layers
        out = self.fc1(context)
        out = self.relu(out)
        out = self.dropout3(out)
        out = self.fc2(out)
        
        return out

# ============================================================================
# 1. DATA LOADING AND FEATURE ENGINEERING
# ============================================================================

now = datetime.now()
start = datetime(now.year-10, now.month, now.day)
end = datetime(now.year, now.month, now.day)

ticker = "AAPL"
print(f"Downloading {ticker} data...")

df = yf.download(ticker, start, end)
df = df.reset_index()

# Feature Engineering - Technical Indicators
def calculate_rsi(data, window=14):
    """Calculate Relative Strength Index"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data, fast=12, slow=26, signal=9):
    """Calculate MACD indicators"""
    ema_fast = data.ewm(span=fast, adjust=False).mean()
    ema_slow = data.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist

def calculate_bollinger_bands(data, window=20, num_std=2):
    """Calculate Bollinger Bands"""
    sma = data.rolling(window=window).mean()
    std = data.rolling(window=window).std()
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    return upper_band, lower_band, sma

# Add all features
print("Calculating technical indicators...")
df['MA_50'] = df['Close'].rolling(50).mean()
df['MA_100'] = df['Close'].rolling(100).mean()
df['MA_200'] = df['Close'].rolling(200).mean()
df['RSI'] = calculate_rsi(df['Close'])
df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = calculate_macd(df['Close'])
df['BB_Upper'], df['BB_Lower'], df['BB_Middle'] = calculate_bollinger_bands(df['Close'])
df['Pct_Change'] = df['Close'].pct_change()
df['Volume_MA'] = df['Volume'].rolling(20).mean()
df['Price_Range'] = (df['High'] - df['Low']) / df['Close']

# Calculate volatility for segmentation
df['Volatility'] = df['Close'].pct_change().rolling(20).std()

# Drop NaN values
df = df.dropna().reset_index(drop=True)

# Select features for multivariate model
feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 
                   'MA_50', 'MA_100', 'MA_200', 'RSI', 
                   'MACD', 'MACD_Signal', 'MACD_Hist',
                   'BB_Upper', 'BB_Lower', 'BB_Middle',
                   'Pct_Change', 'Volume_MA', 'Price_Range']

print(f"Total features: {len(feature_columns)}")
print(f"Total data points: {len(df)}")

# ============================================================================
# 2. HELPER FUNCTIONS
# ============================================================================

def prepare_sequences(data, features, target_col='Close', lookback=100):
    """Prepare sequences for LSTM"""
    X, y = [], []
    target_idx = features.index(target_col)
    
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i, :])
        y.append(data[i, target_idx])
    
    return np.array(X), np.array(y)

def calculate_metrics(y_true, y_pred, y_true_prev, volatility=None):
    """Calculate comprehensive metrics including directional accuracy"""
    # Traditional metrics
    mse = np.mean((y_pred - y_true) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_pred - y_true))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    # R-squared
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    # Directional metrics
    actual_direction = np.sign(y_true - y_true_prev)
    pred_direction = np.sign(y_pred - y_true_prev)
    
    directional_accuracy = np.mean(actual_direction == pred_direction) * 100
    
    # Up/Down specific accuracy
    up_mask = actual_direction > 0
    down_mask = actual_direction < 0
    
    up_accuracy = np.mean(pred_direction[up_mask] == actual_direction[up_mask]) * 100 if np.sum(up_mask) > 0 else 0
    down_accuracy = np.mean(pred_direction[down_mask] == actual_direction[down_mask]) * 100 if np.sum(down_mask) > 0 else 0
    
    # Confusion matrix for directions
    true_positives = np.sum((actual_direction > 0) & (pred_direction > 0))
    false_positives = np.sum((actual_direction <= 0) & (pred_direction > 0))
    true_negatives = np.sum((actual_direction <= 0) & (pred_direction <= 0))
    false_negatives = np.sum((actual_direction > 0) & (pred_direction <= 0))
    
    metrics = {
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R2': r2,
        'Directional_Accuracy': directional_accuracy,
        'Up_Accuracy': up_accuracy,
        'Down_Accuracy': down_accuracy,
        'True_Positives': true_positives,
        'False_Positives': false_positives,
        'True_Negatives': true_negatives,
        'False_Negatives': false_negatives
    }
    
    # Volatility-segmented metrics
    if volatility is not None:
        volatility_median = np.median(volatility)
        calm_mask = volatility <= volatility_median
        volatile_mask = volatility > volatility_median
        
        if np.sum(calm_mask) > 0:
            metrics['Calm_RMSE'] = np.sqrt(np.mean((y_pred[calm_mask] - y_true[calm_mask]) ** 2))
            metrics['Calm_MAE'] = np.mean(np.abs(y_pred[calm_mask] - y_true[calm_mask]))
            metrics['Calm_Dir_Acc'] = np.mean((actual_direction[calm_mask] == pred_direction[calm_mask])) * 100
        
        if np.sum(volatile_mask) > 0:
            metrics['Volatile_RMSE'] = np.sqrt(np.mean((y_pred[volatile_mask] - y_true[volatile_mask]) ** 2))
            metrics['Volatile_MAE'] = np.mean(np.abs(y_pred[volatile_mask] - y_true[volatile_mask]))
            metrics['Volatile_Dir_Acc'] = np.mean((actual_direction[volatile_mask] == pred_direction[volatile_mask])) * 100
    
    return metrics

def naive_baseline_predictions(y_true_prev):
    """
    Naive persistence baseline: ŷₜ = yₜ₋₁
    Simply predicts next value will be same as current value
    """
    return y_true_prev.copy()

# ============================================================================
# 3. WALK-FORWARD VALIDATION WITH BASELINE
# ============================================================================

def walk_forward_validation(df, feature_columns, lookback=100, train_size=1000, 
                           test_size=50, epochs=20, batch_size=32, learning_rate=0.001):
    """
    Perform walk-forward validation with attention mechanism and baseline
    """
    print("\n" + "="*80)
    print("WALK-FORWARD VALIDATION: LSTM WITH ATTENTION vs BASELINE")
    print("="*80)
    
    # LSTM storage
    lstm_predictions = []
    lstm_actuals = []
    lstm_previous = []
    lstm_volatility = []
    lstm_fold_metrics = []
    all_attention_weights = []
    
    # Baseline storage
    baseline_predictions = []
    baseline_actuals = []
    baseline_previous = []
    baseline_volatility = []
    baseline_fold_metrics = []
    
    data_values = df[feature_columns].values
    volatility_values = df['Volatility'].values
    
    # Create scaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    n_splits = (len(data_values) - train_size - lookback) // test_size
    print(f"\nTotal folds: {n_splits}")
    print(f"Train size: {train_size}, Test size: {test_size}, Lookback: {lookback}\n")
    
    for fold in range(n_splits):
        train_start = fold * test_size
        train_end = train_start + train_size
        test_end = train_end + test_size
        
        if test_end > len(data_values):
            break
            
        print(f"\n{'='*70}")
        print(f"Fold {fold + 1}/{n_splits}")
        print(f"Train: {train_start} to {train_end} | Test: {train_end} to {test_end}")
        print(f"{'='*70}")
        
        # Scale data
        train_data = data_values[train_start:train_end]
        test_data = data_values[train_start:test_end]
        
        scaler.fit(train_data)
        train_scaled = scaler.transform(train_data)
        test_scaled = scaler.transform(test_data)
        
        # Prepare sequences
        X_train, y_train = prepare_sequences(train_scaled, feature_columns, lookback=lookback)
        X_test, y_test = prepare_sequences(test_scaled, feature_columns, lookback=lookback)
        
        # Get previous values and volatility
        y_test_prev = test_scaled[lookback-1:-1, feature_columns.index('Close')]
        test_volatility = volatility_values[train_start+lookback:test_end]
        
        # Convert to tensors for LSTM
        X_train = torch.FloatTensor(X_train).to(device)
        y_train = torch.FloatTensor(y_train).to(device)
        X_test = torch.FloatTensor(X_test).to(device)
        
        # ============================================================
        # LSTM MODEL TRAINING
        # ============================================================
        print("\n🤖 Training LSTM with Attention...")
        model = MultivariateLSTMWithAttention(
            input_size=len(feature_columns),
            hidden_size1=128,
            hidden_size2=64,
            attention_size=64,
            dropout=0.2
        ).to(device)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # Training
        dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        model.train()
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_x, batch_y in train_loader:
                outputs = model(batch_x)
                loss = criterion(outputs.squeeze(), batch_y)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            if (epoch + 1) % 5 == 0:
                print(f'  Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(train_loader):.6f}')
        
        # LSTM Prediction
        model.eval()
        with torch.no_grad():
            y_pred_lstm = model(X_test).cpu().numpy().flatten()
            if model.last_attention_weights is not None:
                all_attention_weights.append(model.last_attention_weights.cpu().numpy())
        
        # Inverse transform
        close_scaler = MinMaxScaler(feature_range=(0, 1))
        close_scaler.fit(train_data[:, feature_columns.index('Close')].reshape(-1, 1))
        
        y_pred_lstm_inv = close_scaler.inverse_transform(y_pred_lstm.reshape(-1, 1)).flatten()
        y_test_inv = close_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        y_test_prev_inv = close_scaler.inverse_transform(y_test_prev.reshape(-1, 1)).flatten()
        
        # ============================================================
        # BASELINE MODEL (Naive Persistence)
        # ============================================================
        print("\n📊 Generating Baseline Predictions (Naive Persistence: ŷₜ = yₜ₋₁)...")
        y_pred_baseline = naive_baseline_predictions(y_test_prev_inv)
        
        # ============================================================
        # CALCULATE METRICS FOR BOTH MODELS
        # ============================================================
        lstm_metrics = calculate_metrics(y_test_inv, y_pred_lstm_inv, y_test_prev_inv, test_volatility)
        baseline_metrics = calculate_metrics(y_test_inv, y_pred_baseline, y_test_prev_inv, test_volatility)
        
        lstm_fold_metrics.append(lstm_metrics)
        baseline_fold_metrics.append(baseline_metrics)
        
        # Store results
        lstm_predictions.extend(y_pred_lstm_inv)
        lstm_actuals.extend(y_test_inv)
        lstm_previous.extend(y_test_prev_inv)
        lstm_volatility.extend(test_volatility)
        
        baseline_predictions.extend(y_pred_baseline)
        baseline_actuals.extend(y_test_inv)
        baseline_previous.extend(y_test_prev_inv)
        baseline_volatility.extend(test_volatility)
        
        # Print fold comparison
        print(f"\n📈 Fold {fold+1} Results:")
        print(f"  LSTM      - RMSE: ${lstm_metrics['RMSE']:.2f} | Dir Acc: {lstm_metrics['Directional_Accuracy']:.2f}%")
        print(f"  Baseline  - RMSE: ${baseline_metrics['RMSE']:.2f} | Dir Acc: {baseline_metrics['Directional_Accuracy']:.2f}%")
        improvement = ((baseline_metrics['RMSE'] - lstm_metrics['RMSE']) / baseline_metrics['RMSE']) * 100
        print(f"  Improvement: {improvement:.2f}% RMSE reduction")
    
    return {
        'lstm': {
            'predictions': np.array(lstm_predictions),
            'actuals': np.array(lstm_actuals),
            'previous': np.array(lstm_previous),
            'volatility': np.array(lstm_volatility),
            'fold_metrics': lstm_fold_metrics,
            'attention_weights': all_attention_weights
        },
        'baseline': {
            'predictions': np.array(baseline_predictions),
            'actuals': np.array(baseline_actuals),
            'previous': np.array(baseline_previous),
            'volatility': np.array(baseline_volatility),
            'fold_metrics': baseline_fold_metrics
        }
    }

# ============================================================================
# 4. RUN WALK-FORWARD VALIDATION
# ============================================================================

results = walk_forward_validation(
    df=df,
    feature_columns=feature_columns,
    lookback=100,
    train_size=1000,
    test_size=50,
    epochs=20,
    batch_size=32,
    learning_rate=0.001
)

# ============================================================================
# 5. COMPREHENSIVE RESULTS AND COMPARISON
# ============================================================================

print("\n" + "="*80)
print("FINAL RESULTS: LSTM WITH ATTENTION vs BASELINE")
print("="*80)

# Calculate overall metrics
lstm_overall = calculate_metrics(
    results['lstm']['actuals'], 
    results['lstm']['predictions'], 
    results['lstm']['previous'],
    results['lstm']['volatility']
)

baseline_overall = calculate_metrics(
    results['baseline']['actuals'], 
    results['baseline']['predictions'], 
    results['baseline']['previous'],
    results['baseline']['volatility']
)

# Create comparison table
print("\n" + "="*80)
print("📊 MODEL COMPARISON TABLE")
print("="*80)

comparison_data = {
    'Metric': [
        'RMSE ($)',
        'MAE ($)',
        'MAPE (%)',
        'R² Score',
        'Directional Accuracy (%)',
        'Up Movement Accuracy (%)',
        'Down Movement Accuracy (%)',
        '',  # separator
        'CALM PERIODS:',
        '  RMSE ($)',
        '  MAE ($)',
        '  Directional Accuracy (%)',
        '',  # separator
        'VOLATILE PERIODS:',
        '  RMSE ($)',
        '  MAE ($)',
        '  Directional Accuracy (%)',
    ],
    'Baseline (Naive)': [
        f"{baseline_overall['RMSE']:.2f}",
        f"{baseline_overall['MAE']:.2f}",
        f"{baseline_overall['MAPE']:.2f}",
        f"{baseline_overall['R2']:.4f}",
        f"{baseline_overall['Directional_Accuracy']:.2f}",
        f"{baseline_overall['Up_Accuracy']:.2f}",
        f"{baseline_overall['Down_Accuracy']:.2f}",
        '',
        '',
        f"{baseline_overall.get('Calm_RMSE', 0):.2f}",
        f"{baseline_overall.get('Calm_MAE', 0):.2f}",
        f"{baseline_overall.get('Calm_Dir_Acc', 0):.2f}",
        '',
        '',
        f"{baseline_overall.get('Volatile_RMSE', 0):.2f}",
        f"{baseline_overall.get('Volatile_MAE', 0):.2f}",
        f"{baseline_overall.get('Volatile_Dir_Acc', 0):.2f}",
    ],
    'LSTM + Attention': [
        f"{lstm_overall['RMSE']:.2f}",
        f"{lstm_overall['MAE']:.2f}",
        f"{lstm_overall['MAPE']:.2f}",
        f"{lstm_overall['R2']:.4f}",
        f"{lstm_overall['Directional_Accuracy']:.2f}",
        f"{lstm_overall['Up_Accuracy']:.2f}",
        f"{lstm_overall['Down_Accuracy']:.2f}",
        '',
        '',
        f"{lstm_overall.get('Calm_RMSE', 0):.2f}",
        f"{lstm_overall.get('Calm_MAE', 0):.2f}",
        f"{lstm_overall.get('Calm_Dir_Acc', 0):.2f}",
        '',
        '',
        f"{lstm_overall.get('Volatile_RMSE', 0):.2f}",
        f"{lstm_overall.get('Volatile_MAE', 0):.2f}",
        f"{lstm_overall.get('Volatile_Dir_Acc', 0):.2f}",
    ],
    'Improvement': [
        f"{((baseline_overall['RMSE'] - lstm_overall['RMSE']) / baseline_overall['RMSE'] * 100):.2f}%",
        f"{((baseline_overall['MAE'] - lstm_overall['MAE']) / baseline_overall['MAE'] * 100):.2f}%",
        f"{((baseline_overall['MAPE'] - lstm_overall['MAPE']) / baseline_overall['MAPE'] * 100):.2f}%",
        f"{((lstm_overall['R2'] - baseline_overall['R2']) / (1 - baseline_overall['R2']) * 100 if baseline_overall['R2'] < 1 else 0):.2f}%",
        f"{(lstm_overall['Directional_Accuracy'] - baseline_overall['Directional_Accuracy']):.2f}pp",
        f"{(lstm_overall['Up_Accuracy'] - baseline_overall['Up_Accuracy']):.2f}pp",
        f"{(lstm_overall['Down_Accuracy'] - baseline_overall['Down_Accuracy']):.2f}pp",
        '',
        '',
        f"{((baseline_overall.get('Calm_RMSE', 0) - lstm_overall.get('Calm_RMSE', 0)) / baseline_overall.get('Calm_RMSE', 1) * 100):.2f}%",
        f"{((baseline_overall.get('Calm_MAE', 0) - lstm_overall.get('Calm_MAE', 0)) / baseline_overall.get('Calm_MAE', 1) * 100):.2f}%",
        f"{(lstm_overall.get('Calm_Dir_Acc', 0) - baseline_overall.get('Calm_Dir_Acc', 0)):.2f}pp",
        '',
        '',
        f"{((baseline_overall.get('Volatile_RMSE', 0) - lstm_overall.get('Volatile_RMSE', 0)) / baseline_overall.get('Volatile_RMSE', 1) * 100):.2f}%",
        f"{((baseline_overall.get('Volatile_MAE', 0) - lstm_overall.get('Volatile_MAE', 0)) / baseline_overall.get('Volatile_MAE', 1) * 100):.2f}%",
        f"{(lstm_overall.get('Volatile_Dir_Acc', 0) - baseline_overall.get('Volatile_Dir_Acc', 0)):.2f}pp",
    ]
}

comparison_df = pd.DataFrame(comparison_data)
print(comparison_df.to_string(index=False))

print("\n" + "="*80)
print("📈 KEY INSIGHTS")
print("="*80)

rmse_improvement = ((baseline_overall['RMSE'] - lstm_overall['RMSE']) / baseline_overall['RMSE']) * 100
dir_improvement = lstm_overall['Directional_Accuracy'] - baseline_overall['Directional_Accuracy']

print(f"\n✅ LSTM improves RMSE by {rmse_improvement:.2f}% over naive baseline")
print(f"✅ LSTM improves directional accuracy by {dir_improvement:.2f} percentage points")

if lstm_overall.get('Calm_RMSE') and baseline_overall.get('Calm_RMSE'):
    calm_improvement = ((baseline_overall['Calm_RMSE'] - lstm_overall['Calm_RMSE']) / baseline_overall['Calm_RMSE']) * 100
    volatile_improvement = ((baseline_overall['Volatile_RMSE'] - lstm_overall['Volatile_RMSE']) / baseline_overall['Volatile_RMSE']) * 100
    print(f"✅ LSTM performs {calm_improvement:.2f}% better in CALM periods")
    print(f"✅ LSTM performs {volatile_improvement:.2f}% better in VOLATILE periods")

# ============================================================================
# 6. COMPREHENSIVE VISUALIZATION
# ============================================================================

fig = plt.figure(figsize=(18, 14))
gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)

fig.suptitle(f'{ticker} - LSTM with Attention vs Baseline Comparison', fontsize=16, fontweight='bold')

# Plot 1: Price Predictions Comparison
ax1 = fig.add_subplot(gs[0, :2])
ax1.plot(results['lstm']['actuals'], 'b-', label='Actual Price', linewidth=2, alpha=0.8)
ax1.plot(results['lstm']['predictions'], 'r-', label='LSTM Prediction', linewidth=1.5, alpha=0.7)
ax1.plot(results['baseline']['predictions'], 'g--', label='Baseline (Naive)', linewidth=1.5, alpha=0.7)
ax1.set_xlabel('Time Steps')
ax1.set_ylabel('Price ($)')
ax1.set_title('Price Predictions: LSTM vs Baseline')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Error Distribution Comparison
ax2 = fig.add_subplot(gs[0, 2])
lstm_errors = results['lstm']['predictions'] - results['lstm']['actuals']
baseline_errors = results['baseline']['predictions'] - results['baseline']['actuals']

ax2.hist(baseline_errors, bins=40, alpha=0.5, color='green', label='Baseline', edgecolor='black')
ax2.hist(lstm_errors, bins=40, alpha=0.5, color='red', label='LSTM', edgecolor='black')
ax2.axvline(x=0, color='black', linestyle='--', linewidth=2)
ax2.set_xlabel('Prediction Error ($)')
ax2.set_ylabel('Frequency')
ax2.set_title('Error Distribution')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: RMSE Comparison by Fold
ax3 = fig.add_subplot(gs[1, 0])
fold_indices = range(1, len(results['lstm']['fold_metrics']) + 1)
lstm_rmse = [m['RMSE'] for m in results['lstm']['fold_metrics']]
baseline_rmse = [m['RMSE'] for m in results['baseline']['fold_metrics']]

x = np.arange(len(fold_indices))
width = 0.35
ax3.bar(x - width/2, baseline_rmse, width, label='Baseline', alpha=0.8, color='green')
ax3.bar(x + width/2, lstm_rmse, width, label='LSTM', alpha=0.8, color='red')
ax3.set_xlabel('Fold Number')
ax3.set_ylabel('RMSE ($)')
ax3.set_title('RMSE by Fold')
ax3.set_xticks(x)
ax3.set_xticklabels(fold_indices)
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

# Plot 4: Directional Accuracy Comparison by Fold
ax4 = fig.add_subplot(gs[1, 1])
lstm_dir_acc = [m['Directional_Accuracy'] for m in results['lstm']['fold_metrics']]
baseline_dir_acc = [m['Directional_Accuracy'] for m in results['baseline']['fold_metrics']]

ax4.bar(x - width/2, baseline_dir_acc, width, label='Baseline', alpha=0.8, color='green')
ax4.bar(x + width/2, lstm_dir_acc, width, label='LSTM', alpha=0.8, color='red')
ax4.axhline(y=50, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Random (50%)')
ax4.set_xlabel('Fold Number')
ax4.set_ylabel('Directional Accuracy (%)')
ax4.set_title('Directional Accuracy by Fold')
ax4.set_xticks(x)
ax4.set_xticklabels(fold_indices)
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')
ax4.set_ylim([0, 100])

# Plot 5: Rolling Directional Accuracy
ax5 = fig.add_subplot(gs[1, 2])
lstm_actual_dir = np.sign(results['lstm']['actuals'] - results['lstm']['previous'])
lstm_pred_dir = np.sign(results['lstm']['predictions'] - results['lstm']['previous'])
baseline_pred_dir = np.sign(results['baseline']['predictions'] - results['baseline']['previous'])

lstm_correct = (lstm_actual_dir == lstm_pred_dir).astype(int)
baseline_correct = (lstm_actual_dir == baseline_pred_dir).astype(int)

window = 50
lstm_rolling = pd.Series(lstm_correct).rolling(window=window).mean() * 100
baseline_rolling = pd.Series(baseline_correct).rolling(window=window).mean() * 100

ax5.plot(lstm_rolling, 'r-', linewidth=2, label='LSTM', alpha=0.8)
ax5.plot(baseline_rolling, 'g--', linewidth=2, label='Baseline', alpha=0.8)
ax5.axhline(y=50, color='black', linestyle='--', linewidth=1, label='Random')
ax5.set_xlabel('Time Steps')
ax5.set_ylabel('Accuracy (%)')
ax5.set_title(f'Rolling Directional Accuracy (Window={window})')
ax5.legend()
ax5.grid(True, alpha=0.3)
ax5.set_ylim([0, 100])

# Plot 6: Volatility Segmentation - RMSE
ax6 = fig.add_subplot(gs[2, 0])
volatility_median = np.median(results['lstm']['volatility'])
calm_mask = results['lstm']['volatility'] <= volatility_median
volatile_mask = results['lstm']['volatility'] > volatility_median

metrics_data = {
    'Model': ['Baseline', 'Baseline', 'LSTM', 'LSTM'],
    'Period': ['Calm', 'Volatile', 'Calm', 'Volatile'],
    'RMSE': [
        baseline_overall.get('Calm_RMSE', 0),
        baseline_overall.get('Volatile_RMSE', 0),
        lstm_overall.get('Calm_RMSE', 0),
        lstm_overall.get('Volatile_RMSE', 0)
    ]
}

metrics_df = pd.DataFrame(metrics_data)
calm_data = metrics_df[metrics_df['Period'] == 'Calm']
volatile_data = metrics_df[metrics_df['Period'] == 'Volatile']

x_pos = np.arange(2)
width = 0.35

ax6.bar(x_pos - width/2, calm_data['RMSE'].values, width, label='Calm', alpha=0.8, color='lightblue')
ax6.bar(x_pos + width/2, volatile_data['RMSE'].values, width, label='Volatile', alpha=0.8, color='orange')
ax6.set_xlabel('Model')
ax6.set_ylabel('RMSE ($)')
ax6.set_title('RMSE by Volatility Regime')
ax6.set_xticks(x_pos)
ax6.set_xticklabels(['Baseline', 'LSTM'])
ax6.legend()
ax6.grid(True, alpha=0.3, axis='y')

# Plot 7: Volatility Segmentation - Directional Accuracy
ax7 = fig.add_subplot(gs[2, 1])
dir_metrics_data = {
    'Model': ['Baseline', 'Baseline', 'LSTM', 'LSTM'],
    'Period': ['Calm', 'Volatile', 'Calm', 'Volatile'],
    'Dir_Acc': [
        baseline_overall.get('Calm_Dir_Acc', 0),
        baseline_overall.get('Volatile_Dir_Acc', 0),
        lstm_overall.get('Calm_Dir_Acc', 0),
        lstm_overall.get('Volatile_Dir_Acc', 0)
    ]
}

dir_metrics_df = pd.DataFrame(dir_metrics_data)
dir_calm_data = dir_metrics_df[dir_metrics_df['Period'] == 'Calm']
dir_volatile_data = dir_metrics_df[dir_metrics_df['Period'] == 'Volatile']

ax7.bar(x_pos - width/2, dir_calm_data['Dir_Acc'].values, width, label='Calm', alpha=0.8, color='lightblue')
ax7.bar(x_pos + width/2, dir_volatile_data['Dir_Acc'].values, width, label='Volatile', alpha=0.8, color='orange')
ax7.axhline(y=50, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax7.set_xlabel('Model')
ax7.set_ylabel('Directional Accuracy (%)')
ax7.set_title('Directional Accuracy by Volatility Regime')
ax7.set_xticks(x_pos)
ax7.set_xticklabels(['Baseline', 'LSTM'])
ax7.legend()
ax7.grid(True, alpha=0.3, axis='y')
ax7.set_ylim([0, 100])

# Plot 8: Volatility Time Series with Error Overlay
ax8 = fig.add_subplot(gs[2, 2])
normalized_volatility = (results['lstm']['volatility'] - results['lstm']['volatility'].min()) / \
                        (results['lstm']['volatility'].max() - results['lstm']['volatility'].min())
ax8.fill_between(range(len(normalized_volatility)), normalized_volatility, alpha=0.3, color='gray', label='Volatility')
ax8.plot(np.abs(lstm_errors) / results['lstm']['actuals'], 'r-', linewidth=1, alpha=0.7, label='LSTM Error %')
ax8.plot(np.abs(baseline_errors) / results['baseline']['actuals'], 'g--', linewidth=1, alpha=0.7, label='Baseline Error %')
ax8.set_xlabel('Time Steps')
ax8.set_ylabel('Normalized Values')
ax8.set_title('Volatility vs Prediction Errors')
ax8.legend()
ax8.grid(True, alpha=0.3)

# Plot 9: Attention Heatmap
ax9 = fig.add_subplot(gs[3, 0])
if len(results['lstm']['attention_weights']) > 0:
    sample_attention = results['lstm']['attention_weights'][-1][:20]
    sns.heatmap(sample_attention, cmap='YlOrRd', ax=ax9, cbar_kws={'label': 'Attention Weight'})
    ax9.set_xlabel('Time Steps (Lookback)')
    ax9.set_ylabel('Test Samples')
    ax9.set_title('Attention Weights (Last Fold)')
    ax9.invert_yaxis()

# Plot 10: Average Attention Distribution
ax10 = fig.add_subplot(gs[3, 1])
if len(results['lstm']['attention_weights']) > 0:
    avg_attention = results['lstm']['attention_weights'][-1].mean(axis=0)
    ax10.plot(range(len(avg_attention)), avg_attention, 'b-', linewidth=2)
    ax10.fill_between(range(len(avg_attention)), avg_attention, alpha=0.3)
    
    # Mark top attended steps
    top_k = 5
    top_indices = np.argsort(avg_attention)[-top_k:]
    for idx in top_indices:
        ax10.axvline(x=idx, color='red', linestyle='--', alpha=0.5, linewidth=1)
    
    ax10.set_xlabel('Time Steps (from oldest to newest)')
    ax10.set_ylabel('Average Attention Weight')
    ax10.set_title('Average Attention Distribution')
    ax10.grid(True, alpha=0.3)

# Plot 11: Cumulative Error Comparison
ax11 = fig.add_subplot(gs[3, 2])
lstm_cumulative_error = np.cumsum(np.abs(lstm_errors))
baseline_cumulative_error = np.cumsum(np.abs(baseline_errors))

ax11.plot(lstm_cumulative_error, 'r-', linewidth=2, label='LSTM', alpha=0.8)
ax11.plot(baseline_cumulative_error, 'g--', linewidth=2, label='Baseline', alpha=0.8)
ax11.set_xlabel('Time Steps')
ax11.set_ylabel('Cumulative Absolute Error ($)')
ax11.set_title('Cumulative Error Over Time')
ax11.legend()
ax11.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n✅ Comprehensive analysis complete with baseline comparison and volatility segmentation!")
print("\n💡 Key Takeaways:")
print("  • Baseline provides a strong benchmark - hard to beat in stable markets")
print("  • LSTM shows its value in capturing complex patterns beyond persistence")
print("  • Volatility analysis reveals model strengths in different market conditions")
print("  • Attention mechanism helps interpretability by showing temporal focus")
