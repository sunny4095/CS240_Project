import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# -------------------------------
# 1. Load and preprocess the data
# -------------------------------
df = pd.read_csv('AAPL.csv', parse_dates=['Date'])
df.sort_values('Date', inplace=True)
df.set_index('Date', inplace=True)

# Compute features
df['Return'] = df['Close'].pct_change()
df['LogReturn'] = np.log(df['Close']) - np.log(df['Close'].shift(1))
df['Volatility'] = (df['High'] - df['Low']) / df['Open']
df['PriceChange'] = (df['Close'] - df['Open']) / df['Open']
df['VolumeChange'] = df['Volume'].pct_change()

# Handle NaN or Inf
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# -------------------------
# 2. Fit the Hidden Markov Model
# -------------------------
hmm_features = df[['LogReturn', 'Volatility', 'VolumeChange']].values

# Ensure hmm_features is clean
valid_idx = ~np.isnan(hmm_features).any(axis=1) & ~np.isinf(hmm_features).any(axis=1)
df = df[valid_idx]
hmm_features = hmm_features[valid_idx]

# Fit HMM
n_hidden_states = 5
hmm_model = GaussianHMM(n_components=n_hidden_states, covariance_type='diag', n_iter=1000)
hmm_model.fit(hmm_features)

# Get hidden states and state means
hidden_states = hmm_model.predict(hmm_features)
state_means = hmm_model.means_

# Append HMM-derived features
df['HMM_State'] = hidden_states
df['State_Mean_LogReturn'] = state_means[hidden_states, 0]
df['State_Mean_Volatility'] = state_means[hidden_states, 1]
df['State_Mean_VolumeChange'] = state_means[hidden_states, 2]

# ---------------------------------
# 3. Prepare the data for LSTM
# ---------------------------------
feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume',
               'LogReturn', 'Volatility', 'PriceChange', 'VolumeChange',
               'HMM_State', 'State_Mean_LogReturn', 'State_Mean_Volatility', 'State_Mean_VolumeChange']
target_col = 'Close'
n_lags = 45

# Scale all data together first to maintain relationships
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[feature_cols + [target_col]])

# Split data before creating sequences to avoid leakage
split_idx = int(0.98 * len(scaled_data))
train_data = scaled_data[:split_idx]
test_data = scaled_data[split_idx:]

# Create sequences function
def create_sequences(data, n_lags):
    X, y = [], []
    for i in range(n_lags, len(data)):
        X.append(data[i-n_lags:i, :-1])  # All columns except last (features)
        y.append(data[i, -1])            # Last column (target)
    return np.array(X), np.array(y)

# Create sequences separately for train and test
X_train, y_train = create_sequences(train_data, n_lags)
X_test, y_test = create_sequences(test_data, n_lags)

# -------------------------
# 4. Build and train LSTM
# -------------------------
model = Sequential([
    LSTM(100, return_sequences=True, input_shape=(n_lags, len(feature_cols))),
    LSTM(100),
    Dense(50, activation='relu'),
    Dense(25, activation='relu'),
    Dense(1, activation='linear')
])

# Add early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
              loss='mean_squared_error')

history = model.fit(X_train, y_train, 
                    epochs=20, 
                    batch_size=32, 
                    validation_split=0.1, 
                    verbose=1, 
                    callbacks=[early_stop])

# -------------------------
# 5. Predict and evaluate
# -------------------------
# Predict scaled values
predicted_scaled = model.predict(X_test)

# Create inverse scaler for target (Close price)
# We need to create a dummy array with proper shape for inverse transform
dummy_array = np.zeros((len(predicted_scaled), len(feature_cols)+1))
dummy_array[:, -1] = predicted_scaled.flatten()  # Put predictions in target column
predicted_close = scaler.inverse_transform(dummy_array)[:, -1]

# Do the same for actual values
dummy_array[:, -1] = y_test.flatten()
actual_close = scaler.inverse_transform(dummy_array)[:, -1]

# Calculate metrics
mape = np.mean(np.abs((actual_close - predicted_close)/actual_close)) * 100
direction_true = np.diff(actual_close) > 0
direction_pred = np.diff(predicted_close) > 0
dpa = np.mean(direction_true == direction_pred) * 100

print(f"Mean Absolute Percentage Error: {mape:.2f}%")
print(f"Directional Prediction Accuracy: {dpa:.2f}%")

# -------------------------
# 6. Plot the results
# -------------------------
plt.figure(figsize=(14, 7))
plt.plot(actual_close, label='Actual Close Price', linewidth=2, color='blue')
plt.plot(predicted_close, label='Predicted Close Price', linestyle='--', linewidth=2, color='orange')
plt.title('HMM-LSTM Stock Price Forecast (Corrected)', fontsize=16)
plt.xlabel('Test Time Index', fontsize=14)
plt.ylabel('Close Price', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

# Add secondary plot for residuals
plt.figure(figsize=(14, 4))
residuals = actual_close - predicted_close
plt.plot(residuals, label='Residuals (Actual - Predicted)', color='green')
plt.axhline(y=0, color='red', linestyle='--')
plt.title('Prediction Residuals', fontsize=16)
plt.xlabel('Test Time Index', fontsize=14)
plt.ylabel('Price Difference', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

plt.show()
