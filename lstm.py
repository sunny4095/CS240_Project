import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

# Load and preprocess data
df = pd.read_csv('AAPL.csv', parse_dates=['Date'])
df.sort_values('Date', inplace=True)
df.set_index('Date', inplace=True)

# Create target variable (next day's close price)
df['Target'] = df['Close'].shift(-1)
df.dropna(inplace=True)

# Select features - using only price and volume data
feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
target_col = 'Target'

# Scale data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[feature_cols + [target_col]])

# Create sequences
n_lags = 20  # Number of past days to use for prediction

def create_sequences(data, n_lags):
    X, y = [], []
    for i in range(n_lags, len(data)-1):  # -1 because we're predicting next step
        X.append(data[i-n_lags:i, :-1])  # All columns except last as features
        y.append(data[i, -1])  # Last column as target
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data, n_lags)

# Split data
split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Build LSTM model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(n_lags, len(feature_cols))),
    LSTM(64),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Train model
history = model.fit(X_train, y_train, 
                   epochs=50, 
                   batch_size=32, 
                   validation_split=0.2,
                   verbose=1)

# Make predictions
predicted_scaled = model.predict(X_test)

# Inverse transform predictions
# Create dummy array for inverse transform
dummy_features = np.zeros((len(predicted_scaled), len(feature_cols)+1))
dummy_features[:, -1] = predicted_scaled.flatten()
predicted_prices = scaler.inverse_transform(dummy_features)[:, -1]

# Get actual prices
dummy_features[:, -1] = y_test.flatten()
actual_prices = scaler.inverse_transform(dummy_features)[:, -1]

# Calculate MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mape = mean_absolute_percentage_error(actual_prices, predicted_prices)

# Calculate DPA
def directional_accuracy(y_true, y_pred):
    # Calculate actual and predicted directions
    actual_dir = np.sign(np.diff(y_true))
    predicted_dir = np.sign(np.diff(y_pred))
    # Align arrays (diff reduces length by 1)
    min_length = min(len(actual_dir), len(predicted_dir))
    return np.mean(actual_dir[:min_length] == predicted_dir[:min_length]) * 100

dpa = directional_accuracy(actual_prices, predicted_prices)

# Print results
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
print(f"Directional Prediction Accuracy (DPA): {dpa:.2f}%")

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(actual_prices, label='Actual Prices', linewidth=2)
plt.plot(predicted_prices, label='Predicted Prices', linestyle='--')
plt.title(f'Price Prediction\nMAPE: {mape:.2f}% | DPA: {dpa:.2f}%')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

# Plot training history
plt.figure(figsize=(12, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Training History')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()