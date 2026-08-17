import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt

# Load and prepare data
df = pd.read_csv('AAPL.csv', parse_dates=['Date'])
df.sort_values('Date', inplace=True)
df.set_index('Date', inplace=True)

# Create features
df['Return'] = df['Close'].pct_change()
df['LogReturn'] = np.log(df['Close']) - np.log(df['Close'].shift(1))
df['Volatility'] = (df['High'] - df['Low']) / df['Open']
df['VolumeChange'] = df['Volume'].pct_change()

# Create target (next day's close)
df['Target'] = df['Close'].shift(-1)

# Handle infinite/NaN values
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# Verify no infinite values remain
assert not np.isinf(df.select_dtypes(include=[np.number])).any().any()

# Use features for HMM
features = df[['Close', 'Return', 'LogReturn', 'Volatility', 'VolumeChange']].values
targets = df['Target'].values

# Split data (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    features, targets, test_size=0.2, shuffle=False
)

# Verify training data is clean
assert not np.isnan(X_train).any()
assert not np.isinf(X_train).any()

# Train HMM model
n_components = 5
model = GaussianHMM(
    n_components=n_components,
    covariance_type="diag",
    n_iter=1000,
    random_state=42
)

try:
    model.fit(X_train)
except ValueError as e:
    print("Error during fitting:", e)
    # Additional diagnostics
    print("NaN values in X_train:", np.isnan(X_train).any())
    print("Inf values in X_train:", np.isinf(X_train).any())
    print("Max values in X_train:", np.max(X_train, axis=0))
    print("Min values in X_train:", np.min(X_train, axis=0))
    raise

# Predict hidden states
hidden_states = model.predict(X_test)

# Get state means for Close price (first feature)
state_means = model.means_[:, 0]
predictions = state_means[hidden_states]

# Calculate metrics
mape = mean_absolute_percentage_error(y_test, predictions)
print(f"MAPE: {mape:.2f}%")

def calculate_dpa(actual, predicted):
    actual_dir = np.sign(np.diff(actual))
    predicted_dir = np.sign(np.diff(predicted))
    return np.mean(actual_dir == predicted_dir) * 100

dpa = calculate_dpa(y_test, predictions)
print(f"DPA: {dpa:.2f}%")

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(y_test, label='Actual Prices', linewidth=2)
plt.plot(predictions, label='HMM Predicted Prices', linestyle='--')
plt.title(f'HMM Price Prediction\nMAPE: {mape:.2f}% | DPA: {dpa:.2f}%')
plt.xlabel('Time Index')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()