import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from tqdm import tqdm
from hmmlearn.hmm import GMMHMM
from sklearn.preprocessing import StandardScaler

# 1. Load and compute features
def load_stock_data(ticker='AAPL', start='2020-01-01', end='2023-01-01'):
    df = yf.download(ticker, start=start, end=end)
    df['frac_change'] = (df['Close'] - df['Open']) / df['Open']
    df['frac_high'] = (df['High'] - df['Open']) / df['Open']
    df['frac_low'] = (df['Open'] - df['Low']) / df['Open']
    return df[['Open', 'Close', 'frac_change', 'frac_high', 'frac_low']].dropna()

# 2. Prepare sequences for GMMHMM (rolling window)
def prepare_sequences_real(df, window=10):
    data = df[['frac_change', 'frac_high', 'frac_low']].values
    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    sequences = []
    for i in range(len(data) - window + 1):
        seq = data[i:i+window]
        sequences.append(seq)

    X = np.vstack(sequences)
    lengths = [window] * len(sequences)
    return X, lengths, scaler

# 3. Train the model
def train_hmm(X, lengths, n_components=4, n_mix=4, n_iter=50):
    model = GMMHMM(
        n_components=4,
        n_mix=2,
        covariance_type='diag',
        min_covar=1e-5,      # Add this to prevent zero-variance
        n_iter=50,
        verbose=True
    )

    model.fit(X, lengths)
    return model

# 4. Predict next observation (by likelihood ranking)
def predict_next_obs(model, history, scaler):
    # Try predicting the next observation from multiple candidates
    means = np.linspace(-2, 2, 20)
    candidates = np.array(np.meshgrid(means, means, means)).T.reshape(-1, 3)

    max_prob = -np.inf
    best_obs = None
    for obs in candidates:
        seq = np.vstack([history, obs])
        try:
            prob = model.score(seq)
            if prob > max_prob:
                max_prob = prob
                best_obs = obs
        except:
            continue

    return scaler.inverse_transform([best_obs])[0] if best_obs is not None else None

# 5. Evaluate
def evaluate_predictions(predicted, actual, opens):
    predicted = np.array(predicted)
    actual = np.array(actual)
    opens = np.array(opens)

    mape = np.mean(np.abs(predicted - actual) / np.abs(actual)) * 100
    dpa = np.mean(np.sign(predicted - opens) == np.sign(actual - opens)) * 100
    return mape, dpa

# 6. Main pipeline
def main():
    ticker = 'AAPL'
    df = load_stock_data(ticker)
    print("Loading data done")
    train_df = df.iloc[:200]
    test_df = df.iloc[200:250]

    X, lengths, scaler = prepare_sequences_real(train_df, window=10)
    print("Preparation of sequences done")
    model = train_hmm(X, lengths)
    print("Training done")

    predicted_close = []
    actual_close = []
    open_prices = []

    window = 9  # history length
    test_features = test_df[['frac_change', 'frac_high', 'frac_low']].values
    test_features_scaled = scaler.transform(test_features)

    for i in tqdm(range(window, len(test_df))):
        history = test_features_scaled[i-window:i]
        pred = predict_next_obs(model, history, scaler)
        if pred is not None:
            # Approximate the closing price using: close = open * (1 + frac_change)
            predicted_close_price = test_df['Open'].iloc[i] * (1 + pred[0])
            predicted_close.append(predicted_close_price)
            actual_close.append(test_df['Close'].iloc[i])
            open_prices.append(test_df['Open'].iloc[i])

    print("Prediction done")

    # Evaluation
    mape, dpa = evaluate_predictions(predicted_close, actual_close, open_prices)
    print(f"\nMAPE: {mape:.2f}%")
    print(f"DPA : {dpa:.2f}%")

    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(actual_close, label='Actual Close')
    plt.plot(predicted_close, label='Predicted Close')
    plt.title(f"{ticker} HMM-GMM Predicted vs Actual Close Prices")
    plt.xlabel('Test Day')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
