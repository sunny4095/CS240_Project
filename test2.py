
import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn.hmm import GaussianHMM
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# ------------------ Data Preprocessing ------------------

def fetch_stock_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    df.dropna(inplace=True)
    df['fracChange'] = (df['Close'] - df['Open']) / df['Open']
    df['fracHigh'] = (df['High'] - df['Open']) / df['Open']
    df['fracLow'] = (df['Open'] - df['Low']) / df['Open']
    return df[['Open', 'Close', 'fracChange', 'fracHigh', 'fracLow']]

# ------------------ HMM Processing ------------------

def train_gmm_hmm(X, n_states=4):
    model = GaussianHMM(n_components=n_states, covariance_type='full', n_iter=100)
    model.fit(X)
    return model

def get_state_probabilities(model, X):
    return model.predict_proba(X)  # (T, n_states) matrix

# ------------------ Target Construction ------------------

def build_targets(close_prices, window=10, threshold=0.002):
    y = []
    for i in range(window, len(close_prices) - 1):
        delta = (close_prices[i + 1] - close_prices[i]) / close_prices[i]
        if delta > threshold:
            y.append(2)  # +1 -> class 2
        elif delta < -threshold:
            y.append(0)  # -1 -> class 0
        else:
            y.append(1)  # 0 -> class 1
    return np.array(y)

def build_lstm_input(state_probs, window=10):
    X = []
    for i in range(len(state_probs) - window):
        X.append(state_probs[i:i + window])
    return np.array(X)

# ------------------ LSTM Classifier ------------------

class MarketLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=32):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, 3)  # 3 classes: -1, 0, 1

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.linear(out[:, -1])

def train_lstm_classifier(X_train, y_train, epochs=30, lr=1e-3):
    model = MarketLSTM(input_size=X_train.shape[2])
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                            torch.tensor(y_train, dtype=torch.long))
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model.train()
    for epoch in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()
    return model

def evaluate_classifier(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_test, dtype=torch.float32))
        pred_classes = torch.argmax(logits, dim=1).numpy()
    acc = np.mean(pred_classes == y_test)
    print(f"Classification Accuracy: {acc*100:.2f}%")
    return pred_classes

# ------------------ Main Pipeline ------------------

if __name__ == "__main__":
    ticker = "AAPL"
    train_start, train_end = "2021-01-01", "2022-01-01"
    test_start, test_end = "2023-01-01", "2023-06-30"

    # 1. Load and preprocess
    train_df = fetch_stock_data(ticker, train_start, train_end)
    test_df = fetch_stock_data(ticker, test_start, test_end)
    full_df = pd.concat([train_df, test_df])
    features = full_df[['fracChange', 'fracHigh', 'fracLow']].values

    # 2. Train HMM
    hmm_model = train_gmm_hmm(features, n_states=4)
    state_probs = get_state_probabilities(hmm_model, features)

    # 3. Build dataset
    window = 10
    X = build_lstm_input(state_probs, window=window)
    y = build_targets(full_df['Close'].values, window=window)

    # 4. Split to train/test
    split_idx = len(train_df) - window - 1
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # 5. Train LSTM classifier
    model = train_lstm_classifier(X_train, y_train)

    # 6. Evaluate
    preds = evaluate_classifier(model, X_test, y_test)

    # 7. Plot predictions
    dates = test_df.index[window+1:]
    mapped_preds = np.array([[-1, 0, 1][p] for p in preds])
    plt.figure(figsize=(12,5))
    plt.plot(dates, mapped_preds, label='Predicted Movement')
    plt.title("Predicted Stock Movement: -1=Down, 0=Neutral, 1=Up")
    plt.grid(True)
    plt.legend()
    plt.show()
