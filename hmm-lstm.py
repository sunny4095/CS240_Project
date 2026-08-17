import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GMMHMM
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# --------------------------
# Load and label data
# --------------------------
def load_stock_data(ticker='AAPL', start='2020-01-01', end='2023-01-01'):
    df = yf.download(ticker, start=start, end=end)
    df['frac_change'] = (df['Close'] - df['Open']) / df['Open']
    df['frac_high'] = (df['High'] - df['Open']) / df['Open']
    df['frac_low'] = (df['Open'] - df['Low']) / df['Open']
    df['label'] = (df['Close'] - df['Open']) / df['Open']
    δ = 0.001
    df['label'] = df['label'].apply(lambda x: 1 if x > δ else (-1 if x < -δ else 0))
    return df[['frac_change', 'frac_high', 'frac_low', 'label']].dropna()

# --------------------------
# Train HMM
# --------------------------
def train_hmm(X, n_states=4, n_mix=2):
    model = GMMHMM(n_components=n_states, n_mix=n_mix, covariance_type='diag', n_iter=100, min_covar=1e-4)
    model.fit(X)
    return model

# --------------------------
# Get soft state probabilities for each day
# --------------------------
def get_state_probabilities(model, X):
    logprob, posteriors = model.score_samples(X)
    return posteriors

# --------------------------
# LSTM Classifier
# --------------------------
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])

# --------------------------
# Prepare LSTM dataset
# --------------------------
def prepare_lstm_dataset(probs, labels, seq_len=10):
    X, y = [], []
    for i in range(seq_len, len(probs)):
        X.append(probs[i-seq_len:i])
        y.append(labels[i])
    return np.array(X), np.array(y)

# --------------------------
# Evaluate with Accuracy and DPA
# --------------------------
def evaluate(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    dpa = np.mean(np.sign(y_pred) == np.sign(y_true)) * 100
    print("Accuracy:", accuracy * 100)
    print("DPA:", dpa)
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("Classification Report:\n", classification_report(y_true, y_pred))

# --------------------------
# Main function
# --------------------------
def main():
    # 1. Load & preprocess data
    df = load_stock_data('AAPL')
    features = df[['frac_change', 'frac_high', 'frac_low']].values
    labels = df['label'].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    # 2. Train GMM-HMM
    hmm_model = train_hmm(X_scaled, n_states=4, n_mix=2)

    # 3. Get posterior state probabilities (soft features)
    posteriors = get_state_probabilities(hmm_model, X_scaled)

    # 4. Build sequence dataset for LSTM
    seq_len = 10
    X_seq, y_seq = prepare_lstm_dataset(posteriors, labels, seq_len)

    # 5. Train/test split
    split = int(0.8 * len(X_seq))
    X_train, y_train = X_seq[:split], y_seq[:split]
    X_test, y_test = X_seq[split:], y_seq[split:]

    # 6. LSTM training
    model = LSTMClassifier(input_size=X_seq.shape[2])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                             torch.tensor(y_train + 1, dtype=torch.long))  # map -1,0,1 → 0,1,2
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)

    for epoch in range(20):
        for xb, yb in train_loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # 7. Prediction
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_pred_logits = model(X_test_tensor)
        y_pred = torch.argmax(y_pred_logits, dim=1).numpy() - 1  # map back to -1,0,1

    # 8. Evaluation
    evaluate(y_test, y_pred)

    # Optional: Plot distribution
    plt.figure(figsize=(8, 4))
    plt.title("Predicted Class Distribution")
    plt.hist(y_pred, bins=[-1.5, -0.5, 0.5, 1.5], edgecolor='k', align='mid', rwidth=0.7)
    plt.xticks([-1, 0, 1])
    plt.xlabel("Predicted Movement")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
