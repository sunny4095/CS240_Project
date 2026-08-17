
import streamlit as st
import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# st.set_option('deprecation.showPyplotGlobalUse', False)

st.title("HMM-LSTM Stock Forecasting")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, parse_dates=['Date'])
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

    if 'model' not in st.session_state:
        st.session_state.hmm_model = None
        st.session_state.model = None
        st.session_state.scaler_features = None
        st.session_state.scaler_target = None
        st.session_state.X_train = st.session_state.X_test = st.session_state.y_train = st.session_state.y_test = st.session_state.test_open = None
    
    feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume',
                    'LogReturn', 'Volatility', 'PriceChange', 'VolumeChange',
                    'HMM_State', 'State_Mean_LogReturn', 'State_Mean_Volatility', 'State_Mean_VolumeChange']
    target_col = 'Close'
    n_lags = 45 

    if st.button("Train Model"):
        hmm_features = df[['LogReturn', 'Volatility', 'VolumeChange']].values

        valid_idx = ~np.isnan(hmm_features).any(axis=1) & ~np.isinf(hmm_features).any(axis=1)
        df = df[valid_idx]
        hmm_features = hmm_features[valid_idx]

        n_hidden_states = 5
        st.session_state.hmm_model = GaussianHMM(n_components=n_hidden_states, covariance_type='diag', n_iter=1000)
        st.session_state.hmm_model.fit(hmm_features)

        hidden_states = st.session_state.hmm_model.predict(hmm_features)
        state_means = st.session_state.hmm_model.means_

        df['HMM_State'] = hidden_states
        df['State_Mean_LogReturn'] = state_means[hidden_states, 0]
        df['State_Mean_Volatility'] = state_means[hidden_states, 1]
        df['State_Mean_VolumeChange'] = state_means[hidden_states, 2]

        # Scale data
        st.session_state.scaler_features = MinMaxScaler()
        st.session_state.scaled_features = st.session_state.scaler_features.fit_transform(df[feature_cols])

        st.session_state.scaler_target = MinMaxScaler()
        st.session_state.scaled_target = st.session_state.scaler_target.fit_transform(df[[target_col]])

        scaled_data = np.hstack([st.session_state.scaled_features, st.session_state.scaled_target])

        def create_sequences(data, features, target, n_lags):
            X, y = [], []
            for i in range(n_lags, len(data)):
                X.append(features[i-n_lags:i])
                y.append(target[i, 0])
            return np.array(X), np.array(y)

        X, y = create_sequences(scaled_data, st.session_state.scaled_features, st.session_state.scaled_target, n_lags)

        split_idx = int(0.98 * len(X))
        st.session_state.X_train, st.session_state.X_test = X[:split_idx], X[split_idx:]
        st.session_state.y_train, st.session_state.y_test = y[:split_idx], y[split_idx:]
        st.session_state.test_open = df['Open'][split_idx:]

        st.session_state.model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(n_lags, len(feature_cols))),
            LSTM(50),
            Dense(25, activation='tanh'),
            Dense(1)
        ])

        progress_bar = st.progress(0)
        status_text = st.empty()

        class StreamlitCallback(tf.keras.callbacks.Callback):
            def __init__(self, total_epochs):
                self.total_epochs = total_epochs
                self.epoch = 0

            def on_epoch_end(self, epoch, logs=None):
                self.epoch += 1
                progress = self.epoch / self.total_epochs
                progress_bar.progress(min(int(progress * 100), 100))
                status_text.text(f"Epoch {self.epoch}/{self.total_epochs} - Loss: {logs.get('loss'):.4f}")

        num_epochs = 100

        st.session_state.model.compile(optimizer='adam', loss='mean_squared_error')
        st.session_state.model.fit(st.session_state.X_train, st.session_state.y_train, epochs=num_epochs, batch_size=32, validation_split=0.1, verbose=1, callbacks=[StreamlitCallback(num_epochs)])

        st.success("Training completed!")

    if st.button("Predict and Evaluate"):
        predicted_scaled = st.session_state.model.predict(st.session_state.X_test)
        predicted_close = st.session_state.scaler_target.inverse_transform(predicted_scaled)
        actual_close = st.session_state.scaler_target.inverse_transform(st.session_state.y_test.reshape(-1, 1))

        bias = np.mean(actual_close - predicted_close)
        predicted_close += bias

        mape = np.mean(np.abs((actual_close - predicted_close)/actual_close)) * 100
        direction_true = np.diff(actual_close.flatten()) > 0
        direction_pred = np.diff(predicted_close.flatten()) > 0
        dpa = np.mean(direction_true == direction_pred) * 100

        st.write(f"**MAPE:** {mape:.2f}")
        st.write(f"**DPA:** {dpa:.2f}")

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(actual_close, label='Actual Close Price', linewidth=2)
        ax.plot(predicted_close, label='Predicted Close Price', linestyle='--')
        ax.legend()
        ax.set_title('HMM-LSTM Stock Price Forecast')
        ax.set_xlabel('Test Time Index')
        ax.set_ylabel('Close Price')
        ax.grid(True)
        fig.tight_layout()
        st.pyplot(fig)

