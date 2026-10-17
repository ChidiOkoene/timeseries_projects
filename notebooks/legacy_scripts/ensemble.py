import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, LSTM, Conv1D, 
                                     Bidirectional, Dropout, BatchNormalization, Input)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from xgboost import XGBRegressor
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle
import matplotlib.pyplot as plt
import os

# ====================== Data Preprocessing ======================
class TimeSeriesPipeline:
    def __init__(self, seq_length=30, test_size=0.2):
        self.seq_length = seq_length
        self.test_size = test_size
        self.scalers = {}

    def preprocess_data(self, df, fit_scalers=True, save_scalers=True, scalers_path="scalers.pkl"):
        print("Starting data preprocessing...")
        unscaled_data = df.copy()

        # Define feature categories
        price_cols = ['open', 'high', 'low', 'close', 'av_pr', 'diff', '7_day_SMA', 
                     '30_day_SMA', '7_day_EMA', '30_day_EMA', '12_day_EMA', '26_day_EMA', '20_day_SMA']
        volatility_cols = ['20_day_STD', 'Upper_Band', 'Lower_Band']
        momentum_cols = ['RSI', 'MACD', 'Signal_Line']
        lag_cols = ['lag_1', 'lag_2', 'lag_3']

        # Load or fit scalers
        if not fit_scalers:
            with open(scalers_path, 'rb') as f:
                self.scalers = pickle.load(f)
        else:
            self._fit_scalers(df, price_cols, volatility_cols, momentum_cols, lag_cols)
            if save_scalers:
                with open(scalers_path, 'wb') as f:
                    pickle.dump(self.scalers, f)

        # Transform data
        df = self._transform_features(df, price_cols, volatility_cols, momentum_cols, lag_cols)

        # Create sequences
        X_scaled, y_scaled = self.create_sequences(df)
        X_unscaled, y_unscaled = self.create_sequences(unscaled_data)

        # Split data
        X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = self.time_based_split(X_scaled, y_scaled)
        X_train_unscaled, X_test_unscaled, y_train_unscaled, y_test_unscaled = self.time_based_split(X_unscaled, y_unscaled)

        print(f"Data preprocessing finished. Training samples: {len(X_train_scaled)}, Test samples: {len(X_test_scaled)}")
        return (X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled,
                X_train_unscaled, X_test_unscaled, y_train_unscaled, y_test_unscaled)

    def _fit_scalers(self, df, price_cols, volatility_cols, momentum_cols, lag_cols):
        """Initialize and fit all scalers"""
        self.scalers = {
            'price': RobustScaler().fit(df[price_cols]),
            'volume': RobustScaler().fit(np.log1p(df[['volume']])),
            'volatility': StandardScaler().fit(df[volatility_cols]),
            'momentum': MinMaxScaler().fit(df[momentum_cols]),
            'lag': RobustScaler().fit(df[lag_cols])
        }

    def _transform_features(self, df, price_cols, volatility_cols, momentum_cols, lag_cols):
        """Apply feature transformations"""
        df[price_cols] = self.scalers['price'].transform(df[price_cols])
        df['volume'] = self.scalers['volume'].transform(np.log1p(df[['volume']]))
        df[volatility_cols] = self.scalers['volatility'].transform(df[volatility_cols])
        df[momentum_cols] = self.scalers['momentum'].transform(df[momentum_cols])
        df[lag_cols] = self.scalers['lag'].transform(df[lag_cols])
        return df

    def create_sequences(self, data):
        X, y = [], []
        for i in range(len(data) - self.seq_length):
            seq = data.iloc[i:i+self.seq_length].values
            target = data.iloc[i+self.seq_length]['close']
            X.append(seq)
            y.append(target)
        return np.array(X), np.array(y)

    def time_based_split(self, X, y):
        split_idx = int(len(X) * (1 - self.test_size))
        return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]

    def inverse_transform_predictions(self, pred_scaled, feature_name='close'):
        if feature_name == 'close':
            scaler = self.scalers['price']
            col_index = 3  # Position of 'close' in price_cols
            return pred_scaled * scaler.scale_[col_index] + scaler.center_[col_index]
        raise ValueError(f"Unsupported feature for inversion: {feature_name}")

# ====================== Model Definitions ======================
class HybridCNNRNN:
    def __init__(self, input_shape):
        if input_shape is None:
            raise ValueError("Input shape must be provided for HybridCNNRNN")
        self.model = self._build_model(input_shape)

    def _build_model(self, input_shape):
        inputs = Input(shape=input_shape)
        x = Conv1D(64, 3, dilation_rate=2, activation='relu')(inputs)
        x = Dropout(0.4)(x)
        x = BatchNormalization()(x)
        x = Bidirectional(LSTM(64, return_sequences=True))(x)
        x = Bidirectional(LSTM(32))(x)
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.2)(x)
        outputs = Dense(1)(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(0.001),
                    loss=tf.keras.losses.Huber(delta=1.5),  # Changed to Huber
                    metrics=['mae', 'mse'])
        return model

    def train(self, X_train, y_train, validation_data, epochs=50):
        self.model.fit(X_train, y_train, 
                      validation_data=validation_data,
                      epochs=epochs,
                      batch_size=64,
                      callbacks=[EarlyStopping(patience=7)],
                      verbose=2)

class SimpleXGBoost:
    def __init__(self):
        self.model = XGBRegressor(n_estimators=100, learning_rate=0.01)

    def train(self, X_train, y_train):
        X_train = X_train.reshape(X_train.shape[0], -1)
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X.reshape(X.shape[0], -1))

class MetaModel:
    def __init__(self, input_shape):
        self.model = self._build_model(input_shape)

    def _build_model(self, input_shape):
        model = Sequential([
            Dense(64, activation='relu', input_shape=input_shape),
            Dense(64, activation='relu'),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dense(64, activation='relu'),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dense(64, activation='relu'),
            Dense(64, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer=Adam(0.00008), 
                    loss=tf.keras.losses.Huber(delta=1.5),  # Changed to Huber
                    metrics=['mae', 'mse'])
        return model

    def train(self, X_train, y_train, epochs=1000):
        self.model.fit(X_train, y_train, 
                      epochs=epochs, 
                      batch_size=7,
                      callbacks=[ReduceLROnPlateau(factor=0.5, patience=5)], 
                      verbose=2)

# ====================== Ensemble System ======================
class TemporalEnsemble:
    def __init__(self, pipeline, models):
        self.pipeline = pipeline
        self.models = models
        self.full_data = None
        self.scalers_path = "scalers.pkl"
        self.model_dir = "saved_models"  # Directory to save models

        # Create directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)

    def save_models(self):
        """Save all models and pipeline to disk"""
        print("\nSaving models and pipeline...")
        
        # Save TensorFlow/Keras models
        self.models['cnn_rnn'].model.save(os.path.join(self.model_dir, "cnn_rnn_model.keras"))
        self.models['cnn_rnn_all_data'].model.save(os.path.join(self.model_dir, "cnn_rnn_all_data_model.keras"))
        self.models['meta'].model.save(os.path.join(self.model_dir, "meta_model.keras"))

        # Save XGBoost model
        self.models['xgb'].model.save_model(os.path.join(self.model_dir, "xgb_model.json"))

        # Save pipeline with scalers
        with open(os.path.join(self.model_dir, "pipeline.pkl"), "wb") as f:
            pickle.dump(self.pipeline, f)

        print(f"Models and pipeline saved to {self.model_dir}")

    def load_models(self):
        """Load all models and pipeline from disk"""
        print("\nLoading models and pipeline...")
        
        # Load TensorFlow/Keras models
        self.models['cnn_rnn'].model = tf.keras.models.load_model(
            os.path.join(self.model_dir, "cnn_rnn_model")
        )
        self.models['cnn_rnn_all_data'].model = tf.keras.models.load_model(
            os.path.join(self.model_dir, "cnn_rnn_all_data_model")
        )
        self.models['meta'].model = tf.keras.models.load_model(
            os.path.join(self.model_dir, "meta_model")
        )

        # Load XGBoost model
        self.models['xgb'].model = XGBRegressor()
        self.models['xgb'].model.load_model(
            os.path.join(self.model_dir, "xgb_model.json")
        )

        # Load pipeline with scalers
        with open(os.path.join(self.model_dir, "pipeline.pkl"), "rb") as f:
            self.pipeline = pickle.load(f)

        print(f"Models and pipeline loaded from {self.model_dir}")

    def initialize_models(self, input_shape):
        self.models = {
            'cnn_rnn': HybridCNNRNN(input_shape),
            'xgb': SimpleXGBoost(),
            'cnn_rnn_all_data': HybridCNNRNN(input_shape),
            'meta': MetaModel((4,))
        }

    def train_initial_models(self, X_train, y_train, X_val, y_val, X_all, y_all):
        # Base CNN-RNN model
        print("\nTraining base CNN-RNN...")
        self.models['cnn_rnn'].train(X_train, y_train, (X_val, y_val))
        
        # XGBoost model trained on all data
        print("\nTraining XGBoost on all data...")
        self.models['xgb'].train(X_all, y_all)
        
        # CNN-RNN trained on all data
        print("\nTraining full-data CNN-RNN...")
        self.models['cnn_rnn_all_data'].train(X_all, y_all, (X_all[:1], y_all[:1]))
        
        # Prepare meta-model data
        meta_X_train = self._prepare_meta_input(X_train, X_all, y_train)
        print("\nTraining Meta Model...")
        self.models['meta'].train(meta_X_train, y_train)

    def _prepare_meta_input(self, X_base, X_all, y_true):
        # Generate predictions from all base models
        base_preds = [
            self.models['cnn_rnn'].model.predict(X_base).flatten(),
            self.models['xgb'].predict(X_base),
            [x[-1, 3] for x in X_base],  # Naive forecast from scaled data
            self.models['cnn_rnn_all_data'].model.predict(X_base).flatten()
        ]
        return np.column_stack(base_preds)

    def predict_next_day(self, new_data):
        # Prepare input sequence
        seq_scaled, seq_unscaled = self._prepare_input_sequence(new_data)
        
        # Generate predictions
        base_preds = [
            self.models['cnn_rnn'].model.predict(seq_scaled).item(),
            self.models['xgb'].predict(seq_scaled).item(),
            seq_scaled[-1, 3],  # Naive forecast from raw data
            self.models['cnn_rnn_all_data'].model.predict(seq_scaled).item()
        ]
        
        # Get final prediction
        meta_pred = self.models['meta'].model.predict([base_preds]).item()
        return self.pipeline.inverse_transform_predictions([meta_pred], 'close')[0]

    def _prepare_input_sequence(self, new_data):
        # Update and maintain 30-day window
        self.full_data = pd.concat([self.full_data, new_data]).iloc[-30:]
        
        # Process data using existing scalers
        processed = self.pipeline.preprocess_data(
            self.full_data.copy(), 
            fit_scalers=False, 
            save_scalers=False
        )
        return processed[0][-1:], self.full_data.iloc[-30:].values[-1:]

    def update_models(self, new_data):
        # Update data storage
        self.full_data = pd.concat([self.full_data, new_data])
        
        # Process updated data
        X_all, y_all, _, _, _, _, _, _ = self.pipeline.preprocess_data(
            self.full_data, 
            fit_scalers=False
        )
        
        # Update models incrementally
        print("\nUpdating models with new data...")
        self.models['cnn_rnn'].model.train_on_batch(X_all[-1:], y_all[-1:])
        self.models['xgb'].train(X_all, y_all)
        self.models['cnn_rnn_all_data'].model.train_on_batch(X_all[-1:], y_all[-1:])

# ====================== Main Execution ======================
if __name__ == "__main__":
    # Initialize pipeline and load data
    pipeline = TimeSeriesPipeline(seq_length=30)
    df = pd.read_csv('xrpusdt_daily_dataset_with_features.csv')
    
    # Preprocess data
    (X_train, X_test, y_train, y_test,
     X_train_unscaled, X_test_unscaled, y_test_unscaled, _) = pipeline.preprocess_data(df)
    input_shape = X_train.shape[1:]

    # Prepare combined dataset for full-data models
    X_all = np.concatenate([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])

    # Initialize ALL models
    models = {
        'cnn_rnn': HybridCNNRNN(input_shape, huber_delta=1.3),
        'cnn_rnn_all_data': HybridCNNRNN(input_shape, huber_delta=1.3),  # Must be present
        'meta': MetaModel(input_shape=(4,)),
        'xgb': SimpleXGBoost()
        
        
    }

    # Initialize ensemble system
    ensemble = TemporalEnsemble(pipeline, models)
    ensemble.initialize_models(input_shape)
    ensemble.full_data = df.copy()

    # Train initial models
    ensemble.train_initial_models(X_train, y_train, X_test, y_test, X_all, y_all)
    ensemble.save_models()

        

    # Final evaluation
    test_preds = ensemble.models['meta'].model.predict(
        ensemble._prepare_meta_input(X_test, X_all, y_test)
    )
    final_preds = pipeline.inverse_transform_predictions(test_preds.flatten(), 'close')
    
    metrics = {
        'MAE': mean_absolute_error(y_test_unscaled, final_preds),
        'RMSE': np.sqrt(mean_squared_error(y_test_unscaled, final_preds)),
        'MAPE': np.mean(np.abs((y_test_unscaled - final_preds)/y_test_unscaled)) * 100
    }
    
    print("\nFinal Performance Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # Visualization
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_unscaled, label='Actual')
    plt.plot(final_preds, label='Predicted')
    plt.title("Actual vs Predicted Prices")
    plt.legend()
    plt.show()
    plt.savefig('predviz.jpeg')