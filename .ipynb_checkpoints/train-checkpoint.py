import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# 1. Configuration
config = {
    'data_path': '/mnt/data/batch_results.csv',
    'features': [
        'klne.zqmd.0208',
        'klne.zqmf.0209',
        'klne.zqmd.0214',
        'klne.zqmf.0215'
    ],
    'target_col': 'transmission_efficiency',
    'test_size': 0.2,
    'random_state': 42,
    'batch_size': 32,
    'epochs': 50,
    'learning_rate': 1e-3,
    'loss': {
        'eps': 1e-3
    }
}


# 2. Data loading & preprocessing functions
def load_data(path, features, target_col):
    df = pd.read_csv(path)
    X = df[features].astype(np.float32).values
    y = df[target_col].astype(np.float32).values
    return X, y

def compute_sample_weights(y):
    # Compute frequency of each exact efficiency
    freqs = pd.Series(y).value_counts()
    # Weight inversely by sqrt of frequency, normalize to mean=1
    weight_map = (1.0 / np.sqrt(freqs)).to_dict()
    weights = pd.Series(y).map(weight_map).astype(np.float32)
    return (weights / weights.mean()).values

def transform_targets(y, eps):
    # Scale 0–100 -> 0–1, clip, then logit
    y_scaled = y / 100.0
    y_clipped = np.clip(y_scaled, eps, 1 - eps)
    return np.log(y_clipped / (1 - y_clipped)).astype(np.float32)


# 3. Model definition
def build_model(input_dim, learning_rate):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(input_dim,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, name='logit_output')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss='mse'
    )
    return model


# 4. Main training pipeline
# Load and preprocess
X, y = load_data(config['data_path'], config['features'], config['target_col'])
weights = compute_sample_weights(y)
y_trans = transform_targets(y, config['loss']['eps'])

# Split
X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
    X, y_trans, weights,
    test_size=config['test_size'],
    random_state=config['random_state']
)

# Build and inspect model
model = build_model(input_dim=X.shape[1], learning_rate=config['learning_rate'])
model.summary()

# Train
history = model.fit(
    X_train,
    y_train,
    sample_weight=w_train,
    validation_data=(X_test, y_test, w_test),
    batch_size=config['batch_size'],
    epochs=config['epochs']
)

# Example of inverting the logit for predictions
y_pred_logit = model.predict(X_test)
y_pred = 1 / (1 + np.exp(-y_pred_logit))  # back to [0,1]
y_pred_efficiency = y_pred.flatten() * 100  # back to [0,100]

print(f"Sample predictions (first 5): {y_pred_efficiency[:5]}")

