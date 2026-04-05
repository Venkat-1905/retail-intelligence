import torch
import torch.nn as nn
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

SEQ_LEN     = 14
INPUT_SIZE  = 8
HIDDEN_SIZE = 64
NUM_LAYERS  = 2
EPOCHS      = 50
LR          = 0.001
PATIENCE    = 7

class LSTMForecaster(nn.Module):
    def __init__(self, input_size=INPUT_SIZE):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = HIDDEN_SIZE,
            num_layers  = NUM_LAYERS,
            batch_first = True,
            dropout     = 0.2
        )
        self.fc = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


def train(X, y):
    train_size = int(0.8 * len(X))
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    print(f"  Train: {len(X_train)} | Val: {len(X_val)}")

    # Scale features
    scaler  = StandardScaler()
    N, T, F = X_train.shape
    X_train_scaled = scaler.fit_transform(
        X_train.reshape(-1, F)).reshape(len(X_train), T, F)
    X_val_scaled   = scaler.transform(
        X_val.reshape(-1, F)).reshape(len(X_val), T, F)

    # Scale targets
    y_scaler       = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(
        y_train.reshape(-1, 1)).flatten()
    y_val_scaled   = y_scaler.transform(
        y_val.reshape(-1, 1)).flatten()

    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_scaled, dtype=torch.float32)
    X_val_t   = torch.tensor(X_val_scaled,   dtype=torch.float32)
    y_val_t   = torch.tensor(y_val_scaled,   dtype=torch.float32)

    model     = LSTMForecaster(input_size=F)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn   = nn.MSELoss()

    best_val_loss    = float('inf')
    patience_counter = 0
    best_state       = None

    for epoch in range(EPOCHS):
        # Train step
        model.train()
        optimizer.zero_grad()
        train_loss = loss_fn(model(X_train_t), y_train_t)
        train_loss.backward()
        optimizer.step()

        # Validation step
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val_t), y_val_t)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{EPOCHS} | "
                  f"Train Loss: {train_loss.item():.4f} | "
                  f"Val Loss: {val_loss.item():.4f}")

        # Early stopping
        if val_loss.item() < best_val_loss:
            best_val_loss    = val_loss.item()
            best_state       = {k: v.clone()
                                for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    # Restore best weights
    model.load_state_dict(best_state)

    # Evaluate on original scale
    model.eval()
    with torch.no_grad():
        val_preds_scaled = model(X_val_t).numpy()

    val_preds = y_scaler.inverse_transform(
        val_preds_scaled.reshape(-1, 1)).flatten()
    val_true  = y_val

    rmse = np.sqrt(mean_squared_error(val_true, val_preds))
    mae  = mean_absolute_error(val_true, val_preds)
    mape = np.mean(np.abs(
        (val_true - val_preds) /
        np.where(val_true == 0, 1, val_true)
    )) * 100

    print(f"\n  Validation Metrics (original scale):")
    print(f"    RMSE : {rmse:.2f}")
    print(f"    MAE  : {mae:.2f}")
    print(f"    MAPE : {mape:.2f}%")

    return model, scaler, y_scaler


def save(model, scaler, y_scaler,
         model_path="models/lstm.pt",
         scaler_path="models/scaler.pkl",
         y_scaler_path="models/y_scaler.pkl"):
    torch.save(model.state_dict(), model_path)
    joblib.dump(scaler,   scaler_path)
    joblib.dump(y_scaler, y_scaler_path)
    print(f"\n  Model saved    → {model_path}")
    print(f"  Scaler saved   → {scaler_path}")
    print(f"  Y-Scaler saved → {y_scaler_path}")


def load(model_path, scaler_path, y_scaler_path, input_size=INPUT_SIZE):
    """Load model + both scalers for inference."""
    model = LSTMForecaster(input_size=input_size)
    model.load_state_dict(torch.load(model_path, map_location="cpu",
                                     weights_only=True))
    model.eval()
    scaler   = joblib.load(scaler_path)
    y_scaler = joblib.load(y_scaler_path)
    return model, scaler, y_scaler


def predict(model, scaler, y_scaler, X_raw):
    """Run inference and return predictions on original scale."""
    N, T, F  = X_raw.shape
    X_scaled = scaler.transform(X_raw.reshape(-1, F)).reshape(N, T, F)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        preds_scaled = model(X_tensor).numpy()
    return y_scaler.inverse_transform(
        preds_scaled.reshape(-1, 1)).flatten()