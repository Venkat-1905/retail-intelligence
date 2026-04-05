import sys
sys.path.insert(0, '/home/venkat/retail-intelligence/models')

from feature_engineering import load_daily_sales, build_features, make_sequences
from lstm_forecaster      import train, save

print("=" * 50)
print("Phase 3 — ML Training Pipeline (Fixed Scale)")
print("=" * 50)

print("\n[1/2] Loading + engineering features...")
df = load_daily_sales()
print(f"      {len(df)} daily rows loaded")

top_stores = df['store_id'].value_counts().head(50).index
df = df[df['store_id'].isin(top_stores)]
print(f"      Using top 50 stores: {len(df)} rows")

features = build_features(df)
X, y     = make_sequences(features, seq_len=14)
print(f"      Sequences — X: {X.shape}, y: {y.shape}")

if len(X) == 0:
    print("Not enough data.")
else:
    print("\n[2/2] Training LSTM...")
    model, scaler, y_scaler = train(X, y)
    save(model, scaler, y_scaler,
         model_path="/home/venkat/retail-intelligence/models/lstm.pt",
         scaler_path="/home/venkat/retail-intelligence/models/scaler.pkl",
         y_scaler_path="/home/venkat/retail-intelligence/models/y_scaler.pkl")

print("\nPhase 3 complete!")