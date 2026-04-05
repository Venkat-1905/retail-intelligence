import os

# ── Base paths ────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# ── Database config ───────────────────────────────────
DB_CONFIG = {
    "host":     os.getenv("DB_HOST",     "localhost"),
    "port":     int(os.getenv("DB_PORT", 5432)),
    "database": os.getenv("DB_NAME",     "retail_db"),
    "user":     os.getenv("DB_USER",     "retail_user"),
    "password": os.getenv("DB_PASSWORD", "retail_pass"),
}

# ── Model paths ───────────────────────────────────────
MODEL_PATH    = os.path.join(MODELS_DIR, "lstm.pt")
SCALER_PATH   = os.path.join(MODELS_DIR, "scaler.pkl")
Y_SCALER_PATH = os.path.join(MODELS_DIR, "y_scaler.pkl")

# ── Model hyperparameters ─────────────────────────────
SEQ_LEN    = 14
INPUT_SIZE = 8

# ── Business logic ────────────────────────────────────
LEAD_TIME    = 3    # days for inventory reorder calculation
SAFETY_STOCK_FACTOR = 0.2  # 20% of rolling mean

# ── Dataset date range ────────────────────────────────
DATASET_START = "2013-01-01"
DATASET_END   = "2015-07-31"
FORECAST_DAYS = 7
