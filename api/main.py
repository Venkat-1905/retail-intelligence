import sys
import logging
import time
import numpy as np
import torch
import joblib
import psycopg2
import pandas as pd
from contextlib import asynccontextmanager
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models'))
from lstm_forecaster import LSTMForecaster
from api.config import (
    DB_CONFIG, MODEL_PATH, SCALER_PATH, Y_SCALER_PATH,
    SEQ_LEN, INPUT_SIZE, LEAD_TIME, SAFETY_STOCK_FACTOR
)

# ── Logging ───────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)

# ── Global model state ────────────────────────────────
MODEL    = None
SCALER   = None
Y_SCALER = None

# ── Pydantic schemas ──────────────────────────────────
class PredictionRequest(BaseModel):
    product_id: int = Field(..., gt=0, description="Product ID")
    store_id:   int = Field(..., gt=0, description="Store ID")

class BatchPredictionRequest(BaseModel):
    requests: List[PredictionRequest]

class PredictionResponse(BaseModel):
    product_id:       int
    store_id:         int
    predicted_demand: float
    unit:             str = "units/day"
    latency_ms:       float

class AnomalyItem(BaseModel):
    store_id:     int
    product_id:   int
    anomaly_flag: bool
    z_score:      Optional[float]
    detected_at:  str

class InventoryItem(BaseModel):
    product_id:     int
    current_demand: float
    reorder_point:  float
    safety_stock:   float
    status:         str

# ── Lifespan ──────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL, SCALER, Y_SCALER
    log.info("Loading LSTM model and scalers...")
    try:
        MODEL = LSTMForecaster(input_size=INPUT_SIZE)
        MODEL.load_state_dict(torch.load(
            MODEL_PATH, map_location="cpu", weights_only=True
        ))
        MODEL.eval()
        SCALER   = joblib.load(SCALER_PATH)
        Y_SCALER = joblib.load(Y_SCALER_PATH)
        log.info("Model + scalers loaded successfully")
    except Exception as e:
        log.error(f"Failed to load model: {e}")
    yield
    log.info("Shutting down...")

app = FastAPI(
    title="Retail Intelligence API",
    description="Real-time demand forecasting, anomaly detection & inventory optimization",
    version="2.0.0",
    lifespan=lifespan
)

# ── DB helper ─────────────────────────────────────────
def get_conn():
    try:
        return psycopg2.connect(**DB_CONFIG)
    except Exception as e:
        log.error(f"DB connection failed: {e}")
        raise HTTPException(status_code=503, detail="Database unavailable")

# ── Feature builder ───────────────────────────────────
def build_sequence(store_id: int) -> np.ndarray:
    """
    Builds exact same 8-feature sequence used in training:
    total_quantity, lag_1, lag_7, rolling_7,
    avg_price, day_of_week, is_weekend, month
    """
    conn = get_conn()
    cur  = conn.cursor()
    cur.execute("""
        SELECT sale_date, total_quantity, avg_price
        FROM aggregated_sales
        WHERE store_id = %s
        ORDER BY sale_date DESC
        LIMIT 21
    """, (store_id,))
    rows = cur.fetchall()
    cur.close()
    conn.close()

    if len(rows) < SEQ_LEN:
        raise HTTPException(
            status_code=404,
            detail=f"Need at least {SEQ_LEN} days of data "
                   f"for store {store_id}. Got {len(rows)}."
        )

    df = pd.DataFrame(rows, columns=["sale_date", "total_quantity", "avg_price"])
    df = df.sort_values("sale_date").reset_index(drop=True)
    df["sale_date"]   = pd.to_datetime(df["sale_date"])
    df["lag_1"]       = df["total_quantity"].shift(1)
    df["lag_7"]       = df["total_quantity"].shift(7)
    df["rolling_7"]   = df["total_quantity"].shift(1).rolling(7).mean()
    df["day_of_week"] = df["sale_date"].dt.dayofweek
    df["is_weekend"]  = df["day_of_week"].isin([5, 6]).astype(int)
    df["month"]       = df["sale_date"].dt.month

    df = df.dropna().tail(SEQ_LEN)

    if len(df) < SEQ_LEN:
        raise HTTPException(
            status_code=422,
            detail=f"Not enough clean rows after feature engineering "
                   f"for store {store_id}."
        )

    feature_cols = [
        "total_quantity", "lag_1", "lag_7", "rolling_7",
        "avg_price", "day_of_week", "is_weekend", "month"
    ]
    return df[feature_cols].values.astype(np.float32)

# ── Inference helper ──────────────────────────────────
def run_inference(sequence: np.ndarray) -> float:
    if MODEL is None or SCALER is None or Y_SCALER is None:
        raise HTTPException(status_code=503,
                            detail="Model not loaded")
    X        = sequence.reshape(1, SEQ_LEN, INPUT_SIZE)
    N, T, F  = X.shape
    X_scaled = SCALER.transform(X.reshape(-1, F)).reshape(N, T, F)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    with torch.no_grad():
        pred_scaled = MODEL(X_tensor).numpy()
    pred = Y_SCALER.inverse_transform(
        pred_scaled.reshape(-1, 1)).flatten()[0]
    return float(pred)

# ═════════════════════════════════════════════════════
# ENDPOINTS
# ═════════════════════════════════════════════════════

# ── General ───────────────────────────────────────────
@app.get("/", tags=["General"])
def root():
    return {"message": "Retail Intelligence API v2.0 ✅"}

@app.get("/health", tags=["General"])
def health():
    return {
        "status":         "ok",
        "model":          "LSTM v2",
        "input_size":     INPUT_SIZE,
        "seq_len":        SEQ_LEN,
        "model_loaded":   MODEL    is not None,
        "scaler_loaded":  SCALER   is not None,
        "y_scaler_loaded": Y_SCALER is not None
    }

@app.get("/sales-summary", tags=["General"])
def sales_summary():
    conn = get_conn()
    cur  = conn.cursor()
    cur.execute("""
        SELECT
            COUNT(*)                   AS total_records,
            SUM(total_quantity)        AS total_units,
            MAX(sale_date)             AS latest_date,
            COUNT(DISTINCT store_id)   AS total_stores,
            COUNT(DISTINCT product_id) AS total_products
        FROM aggregated_sales
    """)
    row = cur.fetchone()
    cur.close()
    conn.close()
    return {
        "total_records":  row[0],
        "total_units":    float(row[1]) if row[1] else 0,
        "latest_date":    str(row[2]),
        "total_stores":   row[3],
        "total_products": row[4]
    }

# ── Forecasting ───────────────────────────────────────
@app.get("/predict-demand",
         response_model=PredictionResponse,
         tags=["Forecasting"])
def predict_demand(
    product_id: int = Query(..., gt=0),
    store_id:   int = Query(..., gt=0)
):
    t0       = time.time()
    log.info(f"Predict: product={product_id} store={store_id}")
    sequence = build_sequence(store_id)
    pred     = run_inference(sequence)
    latency  = round((time.time() - t0) * 1000, 2)
    log.info(f"Prediction: {pred:.2f} units | {latency}ms")
    return PredictionResponse(
        product_id=product_id,
        store_id=store_id,
        predicted_demand=round(pred, 2),
        latency_ms=latency
    )

@app.get("/predict-7-days", tags=["Forecasting"])
def predict_7_days(
    product_id: int = Query(..., gt=0),
    store_id:   int = Query(..., gt=0)
):
    """Real multi-step 7-day autoregressive forecast."""
    log.info(f"7-day forecast: product={product_id} store={store_id}")

    conn = get_conn()
    cur  = conn.cursor()
    cur.execute("""
        SELECT MAX(sale_date) FROM aggregated_sales
        WHERE store_id = %s
    """, (store_id,))
    last_date = cur.fetchone()[0]
    cur.close()
    conn.close()

    if last_date is None:
        raise HTTPException(status_code=404,
                            detail=f"No data for store {store_id}")

    sequence    = build_sequence(store_id)
    forecasts   = []
    current_seq = sequence.copy()

    for day in range(7):
        pred = run_inference(current_seq)
        forecast_date = (pd.Timestamp(str(last_date)) +
                         pd.Timedelta(days=int(day + 1)))
        forecasts.append({
            "date":             forecast_date.strftime("%Y-%m-%d"),
            "predicted_demand": round(pred, 2),
            "day":              day + 1
        })
        # Roll sequence forward
        new_row     = current_seq[-1].copy()
        new_row[0]  = pred              # update total_quantity
        new_row[1]  = current_seq[-1][0]  # update lag_1
        current_seq = np.vstack([current_seq[1:], new_row])

    return {
        "product_id":       product_id,
        "store_id":         store_id,
        "last_actual_date": str(last_date),
        "forecast":         forecasts
    }

@app.post("/predict-batch", tags=["Forecasting"])
def predict_batch(body: BatchPredictionRequest):
    """Batch predictions for multiple store/product pairs."""
    results = []
    for req in body.requests:
        try:
            t0       = time.time()
            sequence = build_sequence(req.store_id)
            pred     = run_inference(sequence)
            latency  = round((time.time() - t0) * 1000, 2)
            results.append({
                "product_id":       req.product_id,
                "store_id":         req.store_id,
                "predicted_demand": round(pred, 2),
                "latency_ms":       latency,
                "status":           "ok"
            })
        except Exception as e:
            results.append({
                "product_id": req.product_id,
                "store_id":   req.store_id,
                "status":     "error",
                "detail":     str(e)
            })
    return {"predictions": results, "total": len(results)}

# ── Anomaly Detection ─────────────────────────────────
@app.get("/get-anomalies", tags=["Anomaly Detection"])
def get_anomalies(limit: int = Query(10, gt=0, le=100)):
    log.info(f"Anomaly request: limit={limit}")
    conn = get_conn()
    cur  = conn.cursor()
    cur.execute("""
        SELECT store_id, product_id, anomaly_flag,
               z_score, detected_at
        FROM anomalies
        WHERE anomaly_flag = true
        ORDER BY z_score DESC NULLS LAST
        LIMIT %s
    """, (limit,))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return {
        "total": len(rows),
        "anomalies": [
            {
                "store_id":    r[0],
                "product_id":  r[1],
                "anomaly_flag": r[2],
                "z_score":     float(r[3]) if r[3] else None,
                "detected_at": str(r[4])
            }
            for r in rows
        ]
    }

# ── Inventory ─────────────────────────────────────────
@app.get("/get-inventory-recommendations", tags=["Inventory"])
def get_inventory(store_id: int = Query(..., gt=0)):
    log.info(f"Inventory request: store={store_id}")
    conn = get_conn()
    cur  = conn.cursor()

    cur.execute("""
        SELECT
            sale_date,
            total_quantity,
            AVG(total_quantity) OVER (
                ORDER BY sale_date
                ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
            ) AS rolling_mean_7
        FROM aggregated_sales
        WHERE store_id = %s
        ORDER BY sale_date DESC
        LIMIT 20
    """, (store_id,))
    rows = cur.fetchall()
    cur.close()
    conn.close()

    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"No data for store {store_id}"
        )

    # Overall mean for this store (used as reorder baseline)
    overall_mean = sum(float(r[1]) for r in rows) / len(rows)

    recommendations = []
    for i, r in enumerate(rows):
        sale_date    = str(r[0])
        demand       = float(r[1]) if r[1] else 0
        rolling      = float(r[2]) if r[2] else demand
        safety_stock = rolling * SAFETY_STOCK_FACTOR
        rop          = (rolling * LEAD_TIME) + safety_stock

        # Reorder needed when demand drops significantly below rolling mean
        # (indicates stockout risk in near future)
        needs_reorder = demand < (overall_mean * 0.75)

        recommendations.append({
            "sale_date":      sale_date,
            "current_demand": round(demand, 2),
            "rolling_mean":   round(rolling, 2),
            "reorder_point":  round(rop, 2),
            "safety_stock":   round(safety_stock, 2),
            "status":         "⚠️ Reorder Soon" if needs_reorder else "✅ OK"
        })

    return {"store_id": store_id, "recommendations": recommendations}