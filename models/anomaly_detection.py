import pandas as pd
import numpy as np
import psycopg2
from sklearn.ensemble import IsolationForest

DB_CONN = {
    "host": "localhost", "port": 5432,
    "database": "retail_db",
    "user": "retail_user", "password": "retail_pass"
}

def get_conn():
    return psycopg2.connect(**DB_CONN)

def load_data():
    conn = get_conn()
    query = """
        SELECT
            store_id,
            product_id,
            sale_date,
            total_quantity,
            COALESCE(avg_price, 0)      AS avg_price,
            COALESCE(lag_1, 0)          AS lag_1,
            COALESCE(lag_7, 0)          AS lag_7,
            COALESCE(rolling_mean_7, 0) AS rolling_mean_7
        FROM aggregated_sales
        ORDER BY store_id, sale_date
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df

def detect_zscore(df):
    """Per-store z-score — flags days that deviate > 2σ from store mean."""
    results = []
    for store_id, group in df.groupby("store_id"):
        g    = group.copy()
        mean = g["total_quantity"].mean()
        std  = g["total_quantity"].std()
        if std == 0:
            continue
        g["z_score"]        = (g["total_quantity"] - mean) / std
        g["zscore_anomaly"] = g["z_score"].abs() > 2
        results.append(g)
    return pd.concat(results).reset_index(drop=True)

def detect_isolation_forest(df):
    """Isolation Forest on rich feature space."""
    feature_cols = [
        "total_quantity", "avg_price",
        "lag_1", "lag_7", "rolling_mean_7"
    ]
    features     = df[feature_cols].fillna(0)
    model        = IsolationForest(
        n_estimators  = 100,
        contamination = 0.05,
        random_state  = 42
    )
    df = df.copy()
    df["if_anomaly"] = model.fit_predict(features) == -1
    return df

def save_anomalies(df):
    """Save only flagged anomalies to DB."""
    conn = get_conn()
    cur  = conn.cursor()
    cur.execute("TRUNCATE anomalies;")

    anomaly_rows = df[df["zscore_anomaly"] | df["if_anomaly"]].copy()

    batch = []
    for _, row in anomaly_rows.iterrows():
        batch.append((
            int(row["store_id"]),
            int(row["product_id"]),
            True,
            round(float(row["z_score"]), 4)
        ))

    cur.executemany("""
        INSERT INTO anomalies (store_id, product_id, anomaly_flag, z_score)
        VALUES (%s, %s, %s, %s)
    """, batch)

    conn.commit()
    cur.close()
    conn.close()
    return len(anomaly_rows)

if __name__ == "__main__":
    print("Running anomaly detection on full Rossmann dataset...")
    df = load_data()
    print(f"Total records loaded: {len(df):,}")

    print("Running Z-score detection (per store)...")
    df = detect_zscore(df)

    print("Running Isolation Forest...")
    df = detect_isolation_forest(df)

    zscore_count = int(df["zscore_anomaly"].sum())
    if_count     = int(df["if_anomaly"].sum())
    both_count   = int((df["zscore_anomaly"] | df["if_anomaly"]).sum())

    print(f"\nResults:")
    print(f"  Z-score anomalies:          {zscore_count:,}")
    print(f"  Isolation Forest anomalies: {if_count:,}")
    print(f"  Total unique anomalies:     {both_count:,}")

    print("\nSaving to database...")
    saved = save_anomalies(df)
    print(f"Saved {saved:,} anomaly records")

    # Show top 10 by z-score
    top = df[df["zscore_anomaly"] | df["if_anomaly"]].nlargest(10, "z_score")
    print("\nTop 10 anomalies by Z-score:")
    print(top[["store_id", "product_id", "total_quantity",
               "z_score", "zscore_anomaly", "if_anomaly"]].to_string(index=False))

    # Store distribution
    print("\nAnomalies per store (top 10):")
    store_dist = df[df["zscore_anomaly"] | df["if_anomaly"]]\
        .groupby("store_id").size()\
        .sort_values(ascending=False).head(10)
    print(store_dist.to_string())