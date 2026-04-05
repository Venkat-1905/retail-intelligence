import pandas as pd
import numpy as np
import psycopg2

DB_CONFIG = {
    "dbname":   "retail_db",
    "user":     "retail_user",
    "password": "retail_pass",
    "host":     "localhost",
    "port":     5432
}

def load_daily_sales():
    conn = psycopg2.connect(**DB_CONFIG)
    query = """
        SELECT sale_date, product_id, store_id,
               total_quantity, avg_price
        FROM aggregated_sales
        ORDER BY product_id, store_id, sale_date;
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df

def build_features(df):
    records = []
    for (product_id, store_id), group in df.groupby(["product_id", "store_id"]):
        group = group.sort_values("sale_date").copy()

        # Lag features
        group["lag_1"] = group["total_quantity"].shift(1)
        group["lag_7"] = group["total_quantity"].shift(7)

        # Rolling stats — shift by 1 to avoid data leakage
        group["rolling_7"]   = group["total_quantity"].shift(1).rolling(7).mean()
        group["rolling_std"] = group["total_quantity"].shift(1).rolling(7).std()

        # Time-aware features
        group["sale_date"]   = pd.to_datetime(group["sale_date"])
        group["day_of_week"] = group["sale_date"].dt.dayofweek
        group["is_weekend"]  = group["day_of_week"].isin([5, 6]).astype(int)
        group["month"]       = group["sale_date"].dt.month

        group["product_id"] = product_id
        group["store_id"]   = store_id
        records.append(group)

    result = pd.concat(records).dropna().reset_index(drop=True)
    print(f"  Features built: {len(result)} rows, "
          f"{result['product_id'].nunique()} products, "
          f"{result['store_id'].nunique()} stores")
    return result

def make_sequences(df, seq_len=14):
    feature_cols = [
        "total_quantity", "lag_1", "lag_7",
        "rolling_7", "avg_price",
        "day_of_week", "is_weekend", "month"
    ]
    X_list, y_list = [], []

    for (product_id, store_id), group in df.groupby(["product_id", "store_id"]):
        group  = group.sort_values("sale_date")
        values  = group[feature_cols].values
        targets = group["total_quantity"].values

        for i in range(len(values) - seq_len):
            X_list.append(values[i: i + seq_len])
            y_list.append(targets[i + seq_len])

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)
