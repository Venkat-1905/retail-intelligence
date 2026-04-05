import pandas as pd
import psycopg2
from psycopg2.extras import execute_batch

DB_CONFIG = {
    "dbname": "retail_db",
    "user":   "retail_user",
    "password": "retail_pass",
    "host":   "localhost",
    "port":   5432
}

def load_rossmann():
    print("Loading Rossmann dataset...")
    train = pd.read_csv(
        "data/rossmann/train.csv",
        low_memory=False,
        parse_dates=["Date"]
    )
    store = pd.read_csv("data/rossmann/store.csv")
    df    = train.merge(store, on="Store", how="left")

    # Keep only open stores with sales
    df = df[(df["Open"] == 1) & (df["Sales"] > 0)].copy()

    df = df.rename(columns={
        "Date":      "sale_date",
        "Store":     "store_id",
        "Sales":     "total_quantity",
        "Promo":     "promo_flag",
        "DayOfWeek": "day_of_week"
    })

    df["product_id"] = "P" + df["store_id"].astype(str).str.zfill(3)
    df["avg_price"]  = df["total_quantity"] / df["Customers"].replace(0, 1)
    df["total_revenue"] = df["total_quantity"]

    result = df[[
        "sale_date", "store_id", "product_id",
        "total_quantity", "avg_price", "promo_flag", "day_of_week"
    ]]

    print(f"  {len(result)} rows loaded")
    print(f"  {result['product_id'].nunique()} products")
    print(f"  Date range: {result['sale_date'].min()} → {result['sale_date'].max()}")
    return result

def write_to_postgres(df):
    conn = psycopg2.connect(**DB_CONFIG)
    cur  = conn.cursor()

    # Add missing columns if needed
    cur.execute("""
        ALTER TABLE aggregated_sales
        ADD COLUMN IF NOT EXISTS avg_price    NUMERIC(10,2),
        ADD COLUMN IF NOT EXISTS promo_flag   INTEGER,
        ADD COLUMN IF NOT EXISTS day_of_week  INTEGER;
    """)

    # Clear old simulated data
    cur.execute("TRUNCATE aggregated_sales;")

    rows = [
        (
            row.sale_date, int(row.store_id), int(row.store_id),
            float(row.total_quantity), float(row.avg_price),
            int(row.promo_flag), int(row.day_of_week)
        )
        for row in df.itertuples()
    ]

    execute_batch(cur, """
        INSERT INTO aggregated_sales
            (sale_date, store_id, product_id, total_quantity,
             avg_price, promo_flag, day_of_week)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """, rows, page_size=500)

    conn.commit()
    cur.close()
    conn.close()
    print(f"Written {len(df)} rows to aggregated_sales ✅")

if __name__ == "__main__":
    df = load_rossmann()
    write_to_postgres(df)
    print("\nRossmann data ready!")
