import psycopg2
import os
from psycopg2.extras import execute_batch

# ── Local DB ──────────────────────────────────────────
LOCAL = {
    "host":     "localhost",
    "port":     5432,
    "database": "retail_db",
    "user":     "retail_user",
    "password": "retail_pass"
}

# ── Supabase DB — reads from env vars for security ────
SUPABASE = {
    "host":            os.getenv("DB_HOST", "aws-1-ap-southeast-1.pooler.supabase.com"),
    "port":            int(os.getenv("DB_PORT", 5432)),
    "database":        os.getenv("DB_NAME", "postgres"),
    "user":            os.getenv("DB_USER", "postgres.rlnmjtjffdhjxjeuxusa"),
    "password":        os.getenv("DB_PASSWORD", ""),
    "connect_timeout": 30,
    "sslmode":         "require"
}

def migrate():
    print("Connecting to local DB...")
    local_conn  = psycopg2.connect(**LOCAL)
    print("Connecting to Supabase...")
    remote_conn = psycopg2.connect(**SUPABASE)
    local_cur   = local_conn.cursor()
    remote_cur  = remote_conn.cursor()
    print("Both connections established ✅")

    # ── Create tables on Supabase ─────────────────────
    print("\nCreating tables on Supabase...")
    remote_cur.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            transaction_id  SERIAL PRIMARY KEY,
            product_id      INTEGER,
            store_id        INTEGER,
            quantity        INTEGER,
            price           NUMERIC(10,2),
            timestamp       TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS aggregated_sales (
            id             SERIAL PRIMARY KEY,
            sale_date      DATE,
            store_id       INTEGER,
            product_id     INTEGER,
            total_quantity NUMERIC,
            total_revenue  NUMERIC,
            lag_1          NUMERIC,
            lag_7          NUMERIC,
            rolling_mean_7 NUMERIC,
            avg_price      NUMERIC,
            promo_flag     INTEGER,
            day_of_week    INTEGER
        );

        CREATE TABLE IF NOT EXISTS predictions (
            id               SERIAL PRIMARY KEY,
            product_id       INTEGER,
            predicted_date   DATE,
            predicted_demand NUMERIC,
            created_at       TIMESTAMP DEFAULT NOW()
        );

        CREATE TABLE IF NOT EXISTS anomalies (
            id           SERIAL PRIMARY KEY,
            store_id     INTEGER,
            product_id   INTEGER,
            anomaly_flag BOOLEAN,
            z_score      NUMERIC,
            detected_at  TIMESTAMP DEFAULT NOW()
        );
    """)
    remote_conn.commit()
    print("Tables created ✅")

    # ── Migrate ALL aggregated_sales (all 1115 stores) ──
    print("\nMigrating aggregated_sales (all stores)...")

    # Clear existing data first to avoid duplicates
    print("  Clearing existing Supabase data...")
    remote_cur.execute("TRUNCATE aggregated_sales;")
    remote_conn.commit()

    local_cur.execute("""
        SELECT sale_date, store_id, product_id,
               total_quantity, total_revenue,
               lag_1, lag_7, rolling_mean_7,
               avg_price, promo_flag, day_of_week
        FROM aggregated_sales
        ORDER BY store_id, sale_date
    """)
    rows = local_cur.fetchall()
    print(f"  Fetched {len(rows):,} rows from local DB")

    # Insert in batches of 1000 with progress updates
    batch_size = 1000
    total      = len(rows)
    for i in range(0, total, batch_size * 50):
        batch = rows[i: i + batch_size * 50]
        execute_batch(remote_cur, """
            INSERT INTO aggregated_sales
                (sale_date, store_id, product_id,
                 total_quantity, total_revenue,
                 lag_1, lag_7, rolling_mean_7,
                 avg_price, promo_flag, day_of_week)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, batch, page_size=1000)
        remote_conn.commit()
        print(f"  Progress: {min(i + batch_size * 50, total):,} / {total:,} rows")

    print(f"  Migrated {total:,} rows ✅")

    # ── Migrate anomalies (all flagged) ───────────────
    print("\nMigrating anomalies...")

    # Clear existing anomalies
    remote_cur.execute("TRUNCATE anomalies;")
    remote_conn.commit()

    local_cur.execute("""
        SELECT store_id, product_id, anomaly_flag, z_score
        FROM anomalies
        WHERE anomaly_flag = true
        ORDER BY z_score DESC
        LIMIT 50000
    """)
    anom_rows = local_cur.fetchall()
    execute_batch(remote_cur, """
        INSERT INTO anomalies (store_id, product_id, anomaly_flag, z_score)
        VALUES (%s,%s,%s,%s)
    """, anom_rows, page_size=1000)
    remote_conn.commit()
    print(f"  Migrated {len(anom_rows):,} anomalies ✅")

    # ── Cleanup ───────────────────────────────────────
    local_cur.close()
    remote_cur.close()
    local_conn.close()
    remote_conn.close()

    print("\n" + "="*50)
    print("Migration complete! ✅")
    print("="*50)
    print(f"  aggregated_sales: {total:,} rows")
    print(f"  anomalies:        {len(anom_rows):,} rows")

if __name__ == "__main__":
    migrate()