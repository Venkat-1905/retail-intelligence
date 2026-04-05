from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import psycopg2

DB_CONN = {
    "host": "localhost",
    "port": 5432,
    "database": "retail_db",
    "user": "retail_user",
    "password": "retail_pass"
}

def get_conn():
    return psycopg2.connect(**DB_CONN)

# Task 1 — Aggregate raw transactions into daily sales
def aggregate_sales():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO aggregated_sales (product_id, store_id, sale_date, total_quantity, total_revenue)
        SELECT
            product_id,
            store_id,
            DATE(timestamp)       AS sale_date,
            SUM(quantity)         AS total_quantity,
            SUM(quantity * price) AS total_revenue
        FROM transactions
        WHERE DATE(timestamp) = CURRENT_DATE 
        GROUP BY product_id, store_id, DATE(timestamp)
        ON CONFLICT DO NOTHING;
    """)
    conn.commit()
    cur.close()
    conn.close()
    print("Daily aggregation done.")

# Task 2 — Feature engineering (lag + rolling mean)
def feature_engineering():
    conn = get_conn()
    cur = conn.cursor()

    # Add lag_1 column if not exists
    cur.execute("""
        ALTER TABLE aggregated_sales
        ADD COLUMN IF NOT EXISTS lag_1 NUMERIC(10,2),
        ADD COLUMN IF NOT EXISTS lag_7 NUMERIC(10,2),
        ADD COLUMN IF NOT EXISTS rolling_mean_7 NUMERIC(10,2);
    """)

    # Compute lag_1 (previous day demand)
    cur.execute("""
        UPDATE aggregated_sales a
        SET lag_1 = b.total_quantity
        FROM aggregated_sales b
        WHERE a.product_id = b.product_id
          AND a.store_id   = b.store_id
          AND a.sale_date  = b.sale_date + INTERVAL '1 day';
    """)

    # Compute lag_7 (7 days ago demand)
    cur.execute("""
        UPDATE aggregated_sales a
        SET lag_7 = b.total_quantity
        FROM aggregated_sales b
        WHERE a.product_id = b.product_id
          AND a.store_id   = b.store_id
          AND a.sale_date  = b.sale_date + INTERVAL '7 days';
    """)

    # Compute rolling mean over last 7 days
    cur.execute("""
        UPDATE aggregated_sales a
        SET rolling_mean_7 = sub.avg_qty
        FROM (
            SELECT
                product_id,
                store_id,
                sale_date,
                AVG(total_quantity) OVER (
                    PARTITION BY product_id, store_id
                    ORDER BY sale_date
                    ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
                ) AS avg_qty
            FROM aggregated_sales
        ) sub
        WHERE a.product_id = sub.product_id
          AND a.store_id   = sub.store_id
          AND a.sale_date  = sub.sale_date;
    """)

    conn.commit()
    cur.close()
    conn.close()
    print("Feature engineering done.")

# DAG definition
default_args = {
    "owner": "venkat",
    "retries": 1,
    "retry_delay": timedelta(minutes=5)
}

with DAG(
    dag_id="retail_pipeline",
    default_args=default_args,
    description="Daily retail aggregation and feature engineering",
    schedule="@daily",
    start_date=datetime(2026, 4, 1),
    catchup=False
) as dag:

    t1 = PythonOperator(
        task_id="aggregate_sales",
        python_callable=aggregate_sales
    )

    t2 = PythonOperator(
        task_id="feature_engineering",
        python_callable=feature_engineering
    )

    t1 >> t2   # t1 runs first, then t2
