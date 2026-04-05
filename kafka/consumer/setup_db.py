import psycopg2

conn = psycopg2.connect(
    host="localhost",
    port=5432,
    database="retail_db",
    user="retail_user",
    password="retail_pass"
)

cur = conn.cursor()

cur.execute("""
    CREATE TABLE IF NOT EXISTS transactions (
        transaction_id  SERIAL PRIMARY KEY,
        product_id      INTEGER NOT NULL,
        store_id        INTEGER NOT NULL,
        quantity        INTEGER NOT NULL,
        price           NUMERIC(10, 2) NOT NULL,
        timestamp       TIMESTAMP NOT NULL
    );
""")

cur.execute("""
    CREATE TABLE IF NOT EXISTS aggregated_sales (
        id              SERIAL PRIMARY KEY,
        product_id      INTEGER NOT NULL,
        store_id        INTEGER NOT NULL,
        sale_date       DATE NOT NULL,
        total_quantity  INTEGER,
        total_revenue   NUMERIC(10, 2)
    );
""")

cur.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id              SERIAL PRIMARY KEY,
        product_id      INTEGER NOT NULL,
        predicted_date  DATE NOT NULL,
        predicted_demand NUMERIC(10, 2),
        created_at      TIMESTAMP DEFAULT NOW()
    );
""")

cur.execute("""
    CREATE TABLE IF NOT EXISTS anomalies (
        id              SERIAL PRIMARY KEY,
        store_id        INTEGER NOT NULL,
        product_id      INTEGER NOT NULL,
        anomaly_flag    BOOLEAN DEFAULT FALSE,
        z_score         NUMERIC(10, 4),
        detected_at     TIMESTAMP DEFAULT NOW()
    );
""")

conn.commit()
cur.close()
conn.close()

print("All tables created successfully!")
