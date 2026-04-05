import json
import psycopg2
from datetime import datetime
from kafka import KafkaConsumer

conn = psycopg2.connect(
    host="localhost",
    port=5432,
    database="retail_db",
    user="retail_user",
    password="retail_pass"
)
cur = conn.cursor()

consumer = KafkaConsumer(
    'retail-transactions',
    bootstrap_servers='localhost:9092',
    auto_offset_reset='earliest',
    value_deserializer=lambda v: json.loads(v.decode('utf-8'))
)

print("Consumer started... listening for transactions.")
print("Press Ctrl+C to stop.\n")

count = 0
for message in consumer:
    t = message.value
    cur.execute("""
        INSERT INTO transactions (product_id, store_id, quantity, price, timestamp)
        VALUES (%s, %s, %s, %s, %s)
    """, (
        t['product_id'],
        t['store_id'],
        t['quantity'],
        t['price'],
        datetime.fromisoformat(t['timestamp'])
    ))
    conn.commit()
    count += 1
    print(f"[{count}] Saved → store {t['store_id']} | "
          f"product {t['product_id']} | "
          f"qty {t['quantity']} | price ₹{t['price']}")
