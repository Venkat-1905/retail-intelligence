import json
import time
import random
from datetime import datetime
from kafka import KafkaProducer
from faker import Faker

fake = Faker()

producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

PRODUCTS = list(range(1, 21))   # 20 products
STORES   = list(range(1, 6))    # 5 stores

def generate_transaction():
    return {
        "product_id":  random.choice(PRODUCTS),
        "store_id":    random.choice(STORES),
        "quantity":    random.randint(1, 20),
        "price":       round(random.uniform(10.0, 500.0), 2),
        "timestamp":   datetime.now().isoformat()
    }

print("Producer started... sending transactions every second.")
print("Press Ctrl+C to stop.\n")

count = 0
while True:
    transaction = generate_transaction()
    producer.send('retail-transactions', value=transaction)
    count += 1
    print(f"[{count}] Sent → store {transaction['store_id']} | "
          f"product {transaction['product_id']} | "
          f"qty {transaction['quantity']} | "
          f"price ₹{transaction['price']}")
    time.sleep(1)
