# 🛒 Retail Intelligence System

An end-to-end Machine Learning, Data Engineering, and Analytics platform designed to revolutionize retail operations. The system features real-time demand forecasting, anomaly detection, and inventory optimization, presented through an interactive analytics dashboard.

## 🌟 Key Features

- **Store & Product Analytics**: High-level sales trends and top-performing stores tracking.
- **Demand Forecasting**: 7-day down to 90-day autoregressive demand forecasting using a custom-trained PyTorch LSTM model and Facebook Prophet.
- **Anomaly Detection**: Advanced detection of irregular sales patterns using statistical rules (Z-Score) and Machine Learning (Isolation Forests).
- **Inventory Optimization**: Data-driven recommendations for safety stock and reorder points based on rolling means and forecasted demand.
- **Real-Time Data Streaming**: Robust pipeline handling real-time point-of-sale transactions via Apache Kafka.
- **Batch Processing & ETL**: Scheduled data aggregation and feature engineering pipelines powered by Apache Airflow.

---

## 🏗️ Architecture & Tech Stack

- **Frontend & Visualization**: `Streamlit`, `Plotly Express`, `Plotly Graph Objects`
- **Backend API**: `FastAPI`, `Uvicorn`, `Pydantic`
- **Machine Learning**: `PyTorch` (LSTM), `Prophet`, `scikit-learn` (Isolation Forests, Scaling)
- **Data Engineering**: `Apache Kafka`, `Apache Airflow`
- **Database**: `PostgreSQL` (Local & Supabase cloud migration support)
- **Infrastructure**: `Docker`, `Docker Compose`

---

## 📂 Project Structure

```text
retail-intelligence/
├── airflow/               # Airflow DAGs for ETL and pipeline orchestration (`retail_pipeline.py`)
├── api/                   # FastAPI backend (`main.py`, `config.py`) serving inference & data endpoints
├── dashboard/             # Interactive Streamlit dashboard (`app.py`) for data visualization
├── data/                  # Data loading & migration tools (`load_rossmann.py`, `migrate_to_supabase.py`)
├── kafka/                 # Real-time event streaming components
│   ├── consumer/          # Kafka consumer and DB setup scripts (`consumer.py`, `setup_db.py`)
│   └── producer/          # Synthetic transaction producer (`producer.py`)
├── models/                # Machine Learning models and inference scripts
│   ├── anomaly_detection.py      # Z-score and Isolation Forest detection
│   ├── feature_engineering.py   # Lags, rolling means, scaling
│   ├── forecasting.py            # Baseline Prophet forecasting
│   ├── lstm_forecaster.py        # PyTorch LSTM architecture & inference logic
│   └── run_training.py           # Training pipeline orchestration for the LSTM
├── docker-compose.yml     # Infrastructure setup (Zookeeper, Kafka, PostgreSQL)
└── requirements.txt       # Python dependencies
```

---

## 🚀 Getting Started

### 1. Start Infrastructure (Docker)
Ensure Docker is running and spin up Kafka, Zookeeper, and PostgreSQL:
```bash
docker-compose up -d
```

### 2. Set Up the Environment
Create a virtual environment and install the dependencies:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Initialize Database
Create the necessary tables in PostgreSQL:
```bash
python kafka/consumer/setup_db.py
```

### 4. Load Data
You can either simulate real-time data using Kafka, or load batch historical data (like the Rossmann dataset):
- **Real-time Streaming**:
  Start the Kafka consumer in one terminal:
  ```bash
  python kafka/consumer/consumer.py
  ```
  Start the Kafka producer in another terminal to generate transactions:
  ```bash
  python kafka/producer/producer.py
  ```
- **Batch Loading (Rossmann Data)**:
  ```bash
  python data/load_rossmann.py
  ```

### 5. Train Models (Optional)
If you want to re-train the underlying PyTorch LSTM model:
```bash
python models/run_training.py
```
*Note: Make sure your aggregated sales data is populated via the ETL pipeline or batch script first!*

### 6. Start the FastAPI Backend
Serve the ML models and data endpoints:
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
The API documentation will be accessible at `http://localhost:8000/docs`.

### 7. Run the Streamlit Dashboard
Launch the interactive visualization dashboard:
```bash
streamlit run dashboard/app.py
```
The dashboard will open automatically in your browser at `http://localhost:8501`.

---

## ☁️ Cloud Migration (Supabase)
This project natively supports scaling to the cloud. You can push your local Postgres schemas and data up to a cloud-hosted Supabase instance using the provided migration tool. Ensure your `DB_PASSWORD` and connection variables are set, then run:
```bash
python data/migrate_to_supabase.py
```

## 📜 License
MIT License
