# Insurance Policy Renewal Predictor

A FastAPI-based service that predicts whether a customer will renew their insurance policy, leveraging ensemble machine learning models trained on the “Customer Insurance Policies Prediction” dataset.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Setup](#setup)
- [API Usage](#api-usage)
- [Docker](#docker)
- [Results](#results)

---

## Project Structure

```
.
├── Dockerfile
├── LICENSE
├── Makefile
├── README.md
├── data
│   └── dataset.csv
├── models
│   ├── bagging_model.pkl
│   └── boosting_model.pkl
├── notebooks
│   └── pipeline.ipynb
├── requirements.txt
├── temp
└── webapp
    └── app.py
```

- **data/**: Raw CSV dataset.
- **models/**: Serialized `joblib` models (Random Forest & Gradient Boosting).
- **notebooks/**: EDA and model training pipeline.
- **webapp/**: FastAPI application source code.
- **Dockerfile**, **Makefile**: Containerization and automation scripts.

---

## Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-org/insurance-policy-predictor.git
   cd insurance-policy-predictor
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the API locally:**
   ```bash
   uvicorn webapp.app:app --reload
   ```

---

## API Usage

### Health Check

- **Endpoint:** `GET /`
- **Response:**
  ```json
  { "message": "Insurance Policy Renewal Prediction API" }
  ```

### Predict Renewal

- **Endpoint:** `POST /predict`
- **Content-Type:** `application/json`
- **Request Example:**
  ```json
  {
    "age": 30,
    "driving_license": 1,
    "region_code": 10,
    "previously_insured": 0,
    "annual_premium": 25000.0,
    "policy_sales_channel": 152,
    "vintage": 100,
    "gender": "Male",
    "vehicle_age": "< 1 Year",
    "vehicle_damage": "No",
    "model_type": "bagging"
  }
  ```
- **Response Example:**
  ```json
  {
    "model": "bagging",
    "prediction": 1,
    "probability": 0.87
  }
  ```

- **Interactive API docs:** Visit `/docs` after starting the server.

---

## Docker

To build and run the application using Docker:

```bash
docker build -t insurance-predictor .
docker run -p 8000:8000 insurance-predictor
```

---

## Results

After hyperparameter tuning:
- **Random Forest (Bagging):** ~90.7% accuracy
- **Gradient Boosting:** ~83.5% accuracy

---

