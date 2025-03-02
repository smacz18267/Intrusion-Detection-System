# Intrusion-Detection-System

This repository contains a real-time Intrusion Detection System (IDS) that monitors network traffic and system events to identify potential security threats. It utilizes Flask and Flask-SocketIO for real-time alerting, and deep learning models for advanced anomaly detection.

**Disclaimer:** This project is for research and educational purposes. It is not intended for production use without thorough validation, security auditing, and proper deployment measures.

## Table of Contents

- Overview
- 📁 Project Structure
- Setup and Installation
  - Backend Setup
  - Frontend Setup
- 🎯API Endpoints
- 📌 Detection Mechanism
- 🔥 Future Improvements

## Overview

This IDS is designed to analyze network traffic and system logs to detect:

- Port scans and suspicious network activity.
- Unusual authentication attempts.
- Anomalous system behaviors based on deep learning models.
- Signature-based and heuristic threat detection.

The backend is implemented in Flask and Flask-SocketIO, providing real-time alerts to the frontend, which is built using JavaScript and WebSockets. Deep learning models are used for intelligent threat analysis.
## 📁 Project Structure
```

cyber_ids/                # Root project directory
│── backend/              # Backend (Flask, AI model, WebSockets)
│   ├── app.py            # Main Flask server with WebSockets
│   ├── train.py          # Training script for AI model
│   ├── models.py         # Defines AI model (LSTM/Transformer)
│   ├── config.py         # Configuration file (hyperparams, paths)
│   ├── intrusion_model.pth  # Trained PyTorch model file
│
│── frontend/             # Frontend (HTML, CSS, JS)
│   ├── templates/        # HTML templates for Flask
│   │   ├── dashboard.html  # Main UI for alerts
│   │
│   ├── static/           # Static assets (CSS, JS, images)
│   │   ├── css/
│   │   │   ├── styles.css  # Dashboard styling
│   │   │
│   │   ├── js/
│   │   │   ├── main.js     # Handles WebSockets, updates alerts
│   │   │
│── requirements.txt       # Dependencies (Flask, PyTorch, SocketIO)
```

## Setup and Installation

### **1. Clone the Repository**

```bash
git clone https://github.com/your-username/stock-prediction-model.git
cd stock-prediction-model
```

### **2. Backend Setup**

#### Activate Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

#### Set Up PostgreSQL Database

1. Make sure you have PostgreSQL installed.
2. Create a new database:
```bash
CREATE DATABASE stock_db;
```
3. Grant privileges to your PostgreSQL user:
```bash
GRANT ALL PRIVILEGES ON DATABASE stock_db TO your_username;
```
#### Configure .env File

Create a .env file inside the backend folder:

```bash
DATABASE_URL=postgresql://your_username:your_password@localhost:5432/stock_db
```
Replace your_username and your_password with your actual PostgreSQL credentials.

#### Run the FastAPI Server
```bash
uvicorn app:app --reload
```

The server will start at http://127.0.0.1:8000, and API documentation will be available at http://127.0.0.1:8000/docs.

### **3. Frontend Setup**

#### Navigate to the Frontend Directory
```bash
cd frontend
```

#### Open the Frontend in Your Browser
You can open the index.html file directly or serve it using a local server:
```bash
python -m http.server 8080
```

Then, open http://localhost:8080 in your browser.
 
## 🎯 API Endpoints

### **POST /predict**
- Description: Predicts the next stock price for a given ticker.
- Request Body:

```bash
{
  "ticker": "AAPL"
}
```
- Response:
```bash
{
  "ticker": "AAPL",
  "predicted_price": 123.45,
  "timestamp": "2025-02-24T09:54:10.081924"
}
```
Test the API at http://127.0.0.1:8000/docs.  

## 📌 Model Architecture

This model processes historical stock data using:
- Conv1D Layers: Extracts local features from the stock price time series.
- Bidirectional LSTM Layers: Captures long-term dependencies in stock movements.
- Custom Attention Mechanism: Helps focus on relevant historical patterns.
- Dense Layers: Produces the final predicted stock price.

## 🔥 Future Improvements

- ✅ Data Enrichment: Incorporate real-world news sentiment, macroeconomic indicators, or alternative datasets.
- ✅ Advanced Model Architectures: Experiment with transformer-based models or ensemble methods.
- ✅ Hyperparameter Tuning: Implement systematic tuning with tools like Keras Tuner.
- ✅ Backtesting & Risk Management: Develop frameworks to validate the model’s effectiveness in real-world trading.
