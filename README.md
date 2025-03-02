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
git clone https://github.com/your-username/intrusion-detection-system.git
cd intrusion-detection-system
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

#### Run the Flask Server

```bash
python3 app.py
```
The server will start at http://127.0.0.1:5001.

### **3. Frontend Setup**

1. Navigate to the Frontend Directory

```bash
cd frontend
```

2. Open the Frontend in Your Browser
You can open the dashboard.html file directly or serve it using a local server:
```bash
python -m http.server 5001
```
Then, open http://localhost:5001 in your browser.
 
## 🎯 API Endpoints

### **GET /**
- Description: Loads the Intrusion Detection Dashboard.
  
#### WebSocket Connection

- Description: Listens for real-time alerts from the backend.
- Event: new_alert
- Payload Example:

```bash
{
  "timestamp": "2025-03-02 00:33:01",
  "anomaly_score": 0.75,
  "prediction": 1,
  "details": "Potential intrusion detected."
}
```

## 📌 Detection Mechanism

This system processes network data using:
- LSTM/Transformer Models: Learn patterns in network traffic.
- Autoencoder for Anomaly Detection: Identifies unusual activity.
- Real-time Analysis with Flask-SocketIO: Streams alerts instantly.

## 🔥 Future Improvements

- ✅ Improve Model Accuracy: Fine-tune hyperparameters for better detection.
- ✅ Deploy on Cloud: Host the system on AWS/GCP.
- ✅ Enhance Visualization: Build a dashboard with graphs.
- ✅ Multi-User Support: Allow multiple admins to monitor alerts.
