# app.py
import time
import threading
import random
import numpy as np
import torch
from flask import Flask, render_template
from flask_socketio import SocketIO
from models import IntrusionDetectionModel
from config import Config
import os

app = Flask(__name__, 
            template_folder=os.path.join(os.path.dirname(__file__), "../frontend/templates"),
            static_folder=os.path.join(os.path.dirname(__file__), "../frontend/static"))
app.config.from_object(Config)
socketio = SocketIO(app, async_mode="gevent")

# Load the trained IDS model
model_config = {
    'MODEL_TYPE': Config.MODEL_TYPE,
    'INPUT_DIM': Config.INPUT_DIM,
    'HIDDEN_DIM': Config.HIDDEN_DIM,
    'NUM_LAYERS': Config.NUM_LAYERS,
    'BIDIRECTIONAL': Config.BIDIRECTIONAL,
    'D_MODEL': Config.D_MODEL,
    'NHEAD': Config.NHEAD,
    'NUM_CLASSES': Config.NUM_CLASSES,
    'LATENT_DIM': Config.LATENT_DIM,
    'DEVICE': Config.DEVICE
}
model = IntrusionDetectionModel(model_config)
model_path = os.path.join(os.path.dirname(__file__), "intrusion_model.pth")
model.load_state_dict(torch.load(model_path, map_location=Config.DEVICE))
model.to(Config.DEVICE)
model.eval()

@app.route('/')
def index():
    return render_template('dashboard.html')

def detect_anomalies():
    threshold = 0.5 
    while True:
        # Simulate a network traffic sequence
        sequence = np.random.randn(Config.SEQ_LEN, Config.INPUT_DIM).astype(np.float32)
        if random.random() < 0.05:  # 5% chance to inject an anomaly
            sequence += np.random.uniform(5, 10, size=sequence.shape).astype(np.float32)
        
        sequence_tensor = torch.from_numpy(sequence).unsqueeze(0).to(Config.DEVICE)
        with torch.no_grad():
            logits, reconstruction, features = model(sequence_tensor)
            anomaly_score = torch.mean((features - reconstruction) ** 2).item()
            pred = torch.argmax(logits, dim=1).item()
        
        # Emit alert if anomaly detected
        if anomaly_score > threshold or pred == 1:
            alert = {
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                'anomaly_score': anomaly_score,
                'prediction': pred,
                'details': 'Potential intrusion detected.'
            }
            socketio.emit('new_alert', alert, broadcast=True)
        time.sleep(1)

def start_detection():
    detection_thread = threading.Thread(target=detect_anomalies)
    detection_thread.daemon = True
    detection_thread.start()

if __name__ == '__main__':
    start_detection()
    socketio.run(app, host='0.0.0.0', port=5001, debug=True)

