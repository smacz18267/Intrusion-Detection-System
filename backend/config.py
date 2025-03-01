# config.py
import os

class Config:
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'super_secret_key'

    # Deep Learning Model Parameters
    MODEL_TYPE = 'Transformer'
    INPUT_DIM = 20
    SEQ_LEN = 50
    HIDDEN_DIM = 64
    NUM_LAYERS = 2
    BIDIRECTIONAL = True
    D_MODEL = 64
    NHEAD = 4
    NUM_CLASSES = 2
    LATENT_DIM = 32
    LR = 0.001
    MODEL_SAVE_PATH = 'intrusion_model.pth'
    REC_WEIGHT = 0.05

    # Device configuration
    DEVICE = 'cuda' if os.environ.get('CUDA_AVAILABLE') else 'cpu'
