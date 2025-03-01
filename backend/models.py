# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SequenceAnalyzerLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, bidirectional=True):
        super(SequenceAnalyzerLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=bidirectional
        )
        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), 128)

    def forward(self, x):
        out, (h, c) = self.lstm(x)
        if self.lstm.bidirectional:
            out_last = torch.cat((h[-2, :, :], h[-1, :, :]), dim=1)
        else:
            out_last = h[-1, :, :]
        features = F.relu(self.fc(out_last))
        return features

class SequenceAnalyzerTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers):
        super(SequenceAnalyzerTransformer, self).__init__()
        self.input_fc = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 128)

    def forward(self, x):
        x = self.input_fc(x)
        x = x.permute(1, 0, 2)  # Transformer expects [seq_len, batch, d_model]
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)  # Mean pooling over sequence dimension
        features = F.relu(self.fc(x))
        return features

class Autoencoder(nn.Module):
    def __init__(self, feature_dim, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim)
        )
    
    def forward(self, x):
        z = self.encoder(x)
        reconstruction = self.decoder(z)
        return reconstruction, z

class IntrusionDetectionModel(nn.Module):
    def __init__(self, config):
        super(IntrusionDetectionModel, self).__init__()
        self.model_type = config.get('MODEL_TYPE', 'LSTM')
        if self.model_type == 'LSTM':
            self.sequence_analyzer = SequenceAnalyzerLSTM(
                input_dim=config['INPUT_DIM'],
                hidden_dim=config['HIDDEN_DIM'],
                num_layers=config['NUM_LAYERS'],
                bidirectional=config.get('BIDIRECTIONAL', True)
            )
        else:
            self.sequence_analyzer = SequenceAnalyzerTransformer(
                input_dim=config['INPUT_DIM'],
                d_model=config['D_MODEL'],
                nhead=config['NHEAD'],
                num_layers=config['NUM_LAYERS']
            )
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, config['NUM_CLASSES'])
        )
        self.autoencoder = Autoencoder(feature_dim=128, latent_dim=config['LATENT_DIM'])
        
    def forward(self, x):
        features = self.sequence_analyzer(x)
        logits = self.classifier(features)
        reconstruction, latent = self.autoencoder(features)
        return logits, reconstruction, features
