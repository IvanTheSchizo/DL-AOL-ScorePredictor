import torch
import torch.nn as nn

class DenoisingAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=16):
        super(DenoisingAutoencoder, self).__init__()
        
        # Encoder: Compress
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim) # Compressed Representation
        )
        
        # Decoder: Expand
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim) # Back to original size
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent
    
class FeatureAttention(nn.Module):
    def __init__(self, input_dim):
        super(FeatureAttention, self).__init__()
        # A small network to calculate "importance scores" for each feature
        self.attention = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.Tanh(),
            nn.Linear(32, input_dim),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # x shape: (batch_size, input_dim)
        weights = self.attention(x)
        # Apply weights to the input features (Element-wise multiplication)
        weighted_features = x * weights
        return weighted_features, weights

class StudentPredictor(nn.Module):
    def __init__(self, input_dim, latent_dim=16, pretrained_ae_path=None):
        super(StudentPredictor, self).__init__()
        
        # 1. Load the Autoencoder (Transfer Learning)
        self.autoencoder = DenoisingAutoencoder(input_dim, latent_dim)
        if pretrained_ae_path:
            self.autoencoder.load_state_dict(torch.load(pretrained_ae_path))
            # Freeze the Encoder? (Optional: let's keep it trainable for fine-tuning)
            # for param in self.autoencoder.encoder.parameters():
            #     param.requires_grad = False
        
        # 2. Attention Mechanism
        self.attention = FeatureAttention(latent_dim)
        
        # 3. Final Regressor
        self.regressor = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)  # Output: Exam Score
        )

    def forward(self, x):
        # Step 1: Get compressed knowledge from Autoencoder
        # We only use the encoder part!
        latent_features = self.autoencoder.encoder(x)
        
        # Step 2: Apply Attention
        attended_features, attn_weights = self.attention(latent_features)
        
        # Step 3: Predict
        prediction = self.regressor(attended_features)
        return prediction