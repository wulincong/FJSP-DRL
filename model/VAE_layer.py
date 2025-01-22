import torch
import torch.nn as nn
import torch.nn.functional as F
import torch as th

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        
        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc2_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder (if needed, depending on the design)
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc2_mu(h1), self.fc2_logvar(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))  # Or use another activation based on your needs

    def forward(self, x):
        # Encoding step
        mu, logvar = self.encode(x)
        
        # Reparameterization trick
        z = self.reparameterize(mu, logvar)
        
        # Decoding to reconstruct the input
        reconstructed_x = self.decode(z)
        
        # Return latent variable z, mu, logvar, and the reconstructed input
        return reconstructed_x, mu, logvar


