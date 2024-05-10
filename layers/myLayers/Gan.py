import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class Generator(nn.Module):
    def __init__(self, noise_len, n_samples, alpha):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_len, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(alpha),
            nn.Dropout(0.3),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(alpha),
            nn.Dropout(0.3),
            nn.Linear(512, n_samples)
        )

    def forward(self, x):
        output = self.model(x)
        return output


class Discriminator(nn.Module):
    def __init__(self, n_samples, alpha):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(n_samples, 256),
            # nn.BatchNorm1d(256),
            nn.LeakyReLU(alpha),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            # nn.BatchNorm1d(128),
            nn.LeakyReLU(alpha),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            # nn.BatchNorm1d(64),
            nn.LeakyReLU(alpha),
            nn.Dropout(0.3),
            nn.Linear(64, n_samples),
            nn.Sigmoid(),
        )

    def forward(self, x):
        output = self.model(x)
        return output


# Define Losses
# loss_D = nn.BCELoss()  # Binary Cross Entropy Loss for Generator
# loss_CD = nn.BCELoss()  # Binary Cross Entropy Loss for Central Discriminator
#
# # Initialize Generator, Discriminator, and Central Discriminator
# noise_len = 432  # Length of noise vector
# n_samples = 432   # Number of output samples
# alpha = 0.2      # Leaky ReLU slope
# n_channels = 7   # Number of channels
# generator = Generator(noise_len, n_samples, alpha, n_channels)
# discriminator = Discriminator(n_samples, alpha, n_channels)
#
# # Define Optimizers
# optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
# optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
#
# # Training Loop
# n_epochs = 100
# batch_size = 32
# seq_len = 432
# n_channels = 7
# real_data = torch.tensor(np.random.rand(batch_size, n_channels, seq_len).astype(np.float32))
# for epoch in range(n_epochs):
#     for i in range(10):
#         # Generate noise
#         noise = torch.randn(batch_size ,n_channels, noise_len)
#
#         # Generate fake data
#         generated_data = generator(noise)
#
#         # Train Discriminator
#         optimizer_D.zero_grad()
#         real_output = discriminator(real_data)
#         fake_output = discriminator(generated_data.detach())
#         loss_discriminator = loss_D(real_output, torch.ones_like(real_output)) + \
#                              loss_D(fake_output, torch.zeros_like(fake_output))
#         loss_discriminator.backward()
#         optimizer_D.step()
#
#         # Train Generator
#         optimizer_G.zero_grad()
#         fake_output = discriminator(generated_data)
#         loss_generator = loss_D(fake_output, torch.ones_like(fake_output))
#         loss_generator.backward()
#         optimizer_G.step()
#
#     print(f'Epoch [{epoch}/{n_epochs}], Loss_D: {loss_discriminator.item()}, Loss_G: {loss_generator.item()}')
