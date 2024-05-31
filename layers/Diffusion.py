import numpy as np

class Diffusion:
    def __init__(self, noise_steps=500, beta_start=1e-4, beta_end=0.02):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule()
        self.alpha = 1. - self.beta
        self.alpha_hat = np.cumprod(self.alpha)

    def prepare_noise_schedule(self):
        return np.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = np.sqrt(self.alpha_hat[t])[:, None]
        sqrt_one_minus_alpha_hat = np.sqrt(1 - self.alpha_hat[t])[:, None]
        epsilon = np.random.randn(*x.shape)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon, epsilon

    def sample_timesteps(self, n):
        return np.random.randint(1, self.noise_steps, size=(n,))

    def sample(self, model, n):
        x = np.random.randn(n, 3, self.img_size, self.img_size)
        for i in reversed(range(1, self.noise_steps)):
            t = np.array([i] * x.shape[0])
            predicted_noise = model(x, t)
            alpha = self.alpha[t][:, None]
            alpha_hat = self.alpha_hat[t][:, None]
            beta = self.beta[t][:, None]
            if i > 1:
                noise = np.random.randn(*x.shape)
            else:
                noise = np.zeros_like(x)
            x = 1 / np.sqrt(alpha) * (x - ((1 - alpha) / np.sqrt(1 - alpha_hat)) * predicted_noise) + np.sqrt(beta) * noise
        return x
