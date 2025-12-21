
import torch
import math

def get_cosine_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

timesteps = 1000
betas = get_cosine_schedule(timesteps)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)

print(f"Alpha_bar at T={timesteps}: {alphas_cumprod[-1].item()}")
print(f"Alpha_bar at T=0: {alphas_cumprod[0].item()}")
print(f"Beta at T={timesteps}: {betas[-1].item()}")
