from diffusers import UNet2DModel
import torch
import math
from tqdm import tqdm
from interpolation import interpolation_grid


class DDIM:
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        sample_steps: int = 20,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
        self.timesteps = torch.linspace(num_train_timesteps - 1, 0, sample_steps).long()
    
    @torch.no_grad()
    def sample(
        self,
        unet: UNet2DModel,
        batch_size: int,
        in_channels: int,
        sample_size: int,
        eta: float = 0.0,
    ):
        alphas = self.alphas.to(unet.device)
        alphas_cumprod = self.alpha_cumprod.to(unet.device)
        timesteps = self.timesteps.to(unet.device)
        images = torch.randn((batch_size, in_channels, sample_size, sample_size), device=unet.device)
        for t, tau in tqdm(list(zip(timesteps[:-1], timesteps[1:])), desc="Sampling"):
            pred_noise: torch.Tensor = unet(images, t).sample
            if not math.isclose(eta, 0.0):
                one_minus_alpha_prod_tau = 1.0 - alphas_cumprod[tau]
                one_minus_alpha_prod_t = 1.0 - alphas_cumprod[t]
                one_minus_alpha_t = 1.0 - alphas[t]
                sigma_t = eta * (one_minus_alpha_prod_tau * one_minus_alpha_t / one_minus_alpha_prod_t) ** 0.5
            else:
                sigma_t = torch.zeros_like(alphas[0])
            alpha_cumprod_tau = alphas_cumprod[tau]
            sqrt_alphas_cumprod_tau = alpha_cumprod_tau ** 0.5
            alpha_cumprod_t = alphas_cumprod[t]
            sqrt_alphas_cumprod_t = alpha_cumprod_t ** 0.5
            sqrt_one_minus_alpha_cumprod_t = (1.0 - alpha_cumprod_t) ** 0.5
            first_term = sqrt_alphas_cumprod_tau * (images - sqrt_one_minus_alpha_cumprod_t * pred_noise) / sqrt_alphas_cumprod_t
            coeff = (1.0 - alpha_cumprod_tau - sigma_t ** 2) ** 0.5
            second_term = coeff * pred_noise
            epsilon = torch.randn_like(images)
            images = first_term + second_term + sigma_t * epsilon
        images = (images / 2.0 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()
        return images
    
    @torch.no_grad()
    def interpolation(
        self,
        unet: UNet2DModel,
        rows: int,
        cols: int,
        in_channels: int,
        sample_size: int,
        eta: float = 0.0,
    ):
        alphas = self.alphas.to(unet.device)
        alphas_cumprod = self.alpha_cumprod.to(unet.device)
        timesteps = self.timesteps.to(unet.device)
        images = interpolation_grid(rows, cols, in_channels, sample_size).to(unet.device).reshape(-1, in_channels, sample_size, sample_size)
        for t, tau in tqdm(list(zip(timesteps[:-1], timesteps[1:])), desc="Interpolation"):
            pred_noise: torch.Tensor = unet(images, t).sample
            if not math.isclose(eta, 0.0):
                one_minus_alpha_prod_tau = 1.0 - alphas_cumprod[tau]
                one_minus_alpha_prod_t = 1.0 - alphas_cumprod[t]
                one_minus_alpha_t = 1.0 - alphas[t]
                sigma_t = eta * (one_minus_alpha_prod_tau * one_minus_alpha_t / one_minus_alpha_prod_t) ** 0.5
            else:
                sigma_t = torch.zeros_like(alphas[0])
            alpha_cumprod_tau = alphas_cumprod[tau]
            sqrt_alphas_cumprod_tau = alpha_cumprod_tau ** 0.5
            alpha_cumprod_t = alphas_cumprod[t]
            sqrt_alphas_cumprod_t = alpha_cumprod_t ** 0.5
            sqrt_one_minus_alpha_cumprod_t = (1.0 - alpha_cumprod_t) ** 0.5
            first_term = sqrt_alphas_cumprod_tau * (images - sqrt_one_minus_alpha_cumprod_t * pred_noise) / sqrt_alphas_cumprod_t
            coeff = (1.0 - alpha_cumprod_tau - sigma_t ** 2) ** 0.5
            second_term = coeff * pred_noise
            epsilon = torch.randn_like(images)
            images = first_term + second_term + sigma_t * epsilon
        images = (images / 2.0 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()
        return images


model = UNet2DModel.from_pretrained("ddpm-results").cuda()
