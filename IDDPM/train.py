from accelerate import Accelerator
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.utils import make_image_grid, numpy_to_pil
import torch
import torch.nn.functional as F
import os
import math
from functools import partial
from diffusers import UNet2DModel
from tqdm import tqdm
from config import config
from dataloader import dataloader
from model import model, IDDPM
from utils import extract, ImportanceSampler


def pred_mean_logvar(
    iddpm: IDDPM,
    pred_noises: torch.Tensor,
    pred_vars: torch.Tensor,
    noisy_images: torch.Tensor,
    timesteps: torch.Tensor,
):
    betas = iddpm.betas.to(timesteps.device)
    alphas = iddpm.alphas.to(timesteps.device)
    alphas_cumprod = iddpm.alphas_cumprod.to(timesteps.device)
    alphas_cumprod_prev = iddpm.alphas_cumprod_prev.to(timesteps.device)
    sqrt_recip_alphas_cumprod = (1.0 / alphas_cumprod) ** 0.5
    sqrt_recipm1_alphas_cumprod = (1.0 / alphas_cumprod - 1.0) ** 0.5
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    posterior_log_variance_clipped = torch.log(torch.concat((posterior_variance[1:2], posterior_variance[1:])))
    posterior_mean_coef1 = betas * alphas_cumprod_prev ** 0.5 / (1.0 - alphas_cumprod)
    posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * alphas ** 0.5 / (1.0 - alphas_cumprod)
    _extract = partial(extract, timesteps=timesteps, broadcast_shape=noisy_images.shape)
    x_0 = _extract(sqrt_recip_alphas_cumprod) * noisy_images - _extract(sqrt_recipm1_alphas_cumprod) * pred_noises
    mean = _extract(posterior_mean_coef1) * x_0.clamp(-1, 1) + _extract(posterior_mean_coef2) * noisy_images
    min_log = _extract(posterior_log_variance_clipped)
    max_log = _extract(torch.log(betas))
    frac = (pred_vars + 1.0) / 2.0
    log_variance = frac * max_log + (1.0 - frac) * min_log
    return mean, log_variance

def true_mean_logvar(
    iddpm: IDDPM,
    clean_images: torch.Tensor,
    noisy_images: torch.Tensor,
    timesteps: torch.Tensor,
):
    betas = iddpm.betas.to(timesteps.device)
    alphas = iddpm.alphas.to(timesteps.device)
    alphas_cumprod = iddpm.alphas_cumprod.to(timesteps.device)
    alphas_cumprod_prev = iddpm.alphas_cumprod_prev.to(timesteps.device)
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    posterior_log_varianve_clipped = torch.log(torch.concat((posterior_variance[1:2], posterior_variance[1:])))
    posterior_mean_coef1 = betas * alphas_cumprod_prev ** 0.5 / (1.0 - alphas_cumprod)
    posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * alphas ** 0.5 / (1.0 - alphas_cumprod)
    _extract = partial(extract, timesteps=timesteps, broadcast_shape=noisy_images.shape)
    posterior_mean = _extract(posterior_mean_coef1) * clean_images + _extract(posterior_mean_coef2) * noisy_images
    posterior_log_variance_clipped = _extract(posterior_log_varianve_clipped)
    return posterior_mean, posterior_log_variance_clipped

def gaussian_kl_divergence(
    mean_1: torch.Tensor,
    logvar_1: torch.Tensor,
    mean_2: torch.Tensor,
    logvar_2: torch.Tensor,
):
    return 0.5 * (
        - 1.0
        + logvar_2
        - logvar_1
        + torch.exp(logvar_1 - logvar_2)
        + ((mean_1 - mean_2) ** 2) * torch.exp(-logvar_2)
    )

def approx_standard_normal_cdf(x):
    return 0.5 * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

def gaussian_nll(
    clean_images: torch.Tensor,
    pred_mean: torch.Tensor,
    pred_logvar: torch.Tensor,
):
    centered_x = clean_images - pred_mean
    inv_stdv = torch.exp(-pred_logvar)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in).clamp_min(1e-12)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in).clamp_min(1e-12)
    cdf_delta = (cdf_plus - cdf_min).clamp_min(1e-12)
    log_cdf_plus = torch.log(cdf_plus)
    log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp_min(1e-12))
    log_probs = torch.log(cdf_delta.clamp_min(1e-12))
    log_probs[clean_images < -0.999] = log_cdf_plus[clean_images < -0.999]
    log_probs[clean_images > 0.999] = log_one_minus_cdf_min[clean_images > 0.999]
    return log_probs

def vlb_loss(
    iddpm: IDDPM,
    pred_noises: torch.Tensor,
    pred_vars: torch.Tensor,
    clean_images: torch.Tensor,
    noisy_images: torch.Tensor,
    timesteps: torch.Tensor,
):
    pred_mean, pred_logvar = pred_mean_logvar(iddpm, pred_noises, pred_vars, noisy_images, timesteps)
    true_mean, true_logvar = true_mean_logvar(iddpm, clean_images, noisy_images, timesteps)
    kl = gaussian_kl_divergence(true_mean, true_logvar, pred_mean, pred_logvar)
    kl = kl.mean(dim=list(range(1, len(kl.shape)))) / math.log(2.0)
    nll = gaussian_nll(clean_images, pred_mean, pred_logvar * 0.5)
    nll = nll.mean(dim=list(range(1, len(nll.shape)))) / math.log(2.0)
    results = torch.where(timesteps == 0, nll, kl)
    return results

def training_losses(
    iddpm: IDDPM,
    model: UNet2DModel,
    clean_images: torch.Tensor,
    noise: torch.Tensor,
    noisy_images: torch.Tensor,
    timesteps: torch.Tensor,
    vlb_weight: float = 1e-3,
) -> torch.Tensor:
    _, channels, _, _ = noisy_images.shape
    pred: torch.Tensor = model(noisy_images, timesteps, return_dict=False)[0]
    pred_noises, pred_vars = torch.split(pred, channels, dim=1)
    l_simple = (pred_noises - noise) ** 2
    l_simple = l_simple.mean(dim=list(range(1, len(l_simple.shape))))
    l_vlb = vlb_loss(iddpm, pred_noises.detach(), pred_vars, clean_images, noisy_images, timesteps)
    return l_simple + vlb_weight * l_vlb


model = model.cuda()
iddpm = IDDPM()
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(dataloader) * config.num_epochs),
)
importance_sampler = ImportanceSampler()
accelerator = Accelerator(
    mixed_precision=config.mixed_precision,
    gradient_accumulation_steps=config.gradient_accumulation_steps,
    log_with="tensorboard",
    project_dir=os.path.join(config.output_dir, "logs"),
)
model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
    model, optimizer, dataloader, lr_scheduler
)
global_step = 0
for epoch in range(config.num_epochs):
    progress_bar = tqdm(total=len(dataloader), disable=not accelerator.is_local_main_process, desc=f'Epoch {epoch}')
    for steps, batch in enumerate(dataloader):
        clean_images, _ = batch
        noise = torch.randn(clean_images.shape, device=clean_images.device)
        bs = clean_images.shape[0]
        timesteps, weights = importance_sampler.sample(bs)
        timesteps = timesteps.to(clean_images.device)
        weights = weights.to(clean_images.device)
        noisy_images = iddpm.add_noise(clean_images, noise, timesteps)
        with accelerator.accumulate(model):
            losses = training_losses(iddpm, model, clean_images, noise, noisy_images, timesteps)
            importance_sampler.update(timesteps, losses)
            loss = (losses * weights).mean()
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        progress_bar.update(1)
        logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
        progress_bar.set_postfix(**logs)
        accelerator.log(logs, step=global_step)
        global_step += 1
    if accelerator.is_main_process:
        images = iddpm.sample(model, config.eval_batch_size, 3, config.image_size)
        image_grid = make_image_grid(numpy_to_pil(images), rows=8, cols=8)
        samples_dir = os.path.join(config.output_dir, 'samples')
        os.makedirs(samples_dir, exist_ok=True)
        image_grid.save(os.path.join(samples_dir, f'{global_step}.png'))
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model.module.save_pretrained(config.output_dir)
        else:
            model.save_pretrained(config.output_dir)
