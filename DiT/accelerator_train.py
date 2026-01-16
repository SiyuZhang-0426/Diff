import torch
from torchvision.utils import save_image
from diffusers.optimization import get_cosine_schedule_with_warmup
import os
from tqdm import tqdm
from accelerate import Accelerator
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from config import config
from dataloader import dataloader, num_classes
from models import model


os.makedirs(config.output_dir, exist_ok=True)
os.makedirs(os.path.join(config.output_dir, "samples"), exist_ok=True)
model = model.cuda()
diffusion = create_diffusion(timestep_respacing="")
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(dataloader) * config.num_epochs),
)
accelerator = Accelerator(
    mixed_precision=config.mixed_precision,
    gradient_accumulation_steps=config.gradient_accumulation_steps,
    log_with="tensorboard",
    project_dir=os.path.join(config.output_dir, "logs"),
)
model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
    model, optimizer, dataloader, lr_scheduler
)
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(accelerator.device)
device = accelerator.device
global_step = 0
for epoch in range(config.num_epochs):
    process_bar = tqdm(total=len(dataloader), disable=not accelerator.is_local_main_process, desc=f"Epoch {epoch}")
    for steps, batch in enumerate(dataloader):
        model.train()
        clean_images, labels = batch
        with torch.no_grad():
            latent_images = vae.encode(clean_images).latent_dist.sample().mul_(0.18215)
        timesteps = torch.randint(
            0, diffusion.num_timesteps, (latent_images.shape[0],), device=device,
            dtype=torch.int64
        )
        model_kwargs = dict(y=labels)
        with accelerator.accumulate(model):
            loss_dict = diffusion.training_losses(model, latent_images, timesteps, model_kwargs)
            loss = loss_dict["loss"].mean()
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        process_bar.update(1)
        current_loss = loss.detach().item()
        logs = {"loss": current_loss, "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
        process_bar.set_postfix(**logs)
        accelerator.log(logs, step=global_step)
        global_step += 1
    if accelerator.is_local_main_process:
        model.eval()
        unwrapped_model = accelerator.unwrap_model(model)
        with torch.no_grad():
            sample_noise = torch.randn(
                (config.eval_batch_size, 4, config.latent_size, config.latent_size),
                device=device
            )
            guided_classes = torch.randint(
                low=0, high=num_classes, size=(config.eval_batch_size,), device=device
            )
            sample_noise = torch.cat([sample_noise, sample_noise], dim=0)
            null_classes = torch.tensor([num_classes] * config.eval_batch_size, device=device)
            labels = torch.cat([guided_classes, null_classes], dim=0)
            def model_wrapper(x, t, **kwargs):
                return unwrapped_model.forward_with_cfg(x, t, kwargs['y'], kwargs['cfg_scale'])
            
            model_kwargs = dict(y=labels, cfg_scale=4.0)
            samples = diffusion.p_sample_loop(
                model_wrapper,
                sample_noise.shape,
                sample_noise,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                progress=True,
                device=device,
            )
            samples, _ = samples.chunk(2, dim=0)
            decoded = vae.decode(samples / 0.18215).sample
        save_image(decoded, os.path.join(config.output_dir, "samples", f"{global_step}.png"), nrow=8, normalize=True, value_range=(-1, 1))
        torch.save(unwrapped_model.state_dict(), os.path.join(config.output_dir, "model.pth"))
accelerator.end_training()
