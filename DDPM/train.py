from accelerate import Accelerator
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.utils import make_image_grid, numpy_to_pil
import torch
import torch.nn.functional as F
import os
from tqdm import tqdm
from config import config
from dataloader import dataloader
from model import model, DDPM


model = model.cuda()
ddpm = DDPM()
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
global_step = 0
for epoch in range(config.num_epochs):
    progress_bar = tqdm(total=len(dataloader), disable=not accelerator.is_local_main_process, desc=f"Epoch {epoch}")
    for steps, batch in enumerate(dataloader):
        clean_images, _ = batch
        noise = torch.randn(clean_images.shape, device=clean_images.device)
        bs = clean_images.shape[0]
        timesteps = torch.randint(
            0, ddpm.num_train_timesteps, (bs,), device=clean_images.device,
            dtype=torch.int64
        )
        noisy_images = ddpm.add_noise(clean_images, noise, timesteps)
        with accelerator.accumulate(model):
            noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
            loss = F.mse_loss(noise_pred, noise)
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
    if accelerator.is_local_main_process:
        image = ddpm.sample(model, config.eval_batch_size, 3, config.image_size)
        image_grad = make_image_grid(numpy_to_pil(image), rows=8, cols=8)
        samples_dir = os.path.join(config.output_dir, "samples")
        os.makedirs(samples_dir, exist_ok=True)
        image_grad.save(os.path.join(samples_dir, f"{global_step}.png"))
        model.save_pretrained(config.output_dir)
