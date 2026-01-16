import math
import torch
import numpy as np
from typing import List
from torch.distributed.nn import dist


def make_betas_cosine_schedule(
    num_diffusion_timesteps: int = 1000,
    beta_max: float = 0.999,
    s: float = 8e-3,
):
    fn = lambda t: math.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(1.0 - fn(t2) / fn(t1))
    return torch.tensor(betas, dtype=torch.float32).clamp_max(beta_max)

def extract(
    arr: torch.Tensor,
    timesteps: torch.Tensor,
    broadcast_shape: torch.Size,
):
    arr = arr[timesteps]
    while len(arr.shape) < len(broadcast_shape):
        arr = arr.unsqueeze(-1)
    return arr.expand(broadcast_shape)

class ImportanceSampler:
    def __init__(
        self,
        num_diffusion_timesteps: int = 1000,
        history_per_term: int = 10,
    ):
        self.num_diffusion_timesteps = num_diffusion_timesteps
        self.history_per_term = history_per_term
        self.uniform_prob = 1.0 / num_diffusion_timesteps
        self.loss_history = np.zeros([num_diffusion_timesteps, history_per_term], dtype=np.float64)
        self.loss_counts = np.zeros([num_diffusion_timesteps], dtype=int)
    
    def update(
        self,
        timesteps: torch.Tensor,
        losses: torch.Tensor,
    ):
        if dist.is_initialized():
            world_size = dist.get_world_size()
            batch_sizes = [torch.tensor([0], dtype=torch.int32, device=timesteps.device) for _ in range(world_size)]
            dist.all_gather(batch_sizes, torch.full_like(batch_sizes[0], timesteps.size(0)))
            max_batch_size = max([bs.item() for bs in batch_sizes])
            timestep_batches: List[torch.Tensor] = [torch.zeros(max_batch_size).to(timesteps) for _ in range(world_size)]
            loss_batches: List[torch.Tensor] = [torch.zeros(max_batch_size).to(losses) for _ in range(world_size)]
            dist.all_gather(timestep_batches, timesteps)
            dist.all_gather(loss_batches, losses)
            all_timesteps = [ts.item() for ts_batch, hs in zip(timestep_batches, batch_sizes) for ts in ts_batch[:hs]]
            all_losses = [loss.item() for loss_batch, bs in zip(loss_batches, batch_sizes) for loss in loss_batch[:bs]]
        else:
            all_timesteps = timesteps.tolist()
            all_losses = losses.tolist()
        for timestep, loss in zip(all_timesteps, all_losses):
            if self.loss_counts[timestep] == self.history_per_term:
                self.loss_history[timestep, :-1] = self.loss_history[timestep, 1:]
                self.loss_history[timestep, -1] = loss
            else:
                self.loss_history[timestep, self.loss_counts[timestep]] = loss
                self.loss_counts[timestep] += 1

    def sample(
        self,
        batch_size: int,
    ):
        weights = self.weights
        prob = weights / np.sum(weights)
        timesteps = np.random.choice(self.num_diffusion_timesteps, size=(batch_size,), p=prob)
        weights = 1.0 / (self.num_diffusion_timesteps * prob[timesteps])
        return torch.from_numpy(timesteps).long(), torch.from_numpy(weights).float()
    
    @property
    def weights(self):
        if not np.all(self.loss_counts == self.history_per_term):
            return np.ones([self.num_diffusion_timesteps], dtype=np.float64)
        weights = np.sqrt(np.mean(self.loss_history ** 2, axis=-1))
        weights /= np.sum(weights)
        weights *= 1.0 - self.uniform_prob
        weights += self.uniform_prob / len(weights)
        return weights
