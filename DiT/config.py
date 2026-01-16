from dataclasses import dataclass


@dataclass
class TrainingConfig:
    image_size: int = 64
    latent_size: int = 8
    train_batch_size: int = 1024
    eval_batch_size: int = 64
    num_epochs: int = 10
    gradient_accumulation_steps = 1
    learning_rate: float = 1e-4
    lr_warmup_steps: int = 500
    mixed_precision: str = "fp16"
    output_dir: str = "dit-results"


config = TrainingConfig()
