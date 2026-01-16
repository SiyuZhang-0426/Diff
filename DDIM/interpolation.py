import torch


def slerp(
    x0: torch.Tensor,
    x1: torch.Tensor,
    alpha: float,
):
    theta = torch.acos(torch.sum(x0 * x1) / (torch.norm(x0) * torch.norm(x1)))
    w0 = torch.sin((1-alpha) * theta) / torch.sin(theta)
    w1 = torch.sin(alpha * theta) / torch.sin(theta)
    return w0 * x0 + w1 * x1

def interpolation_grid(
    rows: int,
    cols: int,
    in_channels: int,
    sample_size: int,
):
    images = torch.zeros((rows * cols, in_channels, sample_size, sample_size), dtype=torch.float32)
    images[0, ...] = torch.randn_like(images[0, ...])
    images[cols - 1, ...] = torch.randn_like(images[0, ...])
    images[(rows - 1) * cols, ...] = torch.randn_like(images[0, ...])
    images[-1, ...] = torch.randn_like(images[0, ...])
    for row in range(1, rows - 1):
        alpha = row / (rows - 1)
        images[row * cols, ...] = slerp(images[0, ...], images[(rows - 1) * cols, ...], alpha)
        images[(row + 1) * cols - 1, ...] = slerp(images[cols - 1, ...], images[-1, ...], alpha)
    for col in range(1, cols - 1):
        alpha = col / (cols - 1)
        images[col::cols, ...] = slerp(images[0::cols, ...], images[cols - 1::cols, ...], alpha)
    return images
