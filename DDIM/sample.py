from diffusers.utils import make_image_grid, numpy_to_pil
import os
from model import model, DDIM


ddim = DDIM()
out_dir = "ddim-results"
os.makedirs(out_dir, exist_ok=True)
sample_images = ddim.sample(model, 64, 3, 64)
sample_image_grad = make_image_grid(numpy_to_pil(sample_images), rows=8, cols=8)
sample_image_grad.save(os.path.join(out_dir, "sample.png"))
interpolation_images = ddim.interpolation(model, 8, 8, 3, 64)
interpolation_image_grad = make_image_grid(numpy_to_pil(interpolation_images), rows=8, cols=8)
interpolation_image_grad.save(os.path.join(out_dir, "interpolation.png"))
