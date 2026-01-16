from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
from config import config


def get_transforms():
    preprocess = transforms.Compose([
        transforms.Resize(config.image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    return preprocess

class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [
            f for f in os.listdir(root_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
        ]
        self.image_files.sort()
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, 0


local_data_dir = "/mnt/petrelfs/zhangsiyu/Diff/data"
dataset = ImageDataset(root_dir=local_data_dir, transform=get_transforms())
dataloader = DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)
