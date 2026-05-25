from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor


class NoisyImageDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        exts = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff', '*.tif')
        self.paths = []
        for ext in exts:
            self.paths.extend(sorted(self.data_path.glob(ext)))
        self.transform = Compose([ToTensor()])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('L')
        return self.transform(img), str(self.paths[idx])
