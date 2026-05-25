from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor


class NoisyImageDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        exts = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff', '*.tif')
        self.paths = []
        for ext in exts:
            self.paths.extend(sorted(self.data_path.glob(ext)))
        self.transform = to_tensor

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx])
        if img.mode in ('I;16', 'I;16B', 'I;16L'):
            arr = np.array(img, dtype=np.uint16)
            tensor = torch.from_numpy(arr).unsqueeze(0).float() / 65535.0
        else:
            img = img.convert('L')
            tensor = self.transform(img)
        return tensor, str(self.paths[idx])
