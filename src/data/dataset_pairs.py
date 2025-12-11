import os
import random
from glob import glob
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

class UpscaleDataset(Dataset):
    """
    Lazy Loading
    """
    def __init__(
        self,
        lr_dir: str,
        hr_dir: str,
        transform: Optional[Callable] = None,
        patch_size: Optional[int] = None,
        scale_factor: int = 4
    ) -> None:
        super().__init__()
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.transform = transform
        self.patch_size = patch_size
        self.scale = scale_factor
        if not os.path.exists(lr_dir):
            raise FileNotFoundError(f"Low-resolution directory not found: {lr_dir}")
        if not os.path.exists(hr_dir):
            raise FileNotFoundError(f"High-resolution directory not found: {hr_dir}")
        valid_extensions = ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG")
        lr_paths = []
        hr_paths = []
        for ext in valid_extensions:
            lr_paths.extend(glob(os.path.join(lr_dir, ext)))
            hr_paths.extend(glob(os.path.join(hr_dir, ext)))
            
        lr_paths = sorted(lr_paths)
        hr_paths = sorted(hr_paths)
        lr_names = {os.path.basename(p) for p in lr_paths}
        hr_names = {os.path.basename(p) for p in hr_paths}
        common_names =  sorted(set(lr_names.intersection(hr_names)))
        if not common_names:
            raise ValueError(f"No matching image pairs found between {lr_dir} and {hr_dir}.")
        
        self.pairs: List[Tuple[str, str]] = [
            (os.path.join(lr_dir, name), os.path.join(hr_dir, name)) for name in common_names
        ]
        
        print(f"Found {len(self.pairs)} image pairs.")

    def __len__(self) -> int:
        return len(self.pairs)
    
    def __load_image(self, path: str) -> np.ndarray:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Image corrupted or empty: {path}")
        return img
        
    def __to_tensor(self, img: np.ndarray) -> torch.Tensor:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return TF.to_tensor(img)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        lr_path, hr_path = self.pairs[index]
        
        try:
            lr_image = self.__load_image(lr_path)
            hr_image = self.__load_image(hr_path)
        except Exception as e:
            print(f"Error loading {lr_path}: {e}. Skipping...")
            return self.__getitem__(random.randint(0, len(self.pairs) - 1))
        if self.patch_size is not None:
            lr_h, lr_w, _ = lr_image.shape
            tp = self.patch_size
            ip = tp // self.scale
            
            if lr_w < ip or lr_h < ip:
                pass 
            else:
                ix = random.randrange(0, lr_w - ip + 1)
                iy = random.randrange(0, lr_h - ip + 1)
                tx, ty = ix * self.scale, iy * self.scale
                
                lr_image = lr_image[iy:iy+ip, ix:ix+ip, :]
                hr_image = hr_image[ty:ty+tp, tx:tx+tp, :]
        lr_tensor = self.__to_tensor(lr_image)
        hr_tensor = self.__to_tensor(hr_image)
        if random.random() > 0.5:
            lr_tensor = TF.hflip(lr_tensor)
            hr_tensor = TF.hflip(hr_tensor)
        if random.random() > 0.5:
            lr_tensor = TF.vflip(lr_tensor)
            hr_tensor = TF.vflip(hr_tensor)
        if random.random() > 0.5:
            lr_tensor = torch.rot90(lr_tensor, 1, [1, 2])
            hr_tensor = torch.rot90(hr_tensor, 1, [1, 2])

        return lr_tensor, hr_tensor