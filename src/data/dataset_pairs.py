import os
import random  # <--- 新增這個 import
from glob import glob
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

class UpscaleDataset(Dataset):
    """
    讀取成對的 LR 與 HR 影像。
    包含Data Augmentation。
    """
    
    def __init__(
        self,
        lr_dir: str,
        hr_dir: str,
        transform: Optional[Callable] = None,
        patch_size: Optional[int] = None,
        scale_factor: int = 4,
    ) -> None:
        super().__init__()
        
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.transform = transform
        self.patch_size = patch_size
        self.scale_factor = scale_factor
        
        # 檢查資料夾是否存在
        if not os.path.exists(lr_dir):
            raise FileNotFoundError(f"Low-resolution directory not found: {lr_dir}")
        if not os.path.exists(hr_dir):
            raise FileNotFoundError(f"High-resolution directory not found: {hr_dir}")
        
        # 同時找 png 和 jpg
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
            print(f"DEBUG: LR files found: {len(lr_paths)}")
            print(f"DEBUG: HR files found: {len(hr_paths)}")
            raise ValueError(f"No matching image pairs found between {lr_dir} and {hr_dir}.")
        
        self.pairs: List[Tuple[str, str]] = [
            (os.path.join(lr_dir, name), os.path.join(hr_dir, name)) for name in common_names
        ]
        
        print(f"Found {len(self.pairs)} image pairs.")
        
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __load_image(self, path: str) -> np.ndarray:
        """讀取影像 (BGR)"""
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Image not found or currupted: {path}")
        return img
        
    def __aligned_random_crop(self, lr_img: np.ndarray, hr_img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        對 (LR, HR) 做對齊的隨機裁切 patch。

        lr_img: (H_lr, W_lr, C)
        hr_img: (H_hr, W_hr, C)
        假設 H_hr = H_lr * scale_factor, W_hr = W_lr * scale_factor
        """
        if self.patch_size is None:
            return lr_img, hr_img  # 不啟用 patch training

        ps = self.patch_size
        sf = self.scale_factor

        h_lr, w_lr, _ = lr_img.shape
        h_hr, w_hr, _ = hr_img.shape

        # 安全檢查：尺寸是否符合比例
        if h_hr != h_lr * sf or w_hr != w_lr * sf:
            # 尺寸不符合時，先不要裁切，以免壞掉
            return lr_img, hr_img

        if ps > h_lr or ps > w_lr:
            # patch 太大，無法裁，直接回傳原圖
            return lr_img, hr_img

        # 在 LR 空間隨機選一個左上角
        y_lr = random.randint(0, h_lr - ps)
        x_lr = random.randint(0, w_lr - ps)

        lr_patch = lr_img[y_lr:y_lr+ps, x_lr:x_lr+ps, :]

        # HR 空間對應位置與大小
        y_hr = y_lr * sf
        x_hr = x_lr * sf
        ps_hr = ps * sf
        hr_patch = hr_img[y_hr:y_hr+ps_hr, x_hr:x_hr+ps_hr, :]

        return lr_patch, hr_patch
    
    def __to_tensor(self, img: np.ndarray) -> torch.Tensor:
        """
        將 Numpy (H, W, C) BGR [0, 255]
        轉為 Tensor (C, H, W) RGB [0.0, 1.0]
        """
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return TF.to_tensor(img) 

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        lr_path, hr_path = self.pairs[index]
        lr_image = self.__load_image(lr_path)
        hr_image = self.__load_image(hr_path)
        
        lr_image, hr_image = self.__aligned_random_crop(lr_image, hr_image)
        
        if self.transform:
            lr_image, hr_image = self.transform(lr_image, hr_image)
        lr_tensor = self.__to_tensor(lr_image)
        hr_tensor = self.__to_tensor(hr_image)

        # Data Augmentation

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

if __name__ == "__main__":
    try:
        dataset = UpscaleDataset(
            lr_dir="data/train_lr",
            hr_dir="data/train_hr",
        )
        print(f"Dataset size: {len(dataset)}")
        if len(dataset) > 0:
            lr, hr = dataset[0]
            print(f"LR shape: {lr.shape}, HR shape: {hr.shape}")
    except Exception as e:
        print(f"Test failed: {e}")