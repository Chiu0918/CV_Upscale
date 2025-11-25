import os
from glob import glob
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class UpscaleDataset(Dataset):
    """
    A dataset class for loading pairs of low-resolution and high-resolution images
    for image super-resolution tasks.
    Args:
        root_dir (str): Root directory containing 'lr_dir' and 'hr_dir' subdirectories.
        transform (Optional[Callable]): Optional transform to be applied on a sample.
        lr_ext (str): File extension for low-resolution
        hr_ext (str): File extension for high-resolution
    """
    
    def __init__(
        self,
        lr_dir: str,
        hr_dir: str,
        file_ext: str = "*.jpg",
        transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.transform = transform
        
        if not os.path.exists(lr_dir):
            raise FileNotFoundError(f"Low-resolution directory not found: {lr_dir}")
        if not os.path.exists(hr_dir):
            raise FileNotFoundError(f"High-resolution directory not found: {hr_dir}")
        
        lr_paths = sorted(glob(os.path.join(lr_dir, file_ext)))
        hr_paths = sorted(glob(os.path.join(hr_dir, file_ext)))
        
        lr_names = {os.path.basename(p) for p in lr_paths}
        hr_names = {os.path.basename(p) for p in hr_paths}
        
        common_names =  sorted(set(lr_names.intersection(hr_names)))
        if not common_names:
            raise ValueError("No matching image pairs found between LR and HR directories.")
        
        self.pairs: List[Tuple[str, str]] = [
            (os.path.join(lr_dir, name), os.path.join(hr_dir, name)) for name in common_names
        ]
        
        print(f"Found {len(self.pairs)} image pairs.")
        
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __load_image(self, path: str) -> np.ndarray:
        """
        Load an image from a file path.
        Args:
            path (str): Path to the image file.
            Returns:
            np.ndarray: Loaded image in BGR format.
        """
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Image not found: {path}")
        return img
        
    def __to_tensor(self, img: np.ndarray) -> torch.Tensor:
        """
        Convert a numpy image to a PyTorch tensor.
        Args:
            img (np.ndarray): Image in HWC format.
        Returns:
            torch.Tensor: Image in CHW format as a float tensor.
        """
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC to CHW
        return torch.from_numpy(img)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        lr_path, hr_path = self.pairs[index]
        
        lr_image = self.__load_image(lr_path)
        hr_image = self.__load_image(hr_path)
        
        if self.transform:
            lr_image, hr_image = self.transform(lr_image, hr_image)
        
        lr_tensor = self.__to_tensor(lr_image)
        hr_tensor = self.__to_tensor(hr_image)
        
        return lr_tensor, hr_tensor
    
if __name__ == "__main__":
    # Example usage
    dataset = UpscaleDataset(
        lr_dir="data/train_lr",
        hr_dir="data/train_hr",
    )
    
    print(f"Dataset size: {len(dataset)}")
    lr, hr = dataset[0]
    print(f"LR shape: {lr.shape}, HR shape: {hr.shape}")
