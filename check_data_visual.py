import sys
import os
import cv2
import numpy as np
import torch

sys.path.append(os.getcwd())

from src.data.dataset_pairs import UpscaleDataset

def tensor_to_numpy(tensor):
    """(C, H, W) Tensor -> (H, W, C) Numpy Image BGR"""
    img = tensor.detach().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = (img * 255.0).clip(0, 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

def main():
    try:
        dataset = UpscaleDataset(
            lr_dir="data/train_lr",
            hr_dir="data/train_hr"
        )
    except Exception as e:
        print(f"Fail: {e}")
        return

    print(f"Dataset Find {len(dataset)} images")

    if len(dataset) == 0:
        print("NO images yet")
        return

    idx = 0  # show which images
    
    lr_tensor, hr_tensor = dataset[idx]
    
    print(f" - LR Shape: {lr_tensor.shape}")
    print(f" - HR Shape: {hr_tensor.shape}")
    lr_img = tensor_to_numpy(lr_tensor)    
    hr_img = tensor_to_numpy(hr_tensor)
    h, w, _ = hr_img.shape
    lr_resized = cv2.resize(lr_img, (w, h), interpolation=cv2.INTER_NEAREST)
    sep = np.zeros((h, 10, 3), dtype=np.uint8)
    comparison = np.hstack((lr_resized, sep, hr_img))

    output_filename = "test_dataset_check.png"
    cv2.imwrite(output_filename, comparison)
    
    print(" (LR) | (HR)")

if __name__ == "__main__":
    main()