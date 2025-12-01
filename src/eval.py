import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import cv2  # 用於計算 SSIM
import math
import os
import argparse
from tqdm import tqdm

from src.data.dataset_pairs import UpscaleDataset
from src.models.srcnn import SRCNN
from src.models.unet_sr import UNetSR

def calculate_psnr(img1, img2):
    """
    計算兩張影像的 PSNR (Peak Signal-to-Noise Ratio)
    img1, img2: Numpy Array (H, W, C), 數值範圍 0~255
    """
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def calculate_ssim(img1, img2):
    """
    使用 OpenCV 手動實作 SSIM
    img1, img2: Numpy Array (H, W, C), 數值範圍 0~255
    回傳: SSIM 分數 (float, 0.0 ~ 1.0)
    """
    I1 = img1.astype(np.float64)
    I2 = img2.astype(np.float64)
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    
    kernel_size = 11
    sigma = 1.5

    mu1 = cv2.GaussianBlur(I1, (kernel_size, kernel_size), sigma)
    mu2 = cv2.GaussianBlur(I2, (kernel_size, kernel_size), sigma)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(I1**2, (kernel_size, kernel_size), sigma) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(I2**2, (kernel_size, kernel_size), sigma) - mu2_sq
    sigma12 = cv2.GaussianBlur(I1 * I2, (kernel_size, kernel_size), sigma) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def evaluate(model_name=None, lr_dir='data/train_lr', hr_dir='data/train_hr', checkpoint=None):
    
    if model_name is None:
        if checkpoint is None:
            raise ValueError("Either model_name or checkpoint must be provided.")
        if "srcnn" in checkpoint.lower():
            MODEL_NAME = "srcnn"
        elif "unet" in checkpoint.lower():
            MODEL_NAME = "unet"
        else:
            raise ValueError("Unable to infer model type from checkpoint name. Please specify model_name manually.")
    else:
        MODEL_NAME = model_name    
        
    LR_DIR = lr_dir
    HR_DIR = hr_dir
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    if MODEL_NAME == "srcnn":
        ModelClass = SRCNN
        MODEL_PATH = 'models_ckpt/srcnn_final.pth'
    elif MODEL_NAME == "unet":
        ModelClass = UNetSR
        MODEL_PATH = 'models_ckpt/unet_final.pth'
    else:
        print(f"Unknown: {MODEL_NAME}")
        return
    if checkpoint is not None:
        MODEL_PATH = checkpoint
        
    if not os.path.exists(MODEL_PATH):
        print(f"Fail: {MODEL_PATH}")
        return

    model = ModelClass().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval() 
    dataset = UpscaleDataset(lr_dir=LR_DIR, hr_dir=HR_DIR)
    
    if len(dataset) == 0:
        print(" Dataset is empty.")
        return

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    total_psnr = 0.0
    total_ssim = 0.0    
    with torch.no_grad():
        for lr_tensor, hr_tensor in tqdm(dataloader):
            lr_tensor = lr_tensor.to(DEVICE)
            output_tensor = model(lr_tensor)
            pred_img = output_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            pred_img = (pred_img * 255.0).clip(0, 255)
            gt_img = hr_tensor.squeeze(0).numpy().transpose(1, 2, 0)
            gt_img = (gt_img * 255.0).clip(0, 255)

            # PSNR
            psnr = calculate_psnr(pred_img, gt_img)
            total_psnr += psnr
            
            # SSIM
            ssim_score = calculate_ssim(pred_img, gt_img)
            total_ssim += ssim_score
    avg_psnr = total_psnr / len(dataset)
    avg_ssim = total_ssim / len(dataset)
    
    
    print("-" * 40)
    print(f"Evaluation Results for {MODEL_NAME.upper()}:")
    print(f"   Average PSNR: {avg_psnr:.4f} dB")
    print(f"   Average SSIM: {avg_ssim:.4f}")
    print("-" * 40)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="unet", choices=["srcnn", "unet"])
    parser.add_argument("--lr-dir", type=str, default="data/train_lr")
    parser.add_argument("--hr-dir", type=str, default="data/train_hr")
    parser.add_argument("--checkpoint", type=str, default=None)

    args = parser.parse_args()

    evaluate(
        model_name=args.model,
        lr_dir=args.lr_dir,
        hr_dir=args.hr_dir,
        checkpoint=args.checkpoint,
    )