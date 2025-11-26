import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import cv2
import math
import os
from tqdm import tqdm

from src.models.srcnn import SRCNN
from src.models.unet_sr import UNetSR
from src.data.dataset_pairs import UpscaleDataset

def calculate_psnr(img1, img2):
    """計算 PSNR (img1, img2 必須是 0~255 的 Numpy Array)"""
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def calculate_ssim(img1, img2):
    """
    使用 OpenCV 手動實作 SSIM
    img1, img2: Numpy Array (H, W, C), 數值範圍 0~255
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

def tensor_to_numpy(tensor):
    """(C, H, W) Tensor -> (H, W, C) Numpy Image [0, 255]"""
    img = tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    img = (img * 255.0).clip(0, 255)
    return img

def load_model(model_class, path, device):
    """載入模型的輔助函數"""
    if not os.path.exists(path):
        print(f"can't find path")
        return None
    model = model_class().to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))    
    model.eval()
    return model

def main():
    LR_DIR = 'data/train_lr'
    HR_DIR = 'data/train_hr'
    
    SRCNN_PATH = 'models_ckpt/srcnn_final.pth'
    UNET_PATH = 'models_ckpt/unet_final.pth'
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_srcnn = load_model(SRCNN, SRCNN_PATH, DEVICE)
    model_unet = load_model(UNetSR, UNET_PATH, DEVICE)
    dataset = UpscaleDataset(lr_dir=LR_DIR, hr_dir=HR_DIR)
    if len(dataset) == 0:
        print("❌ Dataset is empty.")
        return

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    results = {
        "Nearest": {"psnr": 0.0, "ssim": 0.0},
        "Bicubic": {"psnr": 0.0, "ssim": 0.0},
        "SRCNN":   {"psnr": 0.0, "ssim": 0.0},
        "UNet":    {"psnr": 0.0, "ssim": 0.0}
    }

    with torch.no_grad():
        for lr_tensor, hr_tensor in tqdm(dataloader):
            hr_img = tensor_to_numpy(hr_tensor) 
            h, w, c = hr_img.shape
            
            lr_img_np = tensor_to_numpy(lr_tensor) 
            lr_input = lr_tensor.to(DEVICE)        
            pred_nearest = cv2.resize(lr_img_np, (w, h), interpolation=cv2.INTER_NEAREST)
            results["Nearest"]["psnr"] += calculate_psnr(pred_nearest, hr_img)
            results["Nearest"]["ssim"] += calculate_ssim(pred_nearest, hr_img)
            pred_bicubic = cv2.resize(lr_img_np, (w, h), interpolation=cv2.INTER_CUBIC)
            results["Bicubic"]["psnr"] += calculate_psnr(pred_bicubic, hr_img)
            results["Bicubic"]["ssim"] += calculate_ssim(pred_bicubic, hr_img)
            if model_srcnn:
                out = model_srcnn(lr_input)
                pred_srcnn = tensor_to_numpy(out)
                results["SRCNN"]["psnr"] += calculate_psnr(pred_srcnn, hr_img)
                results["SRCNN"]["ssim"] += calculate_ssim(pred_srcnn, hr_img)
            if model_unet:
                out = model_unet(lr_input)
                pred_unet = tensor_to_numpy(out)
                results["UNet"]["psnr"] += calculate_psnr(pred_unet, hr_img)
                results["UNet"]["ssim"] += calculate_ssim(pred_unet, hr_img)
    n = len(dataset)
    
    print("\n" + "="*65)
    print(f"{'Method':<15} | {'Average PSNR (dB)':<20} | {'Average SSIM':<15}")
    print("-" * 65)

    def print_row(name, data_dict):
        avg_psnr = data_dict["psnr"] / n
        avg_ssim = data_dict["ssim"] / n
        if avg_psnr == 0 and name not in ["Nearest", "Bicubic"]:
            print(f"{name:<15} | {'-- Not Loaded --':<20} | {'--':<15}")
        else:
            print(f"{name:<15} | {avg_psnr:<20.4f} | {avg_ssim:<15.4f}")
        return avg_psnr, avg_ssim

    _, _ = print_row("Nearest", results["Nearest"])
    _, _ = print_row("Bicubic", results["Bicubic"])
    psnr_srcnn, ssim_srcnn = print_row("SRCNN", results["SRCNN"])
    psnr_unet, ssim_unet = print_row("UNet", results["UNet"])
    
    print("="*65)
    
    if model_unet and model_srcnn:
        diff_psnr = (results['UNet']['psnr'] - results['SRCNN']['psnr']) / n
        diff_ssim = (results['UNet']['ssim'] - results['SRCNN']['ssim']) / n
        
        print("🔎 Analysis:")
        if diff_psnr > 0:
            print(f"PSNR: U-Net 領先 SRCNN {diff_psnr:.4f} dB")
        else:
            print(f"PSNR: U-Net 落後 SRCNN {abs(diff_psnr):.4f} dB")
            
        if diff_ssim > 0:
            print(f"SSIM: U-Net 結構還原度優於 SRCNN (+{diff_ssim:.4f})")
        else:
            print(f"SSIM: U-Net 結構還原度稍差 ({diff_ssim:.4f})")

if __name__ == "__main__":
    main()