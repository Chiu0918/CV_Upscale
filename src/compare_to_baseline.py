import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset 
import numpy as np
import cv2
import math
import os
import argparse
import random 
from tqdm import tqdm


from src.data.dataset_pairs import UpscaleDataset
from src.models.srcnn import SRCNN
from src.models.unet_sr import UNetSR
try:
    from src.models.edsr import EDSR
except ImportError:
    EDSR = None
try:
    from src.models.srgan import Generator as SRGAN_Generator
except ImportError:
    SRGAN_Generator = None

def calculate_psnr(img1, img2):
    """PSNR"""
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def calculate_ssim(img1, img2):
    """ SSIM"""
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

def load_model(model_name, path, device):
    if not os.path.exists(path):
        print(f"Warning: Model path not found: {path}")
        return None
    
    model = None
    if model_name == "srcnn":
        model = SRCNN().to(device)
    elif model_name == "unet":
        model = UNetSR().to(device)
    elif model_name == "edsr":
        if EDSR: model = EDSR(n_resblocks=16, n_feats=64, scale=4).to(device)
    elif model_name == "srgan":
        if SRGAN_Generator: model = SRGAN_Generator().to(device)
    
    if model is None:
        print(f"Error: Failed to initialize model {model_name}")
        return None

    try:
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    except:
        model.load_state_dict(torch.load(path, map_location=device))
        
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser(description="Compare SR Models")
    parser.add_argument("--srcnn", type=str, default=None, help="Path to SRCNN checkpoint")
    parser.add_argument("--unet", type=str, default=None, help="Path to UNet checkpoint")
    parser.add_argument("--edsr", type=str, default=None, help="Path to EDSR checkpoint")
    parser.add_argument("--srgan", type=str, default=None, help="Path to SRGAN checkpoint")
    
    args = parser.parse_args()

    LR_DIR = 'data/val_lr'
    HR_DIR = 'data/val_hr'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    models = {}
    if args.srcnn: models["SRCNN"] = load_model("srcnn", args.srcnn, DEVICE)
    if args.unet:  models["UNet"]  = load_model("unet", args.unet, DEVICE)
    if args.edsr:  models["EDSR"]  = load_model("edsr", args.edsr, DEVICE)
    if args.srgan: models["SRGAN"] = load_model("srgan", args.srgan, DEVICE)

    dataset = UpscaleDataset(lr_dir=LR_DIR, hr_dir=HR_DIR)
    if len(dataset) == 0:
        print("❌ Dataset is empty.")
        return
    total_images = len(dataset)
    NUM_SAMPLES = 10 

    if total_images > NUM_SAMPLES:
        print(f"🎲 Randomly selecting {NUM_SAMPLES} ...")
        indices = random.sample(range(total_images), NUM_SAMPLES)
        dataset = Subset(dataset, indices)
    else:
        print(f"Dataset has only {total_images} images, using all of them.")
    
    n = len(dataset)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    results = {
        "Nearest": {"psnr": 0.0, "ssim": 0.0},
        "Bicubic": {"psnr": 0.0, "ssim": 0.0}
    }
    for name in models.keys():
        results[name] = {"psnr": 0.0, "ssim": 0.0}

    print("Starting comparison...")
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
            for name, model in models.items():
                if model:
                    out = model(lr_input)
                    pred = tensor_to_numpy(out)
                    if pred.shape != hr_img.shape:
                        pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_CUBIC)
                        
                    results[name]["psnr"] += calculate_psnr(pred, hr_img)
                    results[name]["ssim"] += calculate_ssim(pred, hr_img)

    print("\n" + "="*65)
    print(f"📊 Results (Averaged over {n} random samples)")
    print(f"{'Method':<15} | {'Average PSNR (dB)':<20} | {'Average SSIM':<15}")
    print("-" * 65)

    def print_row(name, data_dict):
        avg_psnr = data_dict["psnr"] / n
        avg_ssim = data_dict["ssim"] / n
        print(f"{name:<15} | {avg_psnr:<20.4f} | {avg_ssim:<15.4f}")
    print_row("Nearest", results["Nearest"])
    print_row("Bicubic", results["Bicubic"])
    for name in models.keys():
        print_row(name, results[name])
    print("="*65)
if __name__ == "__main__":
    main()