import sys
import os
import cv2
import numpy as np
import torch

sys.path.append(os.getcwd())
from src.data.dataset_pairs import UpscaleDataset
from src.models.srcnn import SRCNN
from src.models.unet_sr import UNetSR

def tensor_to_numpy(tensor):
    """把 Tensor 轉回圖片格式 (H, W, C) BGR"""
    img = tensor.squeeze(0).detach().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = (img * 255.0).clip(0, 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

def main():
    MODEL_NAME = "srcnn"  # decide which model to check
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    if MODEL_NAME == "srcnn":
        ModelClass = SRCNN
        MODEL_PATH = 'models_ckpt/srcnn_final.pth'
    elif MODEL_NAME == "unet":
        ModelClass = UNetSR
        MODEL_PATH = 'models_ckpt/unet_final.pth'
    else:
        print(f"unknown model: {MODEL_NAME}")
        return
    
    if not os.path.exists(MODEL_PATH):
        print(f"Fail to find: {MODEL_PATH}")
        return

    model = ModelClass().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval() 
    print("Model loaded!")

    dataset = UpscaleDataset(lr_dir="data/train_lr", hr_dir="data/train_hr")
    if len(dataset) == 0:
        print("❌ Dataset is empty.")
        return

    idx = 600 #decide which images to test
    lr_tensor, hr_tensor = dataset[idx]
    lr_input = lr_tensor.unsqueeze(0).to(DEVICE) 

    with torch.no_grad():
        sr_output = model(lr_input)
    lr_img = tensor_to_numpy(lr_input)  
    sr_img = tensor_to_numpy(sr_output)  
    hr_img = tensor_to_numpy(hr_tensor)


    h, w, _ = hr_img.shape
    lr_resized = cv2.resize(lr_img, (w, h), interpolation=cv2.INTER_NEAREST)
    sep = np.zeros((h, 10, 3), dtype=np.uint8)
    result_img = np.hstack((lr_resized, sep, sr_img, sep, hr_img))

    output_file = f"result_{MODEL_NAME}.png"
    cv2.imwrite(output_file, result_img)
    print("LR | result | HR")

if __name__ == "__main__":
    main()