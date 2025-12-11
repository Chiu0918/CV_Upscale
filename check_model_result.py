import sys
import os
import cv2
import numpy as np
import torch
import argparse

sys.path.append(os.getcwd())
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
try:
    from src.models.srgan import Generator as ESRGAN_Generator
except ImportError:
    ESRGAN_Generator = None

def tensor_to_numpy(tensor):
    """Tensor ->(H, W, C) BGR"""
    img = tensor.squeeze(0).detach().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = (img * 255.0).clip(0, 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

def main():
    parser = argparse.ArgumentParser(description="Check SR Model Results")
    parser.add_argument("-m", "--model", type=str, required=True, 
                        choices=["unet", "srcnn", "edsr", "srgan", "esrgan"],
                        help="Select: unet, srcnn, edsr, srgan, esrgan")
    parser.add_argument("-c", "--checkpoint", type=str, required=True, 
                        help=" (.pth)")
    parser.add_argument("--save-name", type=str, default=None, 
                        help="image file name")
    
    args = parser.parse_args()
    MODEL_NAME = args.model
    MODEL_PATH = args.checkpoint
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Loading {MODEL_NAME} from {MODEL_PATH}...")
    if MODEL_NAME == "srcnn":
        model = SRCNN().to(DEVICE)
    elif MODEL_NAME == "unet":
        model = UNetSR().to(DEVICE)
    elif MODEL_NAME == "edsr":
        if EDSR is None:
            print("cant find src/models/edsr.py")
            return
        model = EDSR(n_resblocks=16, n_feats=64, scale=4).to(DEVICE)
        
    elif MODEL_NAME == "srgan":
        if SRGAN_Generator is None:
            print("cant find src/models/srgan.py")
            return
        model = SRGAN_Generator().to(DEVICE)
        
    else:
        print(f"Unknown model: {MODEL_NAME}")
        return
    
    if not os.path.exists(MODEL_PATH):
        print(f"Fail to find file: {MODEL_PATH}")
        return

    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    except:
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        except RuntimeError as e:
            print(f"Fail to upload")
            return
        
    model.eval() 
    print(f"Model loaded successfully!")
    VAL_LR = "data/val_lr"
    VAL_HR = "data/val_hr"
    
    if os.path.exists(VAL_LR) and len(os.listdir(VAL_LR)) > 0:
        print("Testing on VALIDATION dataset (Unseen data)")
        dataset = UpscaleDataset(lr_dir=VAL_LR, hr_dir=VAL_HR)
    else:
        print("Testing on TRAIN dataset")
        dataset = UpscaleDataset(lr_dir="data/train_lr", hr_dir="data/train_hr")

    if len(dataset) == 0:
        print("Dataset is empty.")
        return
    idx = np.random.randint(0, len(dataset))
    print(f"Processing image index: {idx}")

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
    if args.save_name:
        output_file = args.save_name
    else:
        ckpt_name = os.path.basename(MODEL_PATH).replace(".pth", "")
        output_file = f"result_{MODEL_NAME}_{ckpt_name}.png"
        
    cv2.imwrite(output_file, result_img)
    print(f"🎉 Saved result to {output_file}")
    print("左: LR (Nearest) | 中: SR Result | 右: HR (Ground Truth)")

if __name__ == "__main__":
    main()