import os
from glob import glob

from src.data.degrade import load_image, save_image, official_downscale 

def make_dirs(dir_path: str):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def generate_lr_images(hr_dir: str, lr_dir: str):
    """
    Generate low-resolution images from high-resolution images
    using the official downscaling method.

    Parameters:
        hr_dir (str): Directory containing high-resolution images.
        lr_dir (str): Directory to save low-resolution images.
    """
    if not os.path.exists(hr_dir):
        raise FileNotFoundError(f"High-resolution directory not found: {hr_dir}")
    make_dirs(lr_dir)
    
    img_paths = sorted(glob(os.path.join(hr_dir, '*.jpg'), ))
    
    print(f"Found {len(img_paths)} images in {hr_dir}.")
    
    for img_path in img_paths:
        img_name = os.path.basename(img_path)
        hr_img = load_image(img_path)
        lr_img = official_downscale(hr_img)
        save_image(lr_img, os.path.join(lr_dir, img_name))
        print(f"Processed {img_name}: {hr_img.shape} -> {lr_img.shape}")
        
if __name__ == "__main__":
    hr_directory = "data/train_hr"
    lr_directory = "data/train_lr"
    generate_lr_images(hr_directory, lr_directory)