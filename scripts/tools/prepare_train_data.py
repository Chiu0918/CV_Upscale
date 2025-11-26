import os
from glob import glob

# 匯入工具
from src.data.degrade import load_image, save_image, official_downscale 

def make_dirs(dir_path: str):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def generate_lr_images(hr_dir: str, lr_dir: str):
    """
    Generate low-resolution images from high-resolution images
    using the official downscaling method.
    """
    if not os.path.exists(hr_dir):
        raise FileNotFoundError(f"High-resolution directory not found: {hr_dir}")
    make_dirs(lr_dir)
    
    # 支援多種副檔名
    extensions = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG']
    img_paths = []
    for ext in extensions:
        img_paths.extend(glob(os.path.join(hr_dir, ext)))
    
    # 去除重複並排序
    img_paths = sorted(list(set(img_paths)))
    
    print(f"Found {len(img_paths)} images in {hr_dir}.")
    
    for img_path in img_paths:
        try:
            img_name = os.path.basename(img_path)
            hr_img = load_image(img_path)
            lr_img = official_downscale(hr_img)
            save_path = os.path.join(lr_dir, img_name)
            try:
                save_image(lr_img, save_path)
            except TypeError:
                save_image(save_path, lr_img)
            
            print(f"Processed {img_name}: {hr_img.shape} -> {lr_img.shape}")
        except Exception as e:
            print(f"Error processing {img_name}: {e}")
        
if __name__ == "__main__":
    hr_directory = "data/train_hr"
    lr_directory = "data/train_lr"
    
    generate_lr_images(hr_directory, lr_directory)