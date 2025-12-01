import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset_pairs import UpscaleDataset
from src.models.srcnn import SRCNN

from src.config.train_config import TrainConfig as cfg

def _build_exp_prefix() -> str:
    """
    根據 TrainConfig 自動產生實驗名稱。
    如果 cfg.exp_name 有填，就使用手動指定的名稱。
    """
    if cfg.exp_name is not None:
        return cfg.exp_name

    # 自動組名：unet_ps32_bs16_lr1e-4 這種格式
    model = cfg.model_name
    ps = cfg.patch_size if cfg.patch_size is not None else "full"
    bs = cfg.batch_size

    # lr → 1e-4 這種字串
    lr_str = f"{cfg.learning_rate:.0e}".replace("-0", "-")  # 1e-04 -> 1e-4

    return f"{model}_ps{ps}_bs{bs}_lr{lr_str}"

def train():
    """
    srcnn的training
    Loss:將MSE改成L1
    加上learning rate decay
    """
    
    cfg.model_name = "srcnn"
    
    LR_DIR = cfg.lr_dir      
    HR_DIR = cfg.hr_dir      
    
    BATCH_SIZE = cfg.batch_size      
    LEARNING_RATE = cfg.learning_rate     
    NUM_EPOCHS = cfg.num_epochs         
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training Device: {DEVICE}")

    if not os.path.exists(LR_DIR) or not os.path.exists(HR_DIR):
        print("Error")
        return

    dataset = UpscaleDataset(lr_dir=LR_DIR,
                             hr_dir=HR_DIR,
                             patch_size=cfg.patch_size,
                             scale_factor=4)

    if len(dataset) == 0:
        print(" Error: Dataset is empty")
        return

    train_loader = DataLoader(dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              num_workers=cfg.num_workers,
                              pin_memory=True)
    
    prefix = _build_exp_prefix()
    
    model = SRCNN().to(DEVICE)
    
    #criterion = nn.MSELoss() #MSE
    criterion = nn.L1Loss() #L1 better
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5) #adjust learning rate
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    model.train() 
    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        
        for lr_imgs, hr_imgs in progress_bar:
            lr_imgs = lr_imgs.to(DEVICE)
            hr_imgs = hr_imgs.to(DEVICE)
            outputs = model(lr_imgs)
            
            # Calculate Loss
            loss = criterion(outputs, hr_imgs)
            
            #  Backward
            optimizer.zero_grad() 
            loss.backward()       
            optimizer.step()      
            
            # record Loss 
            epoch_loss += loss.item()
            progress_bar.set_postfix({'loss': f"{loss.item():.6f}"})
        avg_loss = epoch_loss / len(train_loader)
        scheduler.step()
        # Checkpoint
        if (epoch + 1) % cfg.save_every == 0:
            save_path = os.path.join(cfg.checkpoint_dir, f"{prefix}_epoch{epoch+1}.pth")
            torch.save(model.state_dict(), save_path)
    final_path = os.path.join(cfg.checkpoint_dir, f"{prefix}_final.pth")
    torch.save(model.state_dict(), final_path)
    print(f"🎉 Training Finished! Final model saved to: {final_path}")

if __name__ == "__main__":
    train()