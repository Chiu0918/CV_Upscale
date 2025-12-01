import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# 匯入 Dataset 和 U-Net
from src.data.dataset_pairs import UpscaleDataset
from src.models.unet_sr import UNetSR

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
    # --- 參數設定 (U-Net 比較吃顯存，Batch Size 可能要小一點) ---
    LR_DIR = cfg.lr_dir
    HR_DIR = cfg.hr_dir
    PATCH_SIZE = cfg.patch_size
    
    BATCH_SIZE = cfg.batch_size
    LEARNING_RATE = cfg.learning_rate
    NUM_EPOCHS = cfg.num_epochs          # U-Net 收斂比較慢，給它多一點時間
    
    cfg.model_name = "unet"
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"🚀 Training UNetSR on {DEVICE}...")

    # --- 準備資料 ---
    if not os.path.exists(LR_DIR) or not os.path.exists(HR_DIR):
        print("❌ 找不到資料夾")
        return

    # num_workers=2 加速讀取
    dataset = UpscaleDataset(lr_dir=LR_DIR,
                             hr_dir=HR_DIR,
                             patch_size=PATCH_SIZE,
                             scale_factor=4)
    
    if len(dataset) == 0:
        print("❌ Dataset 是空的")
        return
    
    train_loader = DataLoader(dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              num_workers=cfg.num_workers,
                              pin_memory=True)
            
    prefix = _build_exp_prefix()

    # --- 建立模型 ---
    model = UNetSR().to(DEVICE)
    
    # 使用 L1 Loss (比 MSE 更能產生銳利邊緣)
    criterion = nn.L1Loss()
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 學習率排程：每 50 輪衰減一半
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    # --- 訓練迴圈 ---
    model.train()
    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        
        for lr_imgs, hr_imgs in progress_bar:
            lr_imgs, hr_imgs = lr_imgs.to(DEVICE), hr_imgs.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(lr_imgs)
            loss = criterion(outputs, hr_imgs)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # 顯示當前 Loss 和 Learning Rate
            current_lr = optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({'loss': f"{loss.item():.6f}", 'lr': f"{current_lr:.6f}"})
            
        # 更新學習率
        scheduler.step()
        
        
        # 每 20 輪存檔一次
        if (epoch + 1) % cfg.save_every == 0:
            save_path = os.path.join(
                cfg.checkpoint_dir,
                f'{prefix}_epoch{epoch+1}.pth'
            )
            torch.save(model.state_dict(), save_path)

    # --- 最終存檔 ---
    final_path = os.path.join(
        cfg.checkpoint_dir,
        f'{prefix}_final.pth'
    )
    torch.save(model.state_dict(), final_path)
    print(f"🎉 UNet Training Finished! Saved to {final_path}")

if __name__ == "__main__":
    train()