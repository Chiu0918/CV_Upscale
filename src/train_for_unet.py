import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# 匯入 Dataset 和 U-Net
from src.data.dataset_pairs import UpscaleDataset
from src.models.unet_sr import UNetSR

def train():
    # --- 參數設定 (U-Net 比較吃顯存，Batch Size 可能要小一點) ---
    LR_DIR = 'data/train_lr'
    HR_DIR = 'data/train_hr'
    
    BATCH_SIZE = 16            # 如果 8 跑不動 (Out of Memory)，請改成 4
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 50          # U-Net 收斂比較慢，給它多一點時間
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"🚀 Training UNetSR on {DEVICE}...")

    # --- 準備資料 ---
    if not os.path.exists(LR_DIR) or not os.path.exists(HR_DIR):
        print("❌ 找不到資料夾")
        return

    # num_workers=2 加速讀取
    dataset = UpscaleDataset(lr_dir=LR_DIR, hr_dir=HR_DIR)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    
    if len(dataset) == 0:
        print("❌ Dataset 是空的")
        return

    # --- 建立模型 ---
    model = UNetSR().to(DEVICE)
    
    # 使用 L1 Loss (比 MSE 更能產生銳利邊緣)
    criterion = nn.L1Loss()
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 學習率排程：每 50 輪衰減一半
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    os.makedirs('models_ckpt', exist_ok=True)

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
        if (epoch + 1) % 20 == 0:
            torch.save(model.state_dict(), f'models_ckpt/unet_epoch_{epoch+1}.pth')

    # --- 最終存檔 ---
    final_path = 'models_ckpt/unet_final.pth'
    torch.save(model.state_dict(), final_path)
    print(f"🎉 UNet Training Finished! Saved to {final_path}")

if __name__ == "__main__":
    train()