import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast 
from tqdm import tqdm

from src.models.edsr import EDSR
from src.data.dataset_pairs import UpscaleDataset
from src.config.train_config import TrainConfig as cfg
from src.utils import append_log_row

def train():
    BATCH_SIZE = 32  
    PATCH_SIZE = 192 #測192、32
    NUM_EPOCHS = 250  
    LR = 1e-4
    
    exp_name = "edsr_P192_div2k"  #記得改名字
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training EDSR on {DEVICE}...")
    print(f"Config: Patch={PATCH_SIZE}, Batch={BATCH_SIZE}, Epochs={NUM_EPOCHS}")

    dataset = UpscaleDataset(
        lr_dir=cfg.lr_dir, 
        hr_dir=cfg.hr_dir, 
        patch_size=PATCH_SIZE, 
        scale_factor=4
    )
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=0, 
        pin_memory=True
    )
    val_dataset = UpscaleDataset(
        lr_dir="data/val_lr", 
        hr_dir="data/val_hr", 
        patch_size=PATCH_SIZE, 
        scale_factor=4
    )
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    model = EDSR(n_resblocks=16, n_feats=64, scale=4).to(DEVICE)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    scaler = GradScaler()

    log_dir = os.path.join("logs", exp_name)
    ckpt_dir = "models_ckpt"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    
    log_path = os.path.join(log_dir, "train_log.csv")
    log_fieldnames = ["epoch", "train_loss", "val_loss", "learning_rate"] 

    best_val_loss = float('inf')

    model.train() 
    
    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")
        for lr, hr in pbar:
            lr, hr = lr.to(DEVICE), hr.to(DEVICE)           
            optimizer.zero_grad()
            with autocast():
                preds = model(lr)
                loss = criterion(preds, hr)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.6f}"})
        
        avg_train_loss = epoch_loss / len(dataloader) 
        current_lr = optimizer.param_groups[0]['lr']

        #  Validation Loop 
        model.eval() 
        val_loss_sum = 0.0
        with torch.no_grad():
            for val_lr, val_hr in val_dataloader:
                val_lr, val_hr = val_lr.to(DEVICE), val_hr.to(DEVICE)
                val_preds = model(val_lr)
                v_loss = criterion(val_preds, val_hr)
                val_loss_sum += v_loss.item()
        
        avg_val_loss = val_loss_sum / len(val_dataloader)
        model.train() 
        print(f"Epoch {epoch+1} Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | LR: {current_lr:.2e}")
        
        append_log_row(log_path, {
            "epoch": epoch+1, 
            "train_loss": avg_train_loss, 
            "val_loss": avg_val_loss,    
            "learning_rate": current_lr
        }, log_fieldnames)
        
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"{ckpt_dir}/{exp_name}_epoch{epoch+1}.pth")
        if avg_val_loss < best_val_loss: 
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f"{ckpt_dir}/{exp_name}_best.pth")
            print(f"🏆 Best Model Saved! ({best_val_loss:.6f})")

    torch.save(model.state_dict(), f"{ckpt_dir}/{exp_name}_final.pth")
    print(f"EDSR Training Finished! Saved to {ckpt_dir}/{exp_name}_final.pth")

if __name__ == "__main__":
    train()