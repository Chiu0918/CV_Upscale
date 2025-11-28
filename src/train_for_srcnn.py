import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset_pairs import UpscaleDataset
from src.models.srcnn import SRCNN

def train():
    """
    srcnn的training
    Loss:將MSE改成L1
    加上learning rate decay
    """
    LR_DIR = 'data/train_lr'      
    HR_DIR = 'data/train_hr'      
    
    BATCH_SIZE = 16      
    LEARNING_RATE = 1e-4     
    NUM_EPOCHS = 50         
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training Device: {DEVICE}")

    if not os.path.exists(LR_DIR) or not os.path.exists(HR_DIR):
        print("Error")
        return

    dataset = UpscaleDataset(lr_dir=LR_DIR, hr_dir=HR_DIR)

    if len(dataset) == 0:
        print(" Error: Dataset is empty")
        return

    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    

    model = SRCNN().to(DEVICE)
    
    #criterion = nn.MSELoss() #MSE
    criterion = nn.L1Loss() #L1 better
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5) #adjust learning rate
    os.makedirs('models_ckpt', exist_ok=True)

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
        if (epoch + 1) % 10 == 0:
            save_path = f'models_ckpt/srcnn_epoch_{epoch+1}.pth'
            torch.save(model.state_dict(), save_path)
    final_path = 'models_ckpt/srcnn_final.pth'
    torch.save(model.state_dict(), final_path)
    print(f"🎉 Training Finished! Final model saved to: {final_path}")

if __name__ == "__main__":
    train()