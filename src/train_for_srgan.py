import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import vgg19, VGG19_Weights
from tqdm import tqdm

from src.models.srgan import Generator, Discriminator
from src.data.dataset_pairs import UpscaleDataset
from src.config.train_config import TrainConfig as cfg
from src.utils import append_log_row

class VGGLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        vgg = vgg19(weights=VGG19_Weights.DEFAULT).features[:36].eval().to(device)
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        self.criterion = nn.MSELoss()
        self.register_buffer("mean", torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device))
        self.register_buffer("std", torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device))

    def forward(self, sr, hr):
        sr = (sr - self.mean) / self.std
        hr = (hr - self.mean) / self.std
        sr_features = self.vgg(sr)
        hr_features = self.vgg(hr)
        return self.criterion(sr_features, hr_features)

def train():
    LR_DIR = cfg.lr_dir
    HR_DIR = cfg.hr_dir
    PATCH_SIZE = 128  #128、96去測
    BATCH_SIZE = 8    
    NUM_EPOCHS = 250   
    exp_name = "srgan_div2k" #記得改                             
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training SRGAN on {DEVICE}...")
    dataset = UpscaleDataset(
        lr_dir=LR_DIR, hr_dir=HR_DIR, patch_size=PATCH_SIZE, scale_factor=4
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_dataset = UpscaleDataset(
        lr_dir="data/val_lr", 
        hr_dir="data/val_hr", 
        patch_size=PATCH_SIZE,
        scale_factor=4
    )
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    netG = Generator().to(DEVICE)
    netD = Discriminator().to(DEVICE)
    
    def weights_init(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
                
    netG.apply(weights_init)
    netD.apply(weights_init)
    
    criterion_GAN = nn.BCELoss()    
    criterion_content = nn.L1Loss()  
    criterion_VGG = VGGLoss(DEVICE)  
    optimizer_G = optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.999))
    
    log_dir = os.path.join("logs", exp_name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs("models_ckpt", exist_ok=True)
    log_path = os.path.join(log_dir, "train_log.csv")
    log_fieldnames = ["epoch", "avg_g_loss", "avg_d_loss", "val_loss", "learning_rate"]
    best_val_loss = float('inf') 

    print("Starting GAN Training loop...")
    
    for epoch in range(NUM_EPOCHS):
        netG.train() 
        netD.train()
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")
        g_losses = []
        d_losses = []

        for lr_imgs, hr_imgs in pbar:
            lr_imgs = lr_imgs.to(DEVICE)
            hr_imgs = hr_imgs.to(DEVICE)
            batch_size = lr_imgs.size(0)
            real_label = torch.ones(batch_size, 1).to(DEVICE)
            fake_label = torch.zeros(batch_size, 1).to(DEVICE)
            optimizer_D.zero_grad()
            output_real = netD(hr_imgs)
            loss_D_real = criterion_GAN(output_real, real_label)
            fake_imgs = netG(lr_imgs)
            output_fake = netD(fake_imgs.detach()) 
            loss_D_fake = criterion_GAN(output_fake, fake_label)

            loss_D = (loss_D_real + loss_D_fake) / 2
            loss_D.backward()
            optimizer_D.step()
            optimizer_G.zero_grad()
            output_fake_for_G = netD(fake_imgs)
            loss_G_GAN = criterion_GAN(output_fake_for_G, real_label)
            loss_G_VGG = criterion_VGG(fake_imgs, hr_imgs)
            loss_G_Content = criterion_content(fake_imgs, hr_imgs)
            loss_G = loss_G_VGG + (1e-3 * loss_G_GAN) + (10.0 * loss_G_Content)  

            loss_G.backward()
            optimizer_G.step()

            g_losses.append(loss_G.item())
            d_losses.append(loss_D.item())
            
            pbar.set_postfix({'G_Loss': f'{loss_G.item():.4f}', 'D_Loss': f'{loss_D.item():.4f}'})
        
        avg_g = sum(g_losses) / len(g_losses)
        avg_d = sum(d_losses) / len(d_losses)
        netG.eval() 
        val_loss_sum = 0.0
        with torch.no_grad():
            for val_lr, val_hr in val_dataloader:
                val_lr, val_hr = val_lr.to(DEVICE), val_hr.to(DEVICE)
                val_sr = netG(val_lr)
                val_loss_sum += criterion_content(val_sr, val_hr).item()
        
        avg_val_loss = val_loss_sum / len(val_dataloader)
        netG.train() 
        current_lr = optimizer_G.param_groups[0]['lr']
        print(f"Epoch {epoch+1} | G Loss: {avg_g:.4f} | D Loss: {avg_d:.4f} | Val L1: {avg_val_loss:.6f}")
        
        append_log_row(log_path, {
            "epoch": epoch+1, 
            "avg_g_loss": avg_g, 
            "avg_d_loss": avg_d, 
            "val_loss": avg_val_loss,
            "learning_rate": current_lr
        }, log_fieldnames)
        if (epoch + 1) % 5 == 0:
            save_path = f"models_ckpt/{exp_name}_epoch{epoch+1}_G.pth"
            torch.save(netG.state_dict(), save_path)
            print(f"💾 Saved Generator to {save_path}")
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(netG.state_dict(), f"models_ckpt/{exp_name}_best_G.pth")
            print(f"🏆 Best Generator Saved! (Val L1: {best_val_loss:.6f})")

    final_save_path = f"models_ckpt/{exp_name}_final_G.pth"
    torch.save(netG.state_dict(), final_save_path)
    print(f"SRGAN Training Finished! Saved to {final_save_path}")

if __name__ == "__main__":
    train()