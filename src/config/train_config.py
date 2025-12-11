class TrainConfig:
    lr_dir = "data/train_lr"
    hr_dir = "data/train_hr"
    patch_size = 192   #192 128 去測試
    batch_size = 16 
    learning_rate = 1e-4
    num_epochs = 250 
    model_name = "unet"   
    checkpoint_dir = "models_ckpt"
    num_workers = 0  
    save_every = 10  #50
    exp_name = "unet_v2_div2k" #做不同測試時記得要改名字