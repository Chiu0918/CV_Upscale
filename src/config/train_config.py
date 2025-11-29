class TrainConfig:
    # Data
    lr_dir = "data/train_lr"
    hr_dir = "data/train_hr"
    patch_size = None     # None=full image, e.g. 32=patch training

    # Training
    batch_size = 16
    learning_rate = 1e-4
    num_epochs = 50

    model_name = "unet"   # "srcnn", "unet"
    checkpoint_dir = "models_ckpt"

    # Others
    num_workers = 2
    save_every = 10
    
    # Optional experiment name (if None -> auto from config)
    exp_name = None