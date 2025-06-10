import torch

config = {
    "image_size": 128,
    "low_res_size": 64,
    "time_steps": 500,
    "batch_size": 16,
    "epochs": 2,
    "lr": 2e-4,
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    "dataset_path": "./dataset/hr_128_kaggle",  # folder of HR images
    "save_dir": "./outputs",
    "model_save_path": "./models/cats.pth",
    "log_interval": 10,
}
