# train.py

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from dataset import SRDataset
from model import UNet
from diffusion import Diffusion
from config import config


def train():
    device = config["device"]
    os.makedirs(config["save_dir"], exist_ok=True)

    # Load dataset
    dataset = SRDataset(config["dataset_path"], hr_size=config["image_size"], lr_size=config["low_res_size"])
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
    
    first_lr, _ = dataset[0]
    first_lr = first_lr.unsqueeze(0).to(device)  # Add batch dimension
    first_lr_up = torch.nn.functional.interpolate(
    first_lr, size=(config["image_size"], config["image_size"]),
        mode="bilinear", align_corners=False
    )

    # Model + diffusion
    model = UNet(in_channels=6, out_channels=3).to(device)
    diffusion = Diffusion(model, image_size=config["image_size"], device=device, timesteps=config["time_steps"])

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    mse_loss = nn.MSELoss()

    for epoch in range(config["epochs"]):
        for i, (lr, hr) in enumerate(dataloader):
            lr, hr = lr.to(device), hr.to(device)

            # ✅ Upsample LR to match HR size
            lr_up = torch.nn.functional.interpolate(
                lr,
                size=(config["image_size"], config["image_size"]),
                mode="bilinear",
                align_corners=False
            )

            # Sample timestep t for each image in the batch
            t = torch.randint(0, config["time_steps"], (hr.size(0),), device=device).long()

            # Forward process: get y_t and target noise
            y_t, noise = diffusion.add_noise(hr, t)

            # Condition on LR
            model_input = torch.cat([lr_up, y_t], dim=1)  # both are now 256x256

            # Predict noise
            pred_noise = model(model_input, t)

            # ✅ Fix: Resize pred_noise to match noise size
            if pred_noise.shape[-1] != noise.shape[-1] or pred_noise.shape[-2] != noise.shape[-2]:
                pred_noise = torch.nn.functional.interpolate(
                    pred_noise,
                    size=noise.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )

            # Loss
            loss = mse_loss(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % config["log_interval"] == 0:
                print(f"Epoch [{epoch}], Step [{i}/{len(dataloader)}], Loss: {loss.item():.4f}")

        # Save intermediate sample
        diffusion.sample(first_lr_up, save_path=f"{config['save_dir']}/sample_epoch_{epoch}.png")
        
    # Save final model
    torch.save(model.state_dict(), config["model_save_path"])
    print(f"[✓] Training complete. Model saved to {config['model_save_path']}")


if __name__ == "__main__":
    train()
