# sample.py

import torch
from PIL import Image
from torchvision import transforms
from model import UNet
from diffusion import Diffusion
from config import config

def load_lr_image(path, target_size):
    transform = transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    img = Image.open(path).convert("RGB")
    tensor = transform(img).unsqueeze(0)  # shape: (1, 3, H, W)
    return tensor

def run_sampling():
    device = config["device"]
    lr_img = load_lr_image("lr.png", config["low_res_size"]).to(device)
    lr_img = torch.nn.functional.interpolate(lr_img, size=config["image_size"], mode="bicubic")

    # Load model
    model = UNet(in_channels=6, out_channels=3).to(device)
    model.load_state_dict(torch.load(config["model_save_path"], map_location=device))

    # Run diffusion sampling
    diffusion = Diffusion(model, image_size=config["image_size"], device=device, timesteps=config["time_steps"])
    diffusion.sample(lr_img, save_path="sr_output.png")

if __name__ == "__main__":
    run_sampling()
