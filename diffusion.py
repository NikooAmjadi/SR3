import torch
import torch.nn.functional as F
import math


class Diffusion:
    def __init__(self, model, image_size, device, timesteps=1000, beta_start=1e-4, beta_end=0.01):
        self.model = model
        self.image_size = image_size
        self.device = device
        self.timesteps = timesteps

        self.betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
        self.alphas = 1. - self.betas
        self.alpha_hats = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, y_0, t):
        """
        Forward process q(y_t | y_0): Adds noise to y_0
        """
        noise = torch.randn_like(y_0)
        alpha_hat = self.alpha_hats[t].view(-1, 1, 1, 1).to(self.device)
        y_t = torch.sqrt(alpha_hat) * y_0 + torch.sqrt(1 - alpha_hat) * noise
        return y_t, noise

    def sample(self, lr_img, save_path="sr_sample.png"):
        """
        Reverse denoising process: from noise to clean image conditioned on LR input
        """
        from torchvision.utils import save_image
        self.model.eval()
        lr_img = lr_img.to(self.device)
        
        # Upscale LR to match HR resolution
        y = F.interpolate(lr_img, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
        
        with torch.no_grad():
            for t in reversed(range(1, self.timesteps)):
                t_tensor = torch.full((lr_img.size(0),), t, device=self.device, dtype=torch.long)

                alpha = self.alphas[t]
                alpha_hat = self.alpha_hats[t]
                beta = self.betas[t]

                cond_input = torch.cat([lr_img, y], dim=1)
                pred_noise = self.model(cond_input, t_tensor)

                # Ensure pred_noise matches y's resolution
                if pred_noise.shape[-2:] != y.shape[-2:]:
                    pred_noise = F.interpolate(pred_noise, size=y.shape[-2:], mode='bilinear', align_corners=False)

                y = (1 / torch.sqrt(alpha)) * (
                    y - ((1 - alpha) / torch.sqrt(1 - alpha_hat)) * pred_noise
                )

                if t > 1:
                    y += torch.sqrt(beta) * torch.randn_like(y)

        #save_image(y, save_path)
        save_image((y + 1) / 2, save_path)      
        print(f"[✓] Sample saved to {save_path}")

