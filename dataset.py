import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class SRDataset(Dataset):
    def __init__(self, folder, hr_size=64, lr_size=16):
        self.folder = folder
        self.hr_size = hr_size
        self.lr_size = lr_size
        self.files = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".png"))]

        self.transform = transforms.Compose([
            transforms.CenterCrop(hr_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.folder, self.files[idx])).convert("RGB")
        hr = self.transform(img)
        lr = TF.resize(hr, [self.lr_size, self.lr_size], interpolation=Image.BICUBIC)
        lr = TF.resize(lr, [self.hr_size, self.hr_size], interpolation=Image.BICUBIC)
        return lr, hr  # x (LR), y (HR)
