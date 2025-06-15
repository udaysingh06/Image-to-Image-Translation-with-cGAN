import os
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

os.makedirs("output", exist_ok=True)

# ===============================
# 1. Dataset
# ===============================

class FacadesDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = [f for f in os.listdir(root_dir) if f.endswith(".jpg")]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.images[idx])
        image = Image.open(img_path).convert("RGB")
        w, h = image.size
        input_image = image.crop((0, 0, w // 2, h))
        target_image = image.crop((w // 2, 0, w, h))

        if self.transform:
            input_image = self.transform(input_image)
            target_image = self.transform(target_image)

        return input_image, target_image

# ===============================
# 2. Models (Simplified UNet Generator & PatchGAN Discriminator)
# ===============================

class UNetGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1), nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(6, 64, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 1, 4, 1, 1), nn.Sigmoid()
        )

    def forward(self, x, y):
        return self.main(torch.cat([x, y], 1))

# ===============================
# 3. Training Setup
# ===============================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 10
LR = 2e-4
BATCH_SIZE = 1

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

dataset = FacadesDataset("data/facades/train", transform=transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

G = UNetGenerator().to(DEVICE)
D = Discriminator().to(DEVICE)
loss_fn = nn.BCELoss()
l1_loss = nn.L1Loss()

g_opt = optim.Adam(G.parameters(), lr=LR, betas=(0.5, 0.999))
d_opt = optim.Adam(D.parameters(), lr=LR, betas=(0.5, 0.999))

# ===============================
# 4. Training Loop
# ===============================

for epoch in range(EPOCHS):
    for idx, (x, y) in enumerate(loader):
        x, y = x.to(DEVICE), y.to(DEVICE)

        # Train D
        fake = G(x)
        real_label = torch.ones_like(D(x, y)).to(DEVICE)
        fake_label = torch.zeros_like(D(x, fake.detach())).to(DEVICE)

        d_real = D(x, y)
        d_fake = D(x, fake.detach())

        d_loss = loss_fn(d_real, real_label) + loss_fn(d_fake, fake_label)
        d_opt.zero_grad()
        d_loss.backward()
        d_opt.step()

        # Train G
        d_fake = D(x, fake)
        g_adv = loss_fn(d_fake, real_label)
        g_l1 = l1_loss(fake, y) * 100
        g_loss = g_adv + g_l1

        g_opt.zero_grad()
        g_loss.backward()
        g_opt.step()

        if idx % 100 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Step {idx} | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")
            save_image(fake * 0.5 + 0.5, f"output/sample_{epoch+1}_{idx}.png")

# Save model
torch.save(G.state_dict(), "generator.pth")
torch.save(D.state_dict(), "discriminator.pth")
