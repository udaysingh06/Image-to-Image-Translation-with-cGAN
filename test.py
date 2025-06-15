import torch
from torchvision import transforms
from PIL import Image
from models.generator import UNetGenerator
from utils import save_sample
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

gen = UNetGenerator().to(DEVICE)
gen.load_state_dict(torch.load("generator.pth", map_location=DEVICE))
gen.eval()

for filename in os.listdir("data/facades/test"):
    img = Image.open(os.path.join("data/facades/test", filename)).convert("RGB")
    input_img = transform(img.crop((0, 0, img.width // 2, img.height))).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        fake_img = gen(input_img)
        save_sample(fake_img, "output/test", filename)
