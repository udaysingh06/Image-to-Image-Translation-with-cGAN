import torchvision.utils as vutils
import os

def save_sample(fake_img, output_path, filename):
    os.makedirs(output_path, exist_ok=True)
    vutils.save_image(fake_img * 0.5 + 0.5, os.path.join(output_path, filename))
