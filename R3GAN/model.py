import pickle
import torch
from PIL import Image
import numpy as np


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
label_to_index = {
    "Normal": 0,        # cdr_0
    "Mild": 1,          # cdr_0_5
    "Moderate": 2,      # cdr_1
    "Severe": 3,        # cdr_2
    "Very Severe": 4    # cdr_3
}

index_to_label = {v: k for k, v in label_to_index.items()}


# Load the R3GAN model from a .pkl file
def open_model(path: str):
    with open(path, 'rb') as f:
        data = pickle.load(f)

    # Assume standard format: {'G': generator, 'D': discriminator}
    generator = data['G'].to(device).eval()
    discriminator = data['D'].to(device).eval()
    return generator, discriminator


def generate_image(generator, severity: str, z_dim=512):
    print("Conditional dimension:", generator.c_dim)

    label_idx = label_to_index.get(severity, 0)
    z = torch.randn(1, z_dim, device=device)

    # Generate one-hot label
    label = torch.zeros([1, generator.c_dim], device=device)
    label[:, label_idx] = 1

    with torch.no_grad():
        img = generator(z, label)[0].cpu().numpy()

    img = np.clip((img + 1) * 127.5, 0, 255).astype(np.uint8)
    return Image.fromarray(np.transpose(img, (1, 2, 0)))


def classify_image(discriminator, image: Image.Image) -> str:
    # Convert to grayscale if it's RGB
    if image.mode != 'L':
        image = image.convert('L')  # 'L' = grayscale

    # Convert PIL to tensor and normalize
    img_tensor = torch.tensor(np.array(image), dtype=torch.float32, device=device)

    # Ensure shape is [1, 1, H, W]
    if img_tensor.ndim == 2:  # [H, W]
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)
    elif img_tensor.ndim == 3 and img_tensor.shape[2] == 1:  # [H, W, 1]
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
    else:
        raise ValueError(f"Unexpected image shape: {img_tensor.shape}")

    img_tensor = img_tensor / 127.5 - 1  # Normalize to [-1, 1]

    num_classes = 5
    logits_list = []

    with torch.no_grad():
        for class_idx in range(num_classes):
            c = torch.zeros([1, num_classes], device=device)
            c[0, class_idx] = 1  # one-hot
            logit = discriminator(img_tensor, c)
            logits_list.append(logit)

    # Stack logits and find the index with the highest score
    logits = torch.cat(logits_list, dim=0)  # [num_classes, 1]
    pred = torch.argmax(logits).item()

    return index_to_label.get(pred, "Unknown")
