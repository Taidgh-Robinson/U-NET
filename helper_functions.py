import matplotlib.pyplot as plt
import torch
import math
import os
import torch.nn as nn
import pickle
from random import randrange
from logger_config import logger
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import numpy as np

def plot_average_loss(loss_array, title="Training Loss", xlabel="Iteration", ylabel="Loss", window=500):
    loss_array = np.array(loss_array)

    if len(loss_array) >= window:
        # Moving average over last `window` elements
        kernel = np.ones(window) / window
        smoothed = np.convolve(loss_array, kernel, mode="valid")
        x_vals = np.arange(window - 1, len(loss_array))
        plt.plot(x_vals, smoothed)
    else:
        # Not enough values yet → just plot raw loss
        plt.plot(loss_array)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def display_image_and_mask(image, mask, filename=None):
    # Make sure tensors are CPU and NumPy
    # Select first sample from batch

    if len(image.size()) == 4:
        image = image.squeeze(0)

    if len(mask.size()) == 4:
        mask = mask.squeeze(0)

    if len(image.size()) == 3:
        image = image.squeeze(0)

    if len(mask.size()) == 3:
        mask = mask.squeeze(0)

    if isinstance(image, torch.Tensor):
        image_np = image.detach().cpu().numpy()
    else:
        image_np = image

    if isinstance(mask, torch.Tensor):
        mask_np = mask.detach().cpu().numpy()
    else:
        mask_np = mask

    # Rescale image if normalized [-1,1]
    image_np = (image_np + 1.0) / 2.0
    image_np = image_np.clip(0, 1)

    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.title("Image")
    plt.imshow(image_np, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Mask")
    plt.imshow(mask_np, cmap="tab20")  # good for multi-class masks
    plt.axis("off")

    if filename:
        plt.savefig(filename)
        plt.close()
        logger.debug(f"Saved visualization to {filename}")
    else:
        plt.show()

def display_image_and_mask_color(image, mask, filename=None):
    # Make sure tensors are CPU and NumPy
    # Select first sample from batch

    if len(image.size()) == 4:
        image = image.squeeze(0)

    if len(mask.size()) == 4:
        mask = mask.squeeze(0)

    if len(image.size()) == 3:
        image = image.squeeze(0)

    if len(mask.size()) == 3:
        mask = mask.squeeze(0)

    if isinstance(image, torch.Tensor):
        image_np = image.detach().cpu().numpy()
    else:
        image_np = image

    if isinstance(mask, torch.Tensor):
        mask_np = mask.detach().cpu().numpy()
    else:
        mask_np = mask

    # Rescale image if normalized [-1,1]
    image_np = (image_np + 1.0) / 2.0
    image_np = image_np.clip(0, 1)
    image_np = np.transpose(image_np, (1, 2, 0))

    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.title("Image")
    plt.imshow(image_np)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Mask")
    plt.imshow(mask_np, cmap="tab20")  # good for multi-class masks
    plt.axis("off")

    if filename:
        plt.savefig(filename)
        plt.close()
        logger.debug(f"Saved visualization to {filename}")
    else:
        plt.show()


def convert_model_output_to_values(model_output):
    return torch.argmax(model_output, dim=1)[0]


def generate_random_crop_bounds(image, crop_size):
    h = image.size()[2]
    w = image.size()[3]
    crop_boundry_h = h - crop_size + 1
    crop_boundry_w = w - crop_size + 1
    h_boundry = randrange(0, crop_boundry_h)
    w_boundry = randrange(0, crop_boundry_w)
    return ((h_boundry, h_boundry + crop_size), (w_boundry, w_boundry + crop_size))


def mirror_pad_to_size(image, size):
    h = image.size()[2]
    w = image.size()[3]
    if h >= size and w >= size:
        return image

    left, right, top, bottom = 0, 0, 0, 0
    if h < size:
        diff = size - h
        diff = (diff + 1) // 2
        top, bottom = diff, diff

    if w < size:
        diff = size - w
        diff = (diff + 1) // 2
        left, right = diff, diff

    pad = nn.ReflectionPad2d((left, right, top, bottom))
    return pad(image)


def compute_class_ratio(train_loader, device):
    total_pixels = 0
    animal_pixels = 0

    for image, mask in train_loader:
        mask = mask.to(device)
        mask = (mask == 1).long()
        total_pixels += mask.numel()
        animal_pixels += mask.sum().item()

    ratio = animal_pixels / total_pixels
    return ratio


def test_model_against_image_and_mask(model, image, mask):
    image = mirror_pad_to_size(image, 572)
    mask = mirror_pad_to_size(mask, 572)
    display_image_and_mask(image, mask, "images/raw.png")

    crop_boundies = generate_random_crop_bounds(image, 572)
    cropped_image = image[
        :,
        :,
        crop_boundies[0][0] : crop_boundies[0][1],
        crop_boundies[1][0] : crop_boundies[1][1],
    ]
    cropped_mask = mask[
        :,
        :,
        crop_boundies[0][0] : crop_boundies[0][1],
        crop_boundies[1][0] : crop_boundies[1][1],
    ]

    center_cropped_mask = TF.center_crop(cropped_mask, output_size=(388, 388))
    center_cropped_mask = center_cropped_mask == 1

    display_image_and_mask(cropped_image, center_cropped_mask, "images/target.png")

    output = model(cropped_image)
    display_image_and_mask(
        cropped_image, convert_model_output_to_values(output), "images/output.png"
    )

def create_required_directories(model_name):
    image_path = os.path.join('images', model_name)
    state_dict_path = os.path.join('model_state_dict', model_name)
    loss_path = 'losses'
    os.makedirs(os.path.join(image_path, 'output'), exist_ok=True)
    os.makedirs(os.path.join(image_path, 'target'), exist_ok=True)
    os.makedirs(state_dict_path, exist_ok=True)
    os.makedirs(loss_path, exist_ok=True)
    return image_path, state_dict_path, loss_path

def save_target_and_output(image, target, output, image_path, idx):

    display_image_and_mask(
        image,
        target,
        f"{os.path.join(image_path, 'target')}/{idx}.jpg"
    )

    display_image_and_mask(
        image,
        convert_model_output_to_values(output),
        f"{os.path.join(image_path, 'output')}/{idx}.jpg"
    )

def save_model_as_state_dict(model, state_dict_path, idx):
    torch.save(
        model.state_dict(),
        f"{state_dict_path}/policy_net-{idx}.pth",
    )

def save_loss_information(loss, loss_path, model_name):
    with open(f'{loss_path}/{model_name}.pkl', "wb") as f:
        pickle.dump(loss, f)

def crop_with_boundaries(tensor, crop_boundaries):
    return tensor[
        :,
        :,
        crop_boundaries[0][0] : crop_boundaries[0][1],
        crop_boundaries[1][0] : crop_boundaries[1][1],
    ]

def apply_model_to_whole_image(model, image):
    """
    Vibe coded - SUCKS - Need to rework

    Applies a UNet model (572x572 input -> 388x388 output) to an arbitrarily sized image
    by tiling with mirror padding.
    
    Args:
        model: trained UNet model
        image: tensor of shape (C, H, W) or (1, C, H, W)
    
    Returns:
        mask: tensor of shape (num_classes, H, W) or (1, num_classes, H, W) matching input shape
    """
    
    INPUT_SIZE = 572
    OUTPUT_SIZE = 388
    BORDER = (INPUT_SIZE - OUTPUT_SIZE) // 2  # 92px border needed on each side
    
    # Ensure we have a batch dimension
    if image.dim() == 3:
        image = image.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False
    
    _, C, H, W = image.shape
    
    # --- Step 1: Pad so height and width are multiples of OUTPUT_SIZE ---
    # We need enough padding so that tiles cover the whole image.
    # Each tile needs a 92px mirror border around it, so we pad the image
    # to the next multiple of OUTPUT_SIZE, then add 92 on each side.
    
    pad_h = math.ceil(H / OUTPUT_SIZE) * OUTPUT_SIZE - H
    pad_w = math.ceil(W / OUTPUT_SIZE) * OUTPUT_SIZE - W
    
    # Pad the image to a multiple of OUTPUT_SIZE (pad on bottom/right)
    # then add the 92px mirror border on all sides
    # F.pad order: (left, right, top, bottom)
    image_padded = F.pad(image, (0, pad_w, 0, pad_h), mode='reflect')
    image_padded = F.pad(image_padded, (BORDER, BORDER, BORDER, BORDER), mode='reflect')
    
    # Tiled canvas dimensions (before border padding)
    tiled_H = H + pad_h
    tiled_W = W + pad_w
    
    num_tiles_h = tiled_H // OUTPUT_SIZE
    num_tiles_w = tiled_W // OUTPUT_SIZE
    
    # --- Step 2: Run model on each tile ---
    model.eval()
    device = next(model.parameters()).device
    
    # We don't know num_classes until we run the model, so collect outputs
    tile_outputs = {}
    
    with torch.no_grad():
        for i in range(num_tiles_h):
            for j in range(num_tiles_w):
                # The crop into image_padded starts at (i*388, j*388) —
                # the border padding means we naturally get the 92px context around each tile
                y_start = i * OUTPUT_SIZE
                x_start = j * OUTPUT_SIZE
                
                crop = image_padded[:, :, y_start:y_start + INPUT_SIZE, x_start:x_start + INPUT_SIZE]
                crop = crop.to(device)
                
                output = model(crop)  # (1, num_classes, 388, 388)
                tile_outputs[(i, j)] = output.cpu()
    
    # --- Step 3: Stitch tiles together ---
    sample_output = next(iter(tile_outputs.values()))
    num_classes = sample_output.shape[1]
    
    full_mask = torch.zeros(1, num_classes, tiled_H, tiled_W)
    
    for (i, j), tile in tile_outputs.items():
        y_start = i * OUTPUT_SIZE
        x_start = j * OUTPUT_SIZE
        full_mask[:, :, y_start:y_start + OUTPUT_SIZE, x_start:x_start + OUTPUT_SIZE] = tile
    
    # --- Step 4: Crop back to original image size ---
    full_mask = full_mask[:, :, :H, :W]
    
    if squeeze_output:
        full_mask = full_mask.squeeze(0)
    
    return full_mask