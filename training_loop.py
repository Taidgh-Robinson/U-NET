import torch
import random
import pickle
import numpy as np
import torchvision.transforms.functional as TF
import torch.nn as nn
from logger_config import logger
from config import NUM_EPOCHS
from loss import combined_loss
from data_loader import OxfordPetDatasetLoader
from models import PetUNet
from helper_functions import (
    display_image_and_mask,
    save_loss_information,
    generate_random_crop_bounds,
    mirror_pad_to_size,
    create_required_directories,
    save_target_and_output,
    save_model_as_state_dict,
    crop_with_boundaries,
)

# Make CUDA operations deterministic
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


def trainPetUNetADAM(model_name, unet_model, train_dataset, is_color=False):
    image_path, state_dict_path, loss_path = create_required_directories(model_name)
    losses = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unet_model = unet_model.to(device)
    optimizer = torch.optim.Adam(unet_model.parameters(), lr=1e-4)

    for epoch in range(NUM_EPOCHS):
        rolling_loss = []
        unet_model.train()
        for image, mask in train_dataset:
            # Ensure image and mask are at least 572x572
            image = mirror_pad_to_size(image, 572).to(device)
            mask = mirror_pad_to_size(mask, 572).to(device)

            # Select random 572x572 section of image + corresponding part of mask
            crop_boundaries = generate_random_crop_bounds(image, 572)
            cropped_image = crop_with_boundaries(image, crop_boundaries)
            cropped_mask = crop_with_boundaries(mask, crop_boundaries)
            center_cropped_mask = TF.center_crop(
                cropped_mask, output_size=(388, 388)
            ).to(device)

            # Oxford pets come as 1: Animal, 2: Background, 3: Border
            # Rework it so 0: Not Animal, 1: Animal
            center_cropped_mask = (center_cropped_mask == 1).long()
            output = unet_model(cropped_image)

            # Calculate loss between model output + centered cropped section of mask
            optimizer.zero_grad()
            # Remove channel from center cropped mask:
            center_cropped_mask = center_cropped_mask.squeeze(1)

            loss = combined_loss(output, center_cropped_mask, device)
            logger.debug(f"Loss at current step is {loss.item()}")

            # Update model
            loss.backward()
            optimizer.step()

            # Store loss in total list + per epoch list to verify the model is learning
            losses.append(loss.item())
            rolling_loss.append(loss.item())

        logger.info(
            f"Average Loss for epoch {epoch}: {sum(rolling_loss) / len(rolling_loss)}"
        )

        save_target_and_output(
            cropped_image, center_cropped_mask, output, image_path, epoch, is_color
        )
        save_model_as_state_dict(unet_model, state_dict_path, epoch)

    save_loss_information(losses, loss_path, model_name)


""" Does not work YET, no meanginful data learned"""


def trainPetUNetSGD(model_name, unet_model):
    image_path, state_dict_path, loss_path = create_required_directories(model_name)
    losses = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset, test_dataset = OxfordPetDatasetLoader(2)
    unet_model = unet_model.to(device)
    optimizer = torch.optim.SGD(unet_model.parameters(), lr=0.01, momentum=0.90)

    for epoch in range(NUM_EPOCHS):
        rolling_loss = []
        unet_model.train()
        run = 0
        for image, mask in train_dataset:
            image = mirror_pad_to_size(image, 572).to(device)
            mask = mirror_pad_to_size(mask, 572).to(device)

            # Select random 572x572 section of image + corresponding part of mask
            crop_boundaries = generate_random_crop_bounds(image, 572)
            cropped_image = crop_with_boundaries(image, crop_boundaries)
            cropped_mask = crop_with_boundaries(mask, crop_boundaries)

            center_cropped_mask = TF.center_crop(
                cropped_mask, output_size=(388, 388)
            ).to(device)

            # Oxford pets come as 1: Animal, 2: Background, 3: Border
            # Rework it so 0: Not Animal, 1: Animal
            center_cropped_mask = (center_cropped_mask == 1).long()
            output = unet_model(cropped_image)

            # Calculate loss between model output + centered cropped section of mask
            optimizer.zero_grad()
            # Remove channel from center cropped mask:
            center_cropped_mask = center_cropped_mask.squeeze(1)

            loss = combined_loss(output, center_cropped_mask, device)
            logger.debug(f"Loss at current step is {loss.item()}")
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            rolling_loss.append(loss.item())

        logger.info(
            f"Average Loss for epoch {epoch}: {sum(rolling_loss) / len(rolling_loss)}"
        )

        save_target_and_output(
            cropped_image, center_cropped_mask, output, image_path, epoch, is_color
        )
        save_model_as_state_dict(unet_model, state_dict_path, epoch)

    save_loss_information(losses, model_name, loss_path)

    return True


"""
Sanity check training loop, train our model on a single image and crop and make sure it can learn 
"""


def trainPetUNetSingleItem(model_name):
    image_path, state_dict_path, loss_path = create_required_directories(model_name)
    losses = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    UNet = PetUNet().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(UNet.parameters(), lr=0.01, momentum=0.90)
    train_dataset, test_dataset = OxfordPetDatasetLoader(2)

    image, mask = next(iter(train_dataset))
    image = mirror_pad_to_size(image, 572)
    mask = mirror_pad_to_size(mask, 572)

    image = image.to(device)
    mask = mask.to(device)
    display_image_and_mask(image, mask)

    crop_boundaries = generate_random_crop_bounds(image, 572)
    cropped_image = crop_with_boundaries(image, crop_boundaries)
    cropped_mask = crop_with_boundaries(mask, crop_boundaries)

    center_cropped_mask = TF.center_crop(cropped_mask, output_size=(388, 388))

    # Oxford pets come as 1: Animal, 2: Background, 3: Border
    # Rework it so 0: Not Animal, 1: Animal
    center_cropped_mask = (center_cropped_mask == 1).long()

    # Ensure mask is on the same device as model
    center_cropped_mask = center_cropped_mask.to(device)
    center_cropped_mask = center_cropped_mask.squeeze(1)  # remove channel
    for epoch in range(1000):
        UNet.train()

        output = UNet(cropped_image)
        # Calculate loss between model output + centered cropped section of mask
        optimizer.zero_grad()

        loss = loss_fn(output, center_cropped_mask)
        loss.backward()
        optimizer.step()

        print(loss.item())
        losses.append(loss.item())

        save_model_as_state_dict(UNet, state_dict_path, epoch)
        save_target_and_output(
            cropped_image, center_cropped_mask, output, image_path, epoch
        )

    save_loss_information(losses, model_name, loss_path)

    return True
