# Elasitic deform
# Traditional deform
# Feed image section into model
# Calculate loss between model output + centered cropped section of mask
from data_loader import OxfordPetDatasetLoader
from models import PetUNet
from helper_functions import (
    display_image_and_mask,
    convert_model_output_to_values,
    generate_random_crop_bounds,
    mirror_pad_to_size,
)
import torchvision.transforms.functional as TF
import torch
import numpy as np
import random
import torch.nn as nn
import pickle
from logger_config import logger
from config import NUM_EPOCHS

# Make CUDA operations deterministic
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


def elastic_deformation(image, mask):
    pass


def traditional_deformation(image, mask):
    pass


def dice_loss(pred, target, smooth=1):
    # pred is raw logits [B, 2, H, W], target is [B, H, W] long
    pred = torch.softmax(pred, dim=1)  # convert to probabilities
    pred_fg = pred[:, 1, :, :]  # grab foreground channel probability

    target_float = target.float()

    intersection = (pred_fg * target_float).sum()
    return 1 - (2.0 * intersection + smooth) / (
        pred_fg.sum() + target_float.sum() + smooth
    )


# Combined loss
def combined_loss(pred, target, device, dice_weight=0.5):
    ce = nn.CrossEntropyLoss().to(device)(pred, target)
    dice = dice_loss(pred, target)
    return ce * (1 - dice_weight) + dice * dice_weight


def trainPetUNet():
    losses = []

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    train_dataset, test_dataset = OxfordPetDatasetLoader(2)

    UNet = PetUNet().to(device)

    # loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(UNet.parameters(), lr=1e-4)

    for epoch in range(NUM_EPOCHS):
        rolling_loss = []
        UNet.train()
        run = 0
        for image, mask in train_dataset:
            image = mirror_pad_to_size(image, 572).to(device)
            mask = mirror_pad_to_size(mask, 572).to(device)
            # Select random 572x572 section of image + corresponding part of mask
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

            center_cropped_mask = TF.center_crop(
                cropped_mask, output_size=(388, 388)
            ).to(device)

            # Oxford pets come as 1: Animal, 2: Background, 3: Border
            # Rework it so 0: Not Animal, 1: Animal
            center_cropped_mask = (center_cropped_mask == 1).long()
            pet_frac = center_cropped_mask.float().mean().item()
            output = UNet(cropped_image)

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

        display_image_and_mask(
            cropped_image,
            center_cropped_mask,
            f"images/full_loop_adam/regular-epoch-{epoch}.jpg",
        )
        display_image_and_mask(
            cropped_image,
            convert_model_output_to_values(output),
            f"images/full_loop_adam/model-epoch-{epoch}.jpg",
        )

        logger.info(
            f"Average Loss for epoch {epoch}: {sum(rolling_loss) / len(rolling_loss)}"
        )
        torch.save(
            UNet.state_dict(),
            f"model_state_dicts/full_model_adam/{epoch}-policy_net.pth",
        )

    with open("losses/losses-adam.pkl", "wb") as f:
        pickle.dump(losses, f)

    return True


""" Does not work YET, no meanginful data learned"""


def trainPetUNetSGD():
    losses = []

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    train_dataset, test_dataset = OxfordPetDatasetLoader(2)

    UNet = PetUNet().to(device)

    # loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(UNet.parameters(), lr=0.01, momentum=0.90)

    for epoch in range(NUM_EPOCHS):
        rolling_loss = []
        UNet.train()
        run = 0
        for image, mask in train_dataset:
            image = mirror_pad_to_size(image, 572).to(device)
            mask = mirror_pad_to_size(mask, 572).to(device)
            # Select random 572x572 section of image + corresponding part of mask
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

            center_cropped_mask = TF.center_crop(
                cropped_mask, output_size=(388, 388)
            ).to(device)

            # Oxford pets come as 1: Animal, 2: Background, 3: Border
            # Rework it so 0: Not Animal, 1: Animal
            center_cropped_mask = (center_cropped_mask == 1).long()
            pet_frac = center_cropped_mask.float().mean().item()
            output = UNet(cropped_image)

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

        display_image_and_mask(
            cropped_image,
            center_cropped_mask,
            f"images/full_loop_sgd/regular-epoch-{epoch}.jpg",
        )
        display_image_and_mask(
            cropped_image,
            convert_model_output_to_values(output),
            f"images/full_loop_sgd/model-epoch-{epoch}.jpg",
        )

        logger.info(
            f"Average Loss for epoch {epoch}: {sum(rolling_loss) / len(rolling_loss)}"
        )
        torch.save(
            UNet.state_dict(),
            f"model_state_dicts/full_model_sgd/{epoch}-policy_net.pth",
        )

    with open("losses-adam.pkl", "wb") as f:
        pickle.dump(losses, f)

    return True


"""
Sanity check training loop, train our model on a single image and crop and make sure it can learn 
"""


def trainPetUNetSingleItem():
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

    # Oxford pets come as 1: Animal, 2: Background, 3: Border
    # Rework it so 0: Not Animal, 1: Animal
    center_cropped_mask = (center_cropped_mask == 1).long()

    # Ensure mask is on the same device as model
    center_cropped_mask = center_cropped_mask.to(device)
    center_cropped_mask = center_cropped_mask.squeeze(1)  # remove channel
    display_image_and_mask(
        cropped_image,
        center_cropped_mask,
        f"images/single_image/regular-mask.jpg",
    )

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

        torch.save(
            UNet.state_dict(),
            f"model_state_dicts/single_image_sgd/{epoch}-policy_net.pth",
        )

        output_mask = convert_model_output_to_values(output)
        display_image_and_mask(
            cropped_image, output_mask, f"images/single_image/model-epoch-{epoch}.jpg"
        )

    with open("losses/losses_single_image.pkl", "wb") as f:
        pickle.dump(losses, f)

    return True


if __name__ == "__main__":
    print(trainPetUNetSingleItem())
