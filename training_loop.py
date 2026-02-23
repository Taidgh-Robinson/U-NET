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
    compute_class_ratio,
)
import torchvision.transforms.functional as TF
import torch.optim as optim
import torch
import numpy as np
import random
import torch.nn as nn
import pickle
from logger_config import logger

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
    return 1 - (2. * intersection + smooth) / (pred_fg.sum() + target_float.sum() + smooth)

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

    # Punish the model for guessing Pets wrong more than we punish it for guessing background wrong
    # Image's are largly background so if we don't do this it'll just guess background for everything
    #ratio = compute_class_ratio(train_dataset, device)
    #weights = torch.tensor([1.0 / (1 - ratio), 1.0 / ratio]).to(device)
    #weights = torch.tensor([50.0, 1.0]).to(device)
    UNet = PetUNet().to(device)

    #loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(UNet.parameters(), lr=1e-4)

    for epoch in range(25):
        rolling_loss = []
        UNet.train()
        run = 0
        # Pull sample image + mask
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
            logger.info(f'Loss at current step is {loss.item()}')
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            rolling_loss.append(loss.item())
            
            display_image_and_mask(
                cropped_image, center_cropped_mask, f"images/full_loop/regular-epoch-{epoch}-{run}.jpg"
            )
            display_image_and_mask(
                cropped_image, convert_model_output_to_values(output), f"images/full_loop/model-epoch-{epoch}-{run}.jpg"
            )
            run += 1
            # print(output.size())

        print(
            f"Average Loss for epoch {epoch}: {sum(rolling_loss) / len(rolling_loss)}"
        )
        torch.save(UNet.state_dict(), f"model_state_dicts/full_model/{epoch}-single-epoch-policy_net.pth")

    with open("losses-adam.pkl", "wb") as f:
        pickle.dump(losses, f)

    return True


def trainPetUNetSingleEpoch():
    losses = []

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    train_dataset, test_dataset = OxfordPetDatasetLoader(2)

    # Punish the model for guessing Pets wrong more than we punish it for guessing background wrong
    # Image's are largly background so if we don't do this it'll just guess background for everything
    #ratio = compute_class_ratio(train_dataset, device)
    #weights = torch.tensor([1.0 / (1 - ratio), 1.0 / ratio]).to(device)
    #weights = torch.tensor([50.0, 1.0]).to(device)
    UNet = PetUNet().to(device)

    #loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(UNet.parameters(), lr=1e-4)

    for epoch in range(1):
        rolling_loss = []
        UNet.train()
        run = 0
        # Pull sample image + mask
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
            logger.info(f'Loss at current step is {loss.item()}')
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            rolling_loss.append(loss.item())
            
            display_image_and_mask(
                cropped_image, center_cropped_mask, f"images/single_epoch/regular-epoch-{run}.jpg"
            )
            display_image_and_mask(
                cropped_image, convert_model_output_to_values(output), f"images/single_epoch/model-run-{run}.jpg"
            )
            run += 1
            # print(output.size())

        print(
            f"Average Loss for epoch {epoch}: {sum(rolling_loss) / len(rolling_loss)}"
        )
        torch.save(UNet.state_dict(), f"model_state_dicts/single_image_sgd/{epoch}-single-epoch-policy_net.pth")

    with open("losses-adam.pkl", "wb") as f:
        pickle.dump(losses, f)

    return True

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

    for epoch in range(1000):
        UNet.train()

        # Select random 572x572 section of image + corresponding part of mask

        output = UNet(cropped_image)
        # Calculate loss between model output + centered cropped section of mask
        optimizer.zero_grad()

        loss = loss_fn(output, center_cropped_mask)
        loss.backward()
        optimizer.step()

        print(loss.item())
        losses.append(loss.item())

        torch.save(
            UNet.state_dict(), f"model_state_dicts/single_image_sgd/{epoch}-policy_net.pth"
        )

        display_image_and_mask(cropped_image, center_cropped_mask, f'images/single_image/regular-epoch-{epoch}.jpg')
        output_mask = convert_model_output_to_values(output)
        display_image_and_mask(cropped_image, output_mask, f'images/single_image/model-epoch-{epoch}.jpg')

    with open("losses_single_image.pkl", "wb") as f:
        pickle.dump(losses, f)

    return True


if __name__ == "__main__":
    print(trainPetUNet())
