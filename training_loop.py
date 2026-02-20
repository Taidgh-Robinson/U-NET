# Elasitic deform
# Traditional deform
# Feed image section into model
# Calculate loss between model output + centered cropped section of mask
from data_loader import OxfordPetDatasetLoader
from models import PetUNet
from helper_functions import display_image_and_mask, convert_model_output_to_values, generate_random_crop_bounds
import torchvision.transforms.functional as TF
import torch.optim as optim
import torch
import numpy as np
import random
import torch.nn as nn
import pickle

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


def trainPetUNet():
    losses = []

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    UNet = PetUNet()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(UNet.parameters(), lr=0.01, momentum=0.99)
    train_dataset, test_dataset = OxfordPetDatasetLoader(2)
    for epoch in range(1):
        UNet.train()
        # Pull sample image + mask
        for image, mask in train_dataset:
            if image.size()[2] <= 572 or image.size()[3] <= 572:
                continue

            # Select random 572x572 section of image + corresponding part of make
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
            center_cropped_mask_mask = center_cropped_mask == 3
            center_cropped_mask[center_cropped_mask_mask] = 2
            center_cropped_mask_mask = center_cropped_mask == 2
            center_cropped_mask[center_cropped_mask_mask] = 0

            output = UNet(cropped_image)

            # Calculate loss between model output + centered cropped section of mask
            optimizer.zero_grad()
            #Remove channel from center cropped mask:
            center_cropped_mask = center_cropped_mask.squeeze(1)

            loss = loss_fn(output, center_cropped_mask.long())
            loss.backward()
            optimizer.step()
            print(loss)
            losses.append(loss.item())
            # print(output.size())

        torch.save(UNet.state_dict(), f"modelz/{epoch}-policy_net.pth")

    with open("losses.pkl", "wb") as f:
        pickle.dump(losses, f)

    return True

def trainPetUNetSingleItem():
    losses = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    UNet = PetUNet().to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(UNet.parameters(), lr=0.1, momentum=0.90)

    train_dataset, test_dataset = OxfordPetDatasetLoader(2)

    image, mask = next(iter(train_dataset))
    if image.size()[2] <= 572 or image.size()[3] <= 572:
        image, mask = next(iter(train_dataset))

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

    for epoch in range(5000):
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

        torch.save(UNet.state_dict(), f"modelz/single_image_adam/{epoch}-policy_net.pth")

    with open("losses_adam.pkl", "wb") as f:
        pickle.dump(losses, f)

    display_image_and_mask(cropped_image, center_cropped_mask)
    output = UNet(cropped_image)
    output_mask = convert_model_output_to_values(output)
    print(output_mask)
    print(output_mask.size())
    display_image_and_mask(cropped_image, output_mask)
    return True


if __name__ == "__main__":
    print(trainPetUNetSingleItem())
