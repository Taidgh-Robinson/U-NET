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

    train_dataset, test_dataset = OxfordPetDatasetLoader(2)

    # Punish the model for guessing Pets wrong more than we punish it for guessing background wrong
    # Image's are largly background so if we don't do this it'll just guess background for everything
    ratio = compute_class_ratio(train_dataset, device)
    weights = torch.tensor([1.0 / (1 - ratio), 1.0 / ratio]).to(device)

    UNet = PetUNet().to(device)

    loss_fn = nn.CrossEntropyLoss(weight=weights).to(device)
    optimizer = optim.SGD(UNet.parameters(), lr=0.02, momentum=0.9)

    for epoch in range(15):
        rolling_loss = []
        UNet.train()
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

            output = UNet(cropped_image)

            # Calculate loss between model output + centered cropped section of mask
            optimizer.zero_grad()
            # Remove channel from center cropped mask:
            center_cropped_mask = center_cropped_mask.squeeze(1)

            loss = loss_fn(output, center_cropped_mask)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            rolling_loss.append(loss.item())

            # print(output.size())

        print(
            f"Average Loss for epoch {epoch}: {sum(rolling_loss) / len(rolling_loss)}"
        )
        torch.save(UNet.state_dict(), f"modelz/{epoch}-policy_net.pth")

    with open("losses.pkl", "wb") as f:
        pickle.dump(losses, f)

    return True


def trainPetUNetAdam():
    losses = []

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    train_dataset, test_dataset = OxfordPetDatasetLoader(2)

    # Punish the model for guessing Pets wrong more than we punish it for guessing background wrong
    # Image's are largly background so if we don't do this it'll just guess background for everything
    ratio = compute_class_ratio(train_dataset, device)
    weights = torch.tensor([1.0 / (1 - ratio), 1.0 / ratio]).to(device)

    UNet = PetUNet().to(device)

    loss_fn = nn.CrossEntropyLoss(weight=weights).to(device)
    optimizer = torch.optim.Adam(UNet.parameters(), lr=1e-3)

    for epoch in range(50):
        rolling_loss = []
        UNet.train()
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

            output = UNet(cropped_image)

            # Calculate loss between model output + centered cropped section of mask
            optimizer.zero_grad()
            # Remove channel from center cropped mask:
            center_cropped_mask = center_cropped_mask.squeeze(1)

            loss = loss_fn(output, center_cropped_mask)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            rolling_loss.append(loss.item())

            # print(output.size())
        if epoch % 3 == 0:
            UNet.eval()
            display_image_and_mask(
                cropped_image, center_cropped_mask, f"regular-{epoch}.jpg"
            )
            with torch.no_grad():
                output = UNet(cropped_image)
                output = convert_model_output_to_values(output)
                display_image_and_mask(cropped_image, output, f"model-epoch.jpg")

        print(
            f"Average Loss for epoch {epoch}: {sum(rolling_loss) / len(rolling_loss)}"
        )
        torch.save(UNet.state_dict(), f"modelz/{epoch}-adam-policy_net.pth")

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
            UNet.state_dict(), f"modelz/single_image_adam/{epoch}-policy_net.pth"
        )

    with open("losses_single_image.pkl", "wb") as f:
        pickle.dump(losses, f)

    display_image_and_mask(cropped_image, center_cropped_mask)
    output = UNet(cropped_image)
    output_mask = convert_model_output_to_values(output)
    print(output_mask)
    print(output_mask.size())
    display_image_and_mask(cropped_image, output_mask)
    return True


if __name__ == "__main__":
    print(trainPetUNetAdam())
