# Elasitic deform
# Traditional deform
# Feed image section into model
# Calculate loss between model output + centered cropped section of mask
from data_loader import OxfordPetDatasetLoader
from models import PetUNet
from random import randrange
from helper_functions import display_image_and_mask
import torchvision.transforms.functional as TF
import torch.optim as optim
import torch
import torch.nn as nn
import pickle


def generate_random_crop_bounds(image, crop_size):
    h = image.size()[2]
    w = image.size()[3]
    crop_boundry_h = h - crop_size
    crop_boundry_w = w - crop_size
    h_boundry = randrange(0, crop_boundry_h)
    w_boundry = randrange(0, crop_boundry_w)
    return ((h_boundry, h_boundry + crop_size), (w_boundry, w_boundry + crop_size))


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

    UNet = PetUNet().to(device)
    loss_fn = nn.CrossEntropyLoss().to(device)
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

            loss = loss_fn(output.to(device), center_cropped_mask.long().to(device))
            loss.backward()
            optimizer.step()
            print(loss)
            losses.append(loss.item())
            # print(output.size())

        torch.save(UNet.state_dict(), f"modelz/{epoch}-policy_net.pth")

    with open("losses.pkl", "wb") as f:
        pickle.dump(losses, f)

    return True


if __name__ == "__main__":
    print(trainPetUNet())
