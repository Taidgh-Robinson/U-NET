import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF

def OxfordPetDatasetLoader(image_scale):

    
    image_transform = transforms.Compose(
        [
            #We have to scale the images by a factor since they are samller than 572X572 which is what the original architecture of the model was, and that is what I'm trying to faithfully reproduce
            transforms.Lambda(
                lambda img: TF.resize(
                    img,
                    [img.height * image_scale, img.width * image_scale ],
                    interpolation=TF.InterpolationMode.BILINEAR,
                )
            ),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )

    mask_transform = transforms.Compose(
        [
            #Same deal for the masks, it isn't perfect but its honest work. 
            transforms.Lambda(
                lambda img: TF.resize(
                    img,
                    [img.height * image_scale, img.width * image_scale],
                    interpolation=TF.InterpolationMode.NEAREST,
                )
            ),
            transforms.PILToTensor(),
        ]
    )

    train_dataset = datasets.OxfordIIITPet(
        root="./data",
        split="trainval",
        target_types="segmentation",
        download=True,
        transform=image_transform,
        target_transform=mask_transform,
    )

    test_dataset = datasets.OxfordIIITPet(
        root="./data",
        split="test",
        target_types="segmentation",
        download=True,
        transform=image_transform,
        target_transform=mask_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True   
    )
    test_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True
    )


    return (train_loader, test_loader)


def OxfordPetDatasetLoaderNoChanges(IMAGE_SIZE):
    image_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])]
    )

    mask_transform = transforms.Compose(
        [
            transforms.PILToTensor()  # keeps integer labels
        ]
    )

    train_dataset = datasets.OxfordIIITPet(
        root="./data",
        split="trainval",
        target_types="segmentation",
        download=True,
    )

    test_dataset = datasets.OxfordIIITPet(
        root="./data",
        split="test",
        target_types="segmentation",
        download=True,
    )

    return (train_dataset, test_dataset)
