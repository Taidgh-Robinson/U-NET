import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def OxfordPetDatasetLoader(IMAGE_SIZE):
    image_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    mask_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.CenterCrop(388),      # crop center to match network output
        transforms.PILToTensor()  # keeps integer labels
    ])

    train_dataset = datasets.OxfordIIITPet(
        root="./data",
        split="trainval",
        target_types="segmentation",
        download=True,
        transform=image_transform,
        target_transform=mask_transform
    )

    test_dataset = datasets.OxfordIIITPet(
        root="./data",
        split="test",
        target_types="segmentation",
        download=True,
        transform=image_transform,
        target_transform=mask_transform
    )

    return (train_dataset, test_dataset)

def OxfordPetDatasetLoaderNoChanges(IMAGE_SIZE):
    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    mask_transform = transforms.Compose([
        transforms.PILToTensor()  # keeps integer labels
    ])

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
