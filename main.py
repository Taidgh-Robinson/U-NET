from data_loader import OxfordPetDatasetLoader
from helper_functions import calculate_final_model_accuracy
from models import PetUNet, PetUNetColor
import numpy as np
import torch
import random
from training_loop import trainPetUNetADAM, trainPetUNetADAMWithRandomTransforms
from data_loader import OxfordPetDatasetLoader, OxfordPetDatasetLoaderColor
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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluate_color_UNet("model_state_dict.pth", device)

def train_white_paper_UNet():
    unet_model = PetUNet()
    train_dataset, _ = OxfordPetDatasetLoader(2)
    trainPetUNetADAM("full_model_adam_{NUM_EPOCHS}_epoch", unet_model, train_dataset)


def evaluate_UNet(model_path, device):
    unet_model = PetUNet()
    _, test_dataset = OxfordPetDatasetLoader(2)
    state_dict = torch.load(
        model_path,
        map_location=device,
    )

    unet_model.load_state_dict(state_dict)
    calculate_final_model_accuracy(unet_model, device, test_dataset)


def train_color_UNet():
    unet_model = PetUNetColor()
    train_dataset, _ = OxfordPetDatasetLoaderColor(2)
    trainPetUNetADAM(
        f"full_model_adam_color_{NUM_EPOCHS}_epoch", unet_model, train_dataset, True
    )


def train_color_UNet_with_random_deforms():
    unet_model = PetUNetColor()
    train_dataset, _ = OxfordPetDatasetLoaderColor(2)
    trainPetUNetADAMWithRandomTransforms(
        f"full_model_adam_color_random_transforms_{NUM_EPOCHS}_epoch",
        unet_model,
        train_dataset,
        True,
    )


def evaluate_color_UNet(model_path, device):
    unet_model = PetUNetColor()
    _, test_dataset = OxfordPetDatasetLoaderColor(2)
    state_dict = torch.load(
        model_path,
        map_location=device,
    )

    unet_model.load_state_dict(state_dict)
    calculate_final_model_accuracy(unet_model, device, test_dataset)


if __name__ == "__main__":
    main()
