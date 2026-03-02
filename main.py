from data_loader import OxfordPetDatasetLoader, OxfordPetDatasetLoaderNoChanges
from helper_functions import display_image_and_mask
from models import PetUNet
from training_loop import generate_random_crop_bounds
import numpy as np
from PIL import Image
import torch
import random
from training_loop import trainPetUNetADAM

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
    unet_model = PetUNet()
    trainPetUNetADAM("full_model_adam_50_epoch", unet_model)


if __name__ == "__main__":
    main()
