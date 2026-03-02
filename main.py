from data_loader import OxfordPetDatasetLoader, OxfordPetDatasetLoaderNoChanges
from helper_functions import display_image_and_mask
from models import PetUNet
from training_loop import generate_random_crop_bounds
import numpy as np
from PIL import Image


def main():
    unet_model = PetUNet()
    trainPetUNetADAM('full_model_adam_50_epoch', unet_model)


if __name__ == "__main__":
    main()
