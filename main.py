from data_loader import OxfordPetDatasetLoader, OxfordPetDatasetLoaderNoChanges
from helper_functions import display_image_and_mask, calculate_final_model_accuracy
from models import PetUNet, PetUNetColor
from training_loop import generate_random_crop_bounds
import numpy as np
from PIL import Image
import torch
import random
from training_loop import trainPetUNetADAM
from data_loader import OxfordPetDatasetLoader, OxfordPetDatasetLoaderColor
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
    #train, test = OxfordPetDatasetLoader(2)
    #unet_model = PetUNet()
    #unet_model = unet_model.to(device)   # ← THIS LINE

    #trainPetUNetADAM("full_model_adam_50_epoch", unet_model)
    #state_dict = torch.load("model_state_dict/full_model_adam_50_epoch/policy_net-49.pth", map_location=device)
    #unet_model.load_state_dict(state_dict)

    #print(calculate_final_model_accuracy(unet_model, device, test))
    
    unet_model = PetUNetColor()
    train_dataset, test_dataset = OxfordPetDatasetLoaderColor(2)
    #trainPetUNetADAM('full_model_adam_color_50_epoch', unet_model, train_dataset, True)
    state_dict = torch.load("model_state_dict/full_model_adam_color_50_epoch/policy_net-49.pth", map_location=device)
    unet_model.load_state_dict(state_dict)
    print(calculate_final_model_accuracy(unet_model, device, test_dataset))
    
    """
    Train white paper UNet:
    unet_model = PetUNet()
    train_dataset, _ = OxfordPetDatasetLoader(2)
    trainPetUNetADAM('full_model_adam', unet_model, train_dataset)
    
    """    

    """
    Train color UNet:
    unet_model = PetUNetColor()
    train_dataset, _ = OxfordPetDatasetLoaderColor(2)
    trainPetUNetADAM('full_model_adam_color_50_epoch', unet_model, train_dataset, True)
    """

if __name__ == "__main__":
    main()
