from data_loader import OxfordPetDatasetLoader, OxfordPetDatasetLoaderNoChanges
from helper_functions import display_image_and_mask
from models import PetUNet
import numpy as np
from PIL import Image


def main():
    train, test = OxfordPetDatasetLoaderNoChanges(572)
    img, mask = train[0]
    mask_array = np.array(mask)

    # Scale values for visibility
    # (so small values like 1,2,3 become visible)
    mask_scaled = (mask_array * (255 // mask_array.max())).astype(np.uint8)

    # Convert back to PIL Image
    mask_display = Image.fromarray(mask_scaled)
    mask_display.save('mask_before_edit.png')
    #display_image_and_mask(img, mask)
    #p = PetUNet()
    #p(img)


if __name__ == "__main__":
    main()
