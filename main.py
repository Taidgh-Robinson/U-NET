from data_loader import OxfordPetDatasetLoader, OxfordPetDatasetLoaderNoChanges
from helper_functions import display_image_and_mask
from models import PetUNet
from training_loop import generate_random_crop_bounds
import numpy as np
from PIL import Image


def main():
    train, test = OxfordPetDatasetLoader(2)
    img, mask = next(iter(train))
    display_image_and_mask(img, mask)
    # mask_array = np.array(mask)
    crop_boundies = generate_random_crop_bounds(img, 572)
    cropped_image = img[
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

    # Scale values for visibility
    # (so small values like 1,2,3 become visible)
    # mask_scaled = (mask_array * (255 // mask_array.max())).astype(np.uint8)

    # Convert back to PIL Image
    # mask_display = Image.fromarray(mask_scaled)
    # mask_display.save('mask_before_edit.png')
    # display_image_and_mask(img, mask)
    p = PetUNet()
    p(cropped_image)


if __name__ == "__main__":
    main()
