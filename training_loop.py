



# Elasitic deform
# Traditional deform 
# Feed image section into model 
# Calculate loss between model output + centered cropped section of mask 
from data_loader import OxfordPetDatasetLoader
from models import PetUNet
from random import randrange

def generate_random_crop_bounds(image, crop_size):
    h = image.size()[1]
    w = image.size()[2]
    crop_boundry_h = h - crop_size
    crop_boundry_w = w - crop_size
    h_boundry = randrange(0, crop_boundry_h)
    w_boundry = randrange(0, crop_boundry_w)
    return ((h_boundry, h_boundry+crop_size), (w_boundry, w_boundry+crop_size))
    

def elastic_deformation(image, mask):
    pass

def traditional_deformation(image, mask):
    pass



def trainPetUNet():
    train_dataset, test_dataset = OxfordPetDatasetLoader(2)
    UNet = PetUNet()
    all_greater = True
    for epoch in range(1):
        # Pull sample image + mask
        for image, mask in train_dataset:
            print(image.size())
            print(mask.size())
            if image.size()[2] < 572 or image.size()[3] < 572:
                continue
            
            # Training in batches of 1 so can remove batch dim
            image = image.squeeze(0)
            # Select random 572x572 section of image + corresponding part of make
            crop_boundies = generate_random_crop_bounds(image, 572)
            cropped_image = image[:, crop_boundies[0][0]:crop_boundies[0][1], crop_boundies[1][0]:crop_boundies[1][1]]
            cropped_mask = mask[:, crop_boundies[0][0]:crop_boundies[0][1], crop_boundies[1][0]:crop_boundies[1][1]]
            print(cropped_image.size())


            #output = UNet(cropped)
            #print(output.size())
            break 



    return True
if __name__ == "__main__":
    print(trainPetUNet())