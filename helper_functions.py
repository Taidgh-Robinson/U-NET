import matplotlib.pyplot as plt

def display_image_and_mask(image, mask):
    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.title("Image")
    plt.imshow(image.squeeze(0), cmap="gray")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.title("Mask")
    plt.imshow(mask.squeeze(0), cmap="gray")
    plt.axis("off")

    plt.show()

def save_unedited_image_and_mask():
    