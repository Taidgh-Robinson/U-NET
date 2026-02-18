import matplotlib.pyplot as plt
import torch


def plot_loss(loss_array, title="Training Loss", xlabel="Iteration", ylabel="Loss"):
    """
    Plots a loss array.

    Parameters:
        loss_array (list or array-like): The loss values to plot.
        title (str): Title of the plot.
        xlabel (str): Label for x-axis.
        ylabel (str): Label for y-axis.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(loss_array, marker="o", linestyle="-", color="blue", alpha=0.7)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()


def display_image_and_mask(image, mask, filename=None):
    # Make sure tensors are CPU and NumPy
    # Select first sample from batch
    if len(image.size()) == 3:
        image = image.squeeze(0)

    if len(mask.size()) == 3:
        mask = mask.squeeze(0)

    if isinstance(image, torch.Tensor):
        image_np = image.detach().cpu().numpy()
    else:
        image_np = image

    if isinstance(mask, torch.Tensor):
        mask_np = mask.detach().cpu().numpy()
    else:
        mask_np = mask

    print(len(image.size()))

    # Rescale image if normalized [-1,1]
    image_np = (image_np + 1.0) / 2.0
    image_np = image_np.clip(0, 1)

    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.title("Image")
    plt.imshow(image_np, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Mask")
    plt.imshow(mask_np, cmap="tab20")  # good for multi-class masks
    plt.axis("off")

    if filename:
        plt.savefig(filename)
        plt.close()
        print(f"Saved visualization to {filename}")
    else:
        plt.show()
