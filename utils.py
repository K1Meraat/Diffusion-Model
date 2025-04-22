import os
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import logging



def plot_images(images):
    """
    Plot a 4x4 grid of images.
    
    Args:
        images: Tensor of images [batch, channels, height, width]
    """
    plt.figure(figsize=(16, 16))
    
    # Create a 4x4 grid
    # For each image in the batch (up to 16 images)
    for i in range(min(16, len(images))):
        # Create a subplot in a 4x4 grid at position i+1
        plt.subplot(4, 4, i + 1)
        # Convert from [-1, 1] to [0, 1]
        img = (images[i].permute(1, 2, 0).cpu() + 1) / 2
        plt.imshow(img, cmap='gray')
        # Remove axis ticks and labels for cleaner visualization
        plt.axis('off')
    # Ensure proper spacing between subplots
    plt.tight_layout()
    plt.show()

def save_images(images, path):
    """
    Save a 4x4 grid of images to a file.
    
    Args:
        images: Tensor of images [batch, channels, height, width]
        path: Save path for the grid image
    """
    plt.figure(figsize=(12, 12))
    
    # Create a 4x4 grid
    for i in range(min(16, len(images))):
        plt.subplot(4, 4, i + 1)
        # Convert from [-1, 1] to [0, 1]
        img = (images[i].permute(1, 2, 0).cpu() + 1) / 2
        plt.imshow(img, cmap='gray')
        plt.axis('off')
    
    plt.tight_layout()
    # Save the figure to the specified path with high resolution
    plt.savefig(path, dpi=300)
    plt.close()
    # Log that the images were saved successfully
    logging.info(f"Saved generated images to {path}")


# def plot_images(images):
#     plt.figure(figsize=(16, 16))
#     plt.imshow(torch.cat([
#         torch.cat([i for i in images.cpu()], dim=-1),
#         #PyTorch and torchvision functions like make_grid() output tensors in the [C, H, W] format
#         #But visualization libraries like matplotlib expect images in the [H, W, C] format (height, width, then color channels)
#     ], dim=-2).permute(1, 2, 0).cpu())
#     plt.show()


# def save_images(images, path, **kwargs):
#     grid = torchvision.utils.make_grid(images, **kwargs)
#     ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
#     im = Image.fromarray(ndarr)
#     im.save(path)


def get_data(args):
    
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(160),  # args.image_size + 1/4 *args.image_size
        torchvision.transforms.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)),
        torchvision.transforms.ToTensor(),
        #torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        torchvision.transforms.Normalize((0.5, 0.5), (0.5, 0.5))
    ])
    dataset = torchvision.datasets.ImageFolder(args.dataset_path, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return dataloader


def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)
