import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import logging
from unet import UNet
from utils import save_images

# ========================================================================
# 1.   DIFFUSION PROCESS CLASS
# ========================================================================
class Diffusion:
    """
    diffusion process for image generation.
    """
    def __init__(
        self,
        noise_steps=500,    # number of noise steps
        beta_start=1e-4,    # Starting variance 
        beta_end=0.02,      # Ending variance
        img_size=32,        # image size
        device="cuda"       # Device to run calculations on
    ):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        #noise schedule
        self.beta = self._linear_beta_schedule().to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def _linear_beta_schedule(self):
        """Creates a linear schedule for noise variance."""
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def _extract_timestep_values(self, tensor, timesteps, shape):
        """Extract values for specific timesteps."""
        batch_size = timesteps.shape[0]
        sample = tensor.gather(-1, timesteps.to(self.device))
        return sample.reshape(batch_size, *((1,) * (len(shape) - 1)))

    def add_noise(self, original_images, timesteps):
        """Forward diffusion process: Add noise to images."""
        
        #sample alpha_bar based on timestep
        sqrt_alpha_bar = torch.sqrt(
            self._extract_timestep_values(self.alpha_bar, timesteps, original_images.shape)
        )
        
        sqrt_one_minus_alpha_bar = torch.sqrt(
            1.0 - self._extract_timestep_values(self.alpha_bar, timesteps, original_images.shape)
        )
        #sample noise
        noise = torch.randn_like(original_images)
        
        noisy_images = (
            sqrt_alpha_bar * original_images +
            sqrt_one_minus_alpha_bar * noise
        )
        
        return noisy_images, noise

    def sample_random_timesteps(self, batch_size):
        """Randomly sample timesteps with size of batch."""
        return torch.randint(1, self.noise_steps, (batch_size,), device=self.device)
# ========================================================================
# 2. GENERATING FUNCTION
# ========================================================================
    def generate(self, model, num_samples=8):
        """reverse diffusion process."""
        model.eval()
        #sample random noise x_T
        noisy_images = torch.randn(
            (num_samples, model.img_channels, self.img_size, self.img_size), 
            device=self.device
        )
        
        for timestep in reversed(range(1, self.noise_steps)):
            #create a tensor with the dimension num_samples and fill it with current timestep
            timesteps = torch.full((num_samples,), timestep, device=self.device, dtype=torch.long)
            
            with torch.no_grad():
                predicted_noise = model(noisy_images, timesteps)
            
            alpha_t = self._extract_timestep_values(self.alpha, timesteps, noisy_images.shape)
            alpha_bar_t = self._extract_timestep_values(self.alpha_bar, timesteps, noisy_images.shape)
            beta_t = self._extract_timestep_values(self.beta, timesteps, noisy_images.shape)
            
            mean_component = (1 / torch.sqrt(alpha_t)) * (
                noisy_images - ((1 - alpha_t) / (torch.sqrt(1 - alpha_bar_t))) * predicted_noise
            )
            
            if timestep > 1:
                noise = torch.randn_like(noisy_images)
            else:
                noise = torch.zeros_like(noisy_images)
                
            #sigma*z
            noise_component = torch.sqrt(beta_t) * noise
            #x_t-1
            noisy_images = mean_component + noise_component
        #safeguard to make sure the output from model is indeed between -1 and 1
        generated_images = (noisy_images.clamp(-1, 1) + 1) / 2
        generated_images = (generated_images * 255).type(torch.uint8)
        
        model.train()
        return generated_images


# ========================================================================
# 3. TRAINING FUNCTION
# ========================================================================
def train_diffusion_model(args):
    """training function."""
    # Setup logging
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    logging.basicConfig(level=logging.INFO)
    
    # Device setup
    device = torch.device(args.device)
    
    # Data transforms
    transform = transforms.Compose([
        #Ensures all images have the same input size for the model.
        transforms.Resize(args.img_size),
        #Crops the center of the resized image to args.img_size x args.img_size
        #Acts as a safeguard to ensure exact dimensions (useful if resizing introduces artifacts).
        transforms.CenterCrop(args.img_size),
        #Converts the image (PIL or NumPy) to a PyTorch tensor and scales pixel values to [0, 1]
        transforms.ToTensor(),
        #Normalizes the tensor to have a mean of 0.5 and standard deviation of 0.5
        #Scales pixel values from [0, 1] → [-1, 1] because UNet works with values from [-1,1]
        transforms.Normalize((0.5,), (0.5))
    ])
    
    # Load dataset
    if args.dataset.lower() == "cifar10":
        dataset = datasets.CIFAR10("./data", train=True, download=True, transform=transform)
        img_channels = 3
    elif args.dataset.lower() == "mnist":
        dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
        img_channels = 1
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    # shuffle at the beggining of epoch to avoid overfitting
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Model initialization
    model = UNet(
        img_channels=img_channels,
        base_channels=args.base_channels,
        time_dim=128,
        device=device
    ).to(device)
    
    # Diffusion process
    diffusion = Diffusion(
        noise_steps=args.noise_steps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        img_size=args.img_size,
        device=device
    )
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Cosine Annealing Learning Rate Scheduler
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, 
    #     T_max=args.epochs, 
    #     eta_min=args.lr * 0.1  # Minimum learning rate
    # )
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=args.lr,
        steps_per_epoch=len(dataloader),
        epochs=args.epochs
    )
    

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, (images, _) in enumerate(dataloader):
            images = images.to(device)
            batch_size = images.shape[0]
            
            # Sample random timesteps
            timesteps = diffusion.sample_random_timesteps(batch_size)
            
            # Forward diffusion
            noisy_images, noise_target = diffusion.add_noise(images, timesteps)
            
            # Predict noise
            noise_pred = model(noisy_images, timesteps)
            
            # Compute loss
            loss = F.mse_loss(noise_target, noise_pred)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
         
        avg_loss = epoch_loss / len(dataloader)
        # Scheduler step
        scheduler.step(avg_loss)
        
        # Log epoch statistics
        logging.info(f"Epoch {epoch + 1} - Average Loss: {avg_loss:.6f}")
        
        # Save model and generate samples periodically
        if epoch % args.sample_interval == 0 or epoch == args.epochs - 1:
            torch.save(model.state_dict(), f"models/model_epoch_{epoch}.pt")
            
            model.eval()
            with torch.no_grad():
                generated_images = diffusion.generate(model, num_samples=16)
                save_images(
                    generated_images, 
                    f"results/samples_epoch_{epoch}.png"
                )
    
    logging.info("Training complete!")

# ========================================================================
# 4. MAIN FUNCTION
# ========================================================================
def main():
    """Parse arguments and start training."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train a diffusion model")
    
    # Run configuration
    parser.add_argument("--run_name", type=str, default="diffusion", help="Run name")
    parser.add_argument("--dataset", type=str, default="mnist", help="Dataset to use")
    parser.add_argument("--img_size", type=int, default=32, help="Image size")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    
    # Model parameters
    parser.add_argument("--base_channels", type=int, default=32, help="Base channel count")
    parser.add_argument("--time_dim", type=int, default=128, help="Time embedding dimension")
    
    # Diffusion parameters
    parser.add_argument("--noise_steps", type=int, default=1000, help="Number of diffusion steps")
    parser.add_argument("--beta_start", type=float, default=1e-4, help="Starting beta value")
    parser.add_argument("--beta_end", type=float, default=0.02, help="Ending beta value")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--sample_interval", type=int, default=10, help="Save samples every N epochs")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on")
    
    args = parser.parse_args()
    train_diffusion_model(args)

if __name__ == "__main__":
    main()