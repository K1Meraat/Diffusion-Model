# # -*- coding: utf-8 -*-
# """
# Created on Sat Apr  5 16:47:45 2025

# @author: K1
# """

# import torch
# import torch.nn as nn
# import math




# class TimeEmbedding(nn.Module):
#     """Processes timesteps into a format usable by the UNet."""
#     def __init__(self, time_dim=64, device="cuda"):
#         super().__init__()
#         self.device = device
#         self.time_mlp = nn.Sequential(
#             #model has more room to learn complex patterns
#             nn.Linear(time_dim, time_dim * 2),
#             #to introduce non-linearity
#             nn.ReLU(),
#             nn.Linear(time_dim * 2, time_dim)
#         )

#     def forward(self, timestep):
#         """Create time embeddings, a vector corresponding to timesteps where lower indices 
#         are for broader pattern(a face, a car) and higher indices are for finer details(a pointy edge)."""
#         half_dim = 32  # embedding dimension = 64
#         #create a tensor 0,..31
#         embeddings = torch.exp(torch.arange(half_dim, device=timestep.device) * 
#                                #control the rate of exponential decay -9.21 / 31 ≈ -0.297
#                                (-math.log(10000) / (half_dim - 1)))
#         #scale values by the timestep
#         embeddings = timestep[:, None] * embeddings[None, :]
        
#         embeddings = torch.cat((torch.sin(embeddings), torch.cos(embeddings)), dim=-1)
#         return self.time_mlp(embeddings)

# class UNet(nn.Module):
#     """U-Net for noise prediction with skip connections."""
#     def __init__(
#         self,
#         img_channels=3,      # Number of image channels
#         base_channels=32,    # base channels
#         time_dim=64,         # time embedding dimension
#         device="cuda"
#     ):
#         super().__init__()
        
#         # Store image channels for later use in generation
#         self.img_channels = img_channels
        
#         # Time embedding
#         self.time_embedding = TimeEmbedding(time_dim, device)
        
#         # Initial convolution
#         self.initial_conv = nn.Sequential(
#             #transorm images from image channel to base channel
#             nn.Conv2d(img_channels, base_channels, kernel_size=3, padding=1),
#             #prevent large changes across the distribution
#             nn.GroupNorm(8, base_channels),
#             #x*sigmoid(x)
#             nn.SiLU()
#         )
        
#         # Downsampling path with skip connections
#         #capture global context instead of fine details
#         self.down1 = nn.Sequential(
#             nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
#             nn.GroupNorm(8, base_channels * 2),
#             nn.SiLU()
#         )
        
#         # Bottleneck
#         self.bottleneck = nn.Sequential(
#             nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1),
#             nn.GroupNorm(8, base_channels * 2),
#             nn.SiLU(),
#             nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1),
#             nn.GroupNorm(8, base_channels * 2),
#             nn.SiLU()
#         )
        
#         # Upsampling path with skip connections
#         self.up1 = nn.Sequential(
#             nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1),
#             nn.GroupNorm(8, base_channels),
#             nn.SiLU()
#         )
        
#         # Skip connection convolution to match channels
#         self.skip_conv = nn.Conv2d(base_channels, base_channels, kernel_size=1)
        
#         # Final convolution to predict noise
#         self.final_conv = nn.Sequential(
#             nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, padding=1),
#             nn.GroupNorm(8, base_channels),
#             nn.SiLU(),
#             nn.Conv2d(base_channels, img_channels, kernel_size=3, padding=1)
#         )

#     def forward(self, x, timestep):
#         """forward pass with skip connections."""
#         # Time embedding
#         time_emb = self.time_embedding(timestep)
        
#         # Initial processing
#         h = self.initial_conv(x)
#         skip_connection = h  # Store initial feature map for skip connection
        
#         # Downsampling
#         h = self.down1(h)
        
#         #I’ve looked at the image (downsampled it), now tell me how much noise I should expect.
#         # Add time embedding
#         #reshape time_emb to h so addition is possible
#         time_emb_reshaped = time_emb.reshape(time_emb.shape[0], -1, 1, 1)
#         h = h + time_emb_reshaped
        
#         # Bottleneck
#         h = self.bottleneck(h)
        
#         # Upsampling
#         h = self.up1(h)
        
#         # Process skip connection
#         skip_connection = self.skip_conv(skip_connection)
        
#         # Concatenate skip connection with upsampled features
#         #combine them along channel axis to bring back early details
#         h = torch.cat([h, skip_connection], dim=1)
        
#         # Final noise prediction
#         return self.final_conv(h)


















import torch
import torch.nn as nn
import math


class TimeEmbedding(nn.Module):
    """Processes timesteps into a format usable by the UNet."""
    def __init__(self, time_dim=128, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            #model has more room to learn complex patterns
            nn.Linear(time_dim, time_dim * 2),
            #to introduce non-linearity
            #The Rectified Linear Unit
            #determines the output of a neuron
            nn.ReLU(),
            nn.Linear(time_dim * 2, time_dim)
        )

    def forward(self, timestep):
        """Create time embeddings, a vector corresponding to timesteps where lower indices 
        are for broader pattern(a face, a car) and higher indices are for finer details(a pointy edge)."""
        #time_dim // 2
        half_dim = self.time_dim // 2 
        #create a tensor 0,..half_dim
        embeddings = torch.exp(torch.arange(half_dim, device=timestep.device) * 
                               #control the rate of exponential decay -9.21 / half_dim
                               (-math.log(10000) / (half_dim - 1)))
        #scale values by the timestep
        embeddings = timestep[:, None] * embeddings[None, :]
        
        embeddings = torch.cat((torch.sin(embeddings), torch.cos(embeddings)), dim=-1)
        return self.time_mlp(embeddings)

class UNet(nn.Module):
    """U-Net for noise prediction with skip connections."""
    def __init__(
        self,
        img_channels=3,      # Number of image channels
        base_channels=32,    # base channels
        time_dim=128,         # time embedding dimension
        device="cuda"
    ):
        super().__init__()
        
        # Store image channels for later use in generation
        self.img_channels = img_channels
        
        # Time embedding
        self.time_embedding = TimeEmbedding(time_dim, device)
        
        # Initial convolution
        self.initial_conv = nn.Sequential(
            #transorm images from image channel to base channel
            nn.Conv2d(img_channels, base_channels, kernel_size=3, padding=1),
            #prevent large changes across the distribution
            nn.GroupNorm(8, base_channels),
            #x*sigmoid(x)
            #Sigmoid Linear Unit
            #determines the output of a neuron
            #smoothness and preventing dead neurons
            nn.SiLU()
        )
        
        # Downsampling path with skip connections
        # Capture global context instead of fine details
        self.down1 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, base_channels * 2),
            nn.SiLU()
        )
        
        # NEW: Additional downsampling layer
        self.down2 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, base_channels * 4),
            nn.SiLU()
        )
        
        # Bottleneck (now with 4x base channels)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_channels * 4),
            #smoothness and preventing dead neurons
            nn.SiLU(),
            nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_channels * 4),
            nn.SiLU()
        )
        
        # NEW: First upsampling layer (from 4x to 2x base channels)
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, base_channels * 2),
            nn.SiLU()
        )
        
        # Second upsampling layer (from 2x to 1x base channels)
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.SiLU()
        )
        
        # Skip connection convolutions
        self.skip_conv1 = nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=1)
        self.skip_conv2 = nn.Conv2d(base_channels, base_channels, kernel_size=1)
        
        # Final convolution to predict noise
        self.final_conv = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, img_channels, kernel_size=3, padding=1)
        )

    def forward(self, x, timestep):
        """forward pass with skip connections."""
        # Time embedding
        time_emb = self.time_embedding(timestep)
        time_emb_reshaped = time_emb.reshape(time_emb.shape[0], -1, 1, 1)
        
        # Initial processing
        h = self.initial_conv(x)
        skip_connection2 = h  # Store first skip connection
        
        # First downsampling
        h = self.down1(h)
        skip_connection1 = h  # Store second skip connection
        
        # Second downsampling
        h = self.down2(h)
        
        # Add time embedding
        h = h + time_emb_reshaped
        
        # Bottleneck
        h = self.bottleneck(h)
        
        # First upsampling
        h = self.up1(h)
        
        # Apply first skip connection
        skip_connection1 = self.skip_conv1(skip_connection1)
        #Concatenate skip connection with upsampled features
        #combine them along channel axis to bring back early details
        h = torch.cat([h, skip_connection1], dim=1)
        
        # Second upsampling
        h = self.up2(h)
        
        # Apply second skip connection
        skip_connection2 = self.skip_conv2(skip_connection2)
        h = torch.cat([h, skip_connection2], dim=1)
        
        # Final noise prediction
        return self.final_conv(h)