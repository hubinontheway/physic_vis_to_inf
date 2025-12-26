from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn

from .physics_layer import PhysicsLayer


class PhysDecompNet(nn.Module):
    """
    Physically inspired decomposition network for separating temperature and emissivity 
    from infrared images.
    
    This network implements a physics-based approach to decompose an infrared image into
    its constituent physical properties: temperature and emissivity. The model is based
    on the physical relationship:
        IR = ε * g(T) + noise
        
    where IR is the infrared image, ε is emissivity, T is temperature, g(T) is a 
    temperature transformation function, and noise is additive noise.
    
    Architecture:
    - Encoder: Extracts features from the input infrared image
    - Three separate heads: Predict temperature, emissivity, and noise respectively
    - Physics Layer: Enforces physical consistency by synthesizing IR from predicted properties
    """


    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        t_min: float = 1.0,
        t_max: float = 10.0,
        use_noise: bool = True,
        use_mlp: bool = False,
    ) -> None:
        """
        Initialize the PhysDecompNet.
        
        Args:
            in_channels (int): Number of input channels (typically 1 for grayscale IR images).
                               Defaults to 1.
            base_channels (int): Base number of channels for the encoder. Defaults to 32.
            t_min (float): Minimum temperature value for scaling. Defaults to 1.0.
            t_max (float): Maximum temperature value for scaling. Defaults to 10.0.
            use_noise (bool): Whether to predict and include noise in the model.
                              Defaults to True.
            use_mlp (bool): Whether to use MLP in the PhysicsLayer. If False, uses SoftPlus.
                            Defaults to False.
        
        Raises:
            ValueError: If t_max <= t_min
        """
        super().__init__()
        if t_max <= t_min:
            raise ValueError("t_max must be greater than t_min")

        # Store temperature range for scaling
        self.t_min = float(t_min)
        self.t_max = float(t_max)
        
        # Flag to determine if noise prediction is used
        self.use_noise = use_noise

        # Encoder network to extract features from infrared image
        # Uses two convolutional layers with ReLU activations
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Head for predicting raw temperature values
        self.head_t = nn.Conv2d(base_channels, 1, kernel_size=1)
        
        # Head for predicting raw emissivity values
        self.head_eps = nn.Conv2d(base_channels, 1, kernel_size=1)
        
        # Head for predicting noise (optional)
        self.head_noise = nn.Conv2d(base_channels, 1, kernel_size=1)

        # Physics layer to enforce physical consistency
        # Implements the forward physical model: IR = ε * g(T) + noise
        self.physics = PhysicsLayer(use_mlp=use_mlp, channels=1)

    def _scale_temperature(self, raw: torch.Tensor) -> torch.Tensor:
        """
        Scale raw temperature predictions to the specified range [t_min, t_max].
        
        This method uses sigmoid activation to ensure the output is in [0, 1] range
        and then linearly maps it to [t_min, t_max].
        
        Args:
            raw (torch.Tensor): Raw temperature predictions of any range
            
        Returns:
            torch.Tensor: Scaled temperature values in [t_min, t_max] range
        """
        scaled = torch.sigmoid(raw)
        return scaled * (self.t_max - self.t_min) + self.t_min

    def forward(
        self, ir: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        Forward pass of the PhysDecompNet.
        
        Decomposes an infrared image into temperature, emissivity, and optionally noise,
        then synthesizes the reconstructed infrared image using the physics layer.
        
        Args:
            ir (torch.Tensor): Input infrared image of shape (B, C, H, W)
            
        Returns:
            Tuple containing:
            - temperature (torch.Tensor): Predicted temperature map (B, 1, H, W)
            - emissivity (torch.Tensor): Predicted emissivity map (B, 1, H, W), in [0, 1] range
            - noise (Optional[torch.Tensor]): Predicted noise map (B, 1, H, W) if use_noise=True, 
                                             otherwise None
            - ir_hat (torch.Tensor): Reconstructed infrared image (B, 1, H, W)
        """
        # Extract features from the input infrared image
        features = self.encoder(ir)
        
        # Predict raw temperature, emissivity, and noise values
        t_raw = self.head_t(features)          # Raw temperature predictions
        eps_raw = self.head_eps(features)      # Raw emissivity predictions
        noise = self.head_noise(features) if self.use_noise else None  # Optional noise prediction

        # Scale temperature to the specified range [t_min, t_max]
        temperature = self._scale_temperature(t_raw)
        
        # Apply sigmoid to ensure emissivity is in [0, 1] range
        emissivity = torch.sigmoid(eps_raw)
        
        # Use physics layer to synthesize reconstructed infrared image
        ir_hat = self.physics(temperature, emissivity, noise)

        return temperature, emissivity, noise, ir_hat
