from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class PhysicsLayer(nn.Module):
    """
    Differentiable physics layer for IR synthesis based on the Planck's law and 
    Stefan-Boltzmann law approximations.
    
    This layer implements the physical relationship between temperature, emissivity,
    and infrared radiation:
        IR = ε * g(T) + noise
        
    where:
    - IR is the synthesized infrared image
    - ε (emissivity) is in the range [0, 1]
    - T is the temperature
    - g(T) is a nonlinear transformation of temperature
    - noise is optional additive noise
    
    The function g(T) can be implemented as either:
    - SoftPlus function (default): A smooth approximation of ReLU that ensures positive values
    - MLP (Multi-Layer Perceptron): A learnable nonlinear transformation
    """

    def __init__(self, use_mlp: bool = False, channels: int = 1) -> None:
        """
        Initialize the PhysicsLayer.
        
        Args:
            use_mlp (bool): Whether to use MLP as the temperature transformation function.
                           If False, uses SoftPlus function. Defaults to False.
            channels (int): Number of input/output channels. Defaults to 1.
        """
        super().__init__()
        self.use_mlp = use_mlp
        
        if use_mlp:
            # MLP implementation for temperature transformation function g(T)
            # This provides more flexibility for learning complex temperature transformations
            self.g = nn.Sequential(
                nn.Conv2d(channels, 8, kernel_size=1),  # First conv layer to expand features
                nn.ReLU(inplace=True),                 # ReLU activation for non-linearity
                nn.Conv2d(8, channels, kernel_size=1), # Final conv layer to return to original channels
                nn.Softplus(),                         # SoftPlus to ensure positive output
            )
        else:
            # Simple SoftPlus implementation for temperature transformation function g(T)
            # SoftPlus ensures positive output and is differentiable everywhere
            self.g = nn.Softplus()

    def forward(
        self,
        temperature: torch.Tensor,
        emissivity: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Synthesize infrared image from temperature and emissivity.
        
        The forward pass implements the physical model:
            IR = ε * g(T) + noise
            
        where g(T) is the temperature transformation function.
        
        Args:
            temperature (torch.Tensor): Input temperature tensor of shape (B, C, H, W)
            emissivity (torch.Tensor): Input emissivity tensor of shape (B, C, H, W)
                                       Values should be in [0, 1] range
            noise (Optional[torch.Tensor]): Optional additive noise tensor of shape (B, C, H, W).
                                          If None, no noise is added. Defaults to None.
        
        Returns:
            torch.Tensor: Synthesized infrared image of shape (B, C, H, W)
        """
        # Apply the temperature transformation function g(T)
        # This models the relationship between temperature and radiated energy
        g_t = self.g(temperature)
        
        # Synthesize IR image using the physical model: IR = ε * g(T)
        # This implements the fundamental physics of infrared radiation
        ir_hat = emissivity * g_t
        
        # Add optional noise term if provided
        if noise is not None:
            ir_hat = ir_hat + noise
            
        return ir_hat
