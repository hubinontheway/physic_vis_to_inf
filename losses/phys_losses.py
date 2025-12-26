from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn
from torch.nn import functional as F


def recon_loss(ir_hat: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Reconstruction loss using L1 norm.
    
    This loss measures the pixel-wise difference between the reconstructed 
    infrared image (ir_hat) and the target image. It encourages the model 
    to generate images that closely match the input.
    
    Args:
        ir_hat (torch.Tensor): Reconstructed infrared image
        target (torch.Tensor): Target infrared image
        
    Returns:
        torch.Tensor: Scalar loss value representing the average absolute difference
    """
    return torch.mean(torch.abs(ir_hat - target))


def smoothness_loss(tensor: torch.Tensor) -> torch.Tensor:
    """
    Smoothness loss based on total variation (TV).
    
    This loss penalizes large spatial variations in the tensor, encouraging
    smoothness. It's particularly useful for temperature maps where we expect
    spatial continuity.
    
    The loss is calculated as the sum of absolute gradients in horizontal 
    and vertical directions.
    
    Args:
        tensor (torch.Tensor): Input tensor to compute smoothness for
        
    Returns:
        torch.Tensor: Scalar loss value representing the total variation
    """
    # Calculate vertical gradients: difference between adjacent rows
    dy = torch.abs(tensor[:, :, 1:, :] - tensor[:, :, :-1, :])
    # Calculate horizontal gradients: difference between adjacent columns
    dx = torch.abs(tensor[:, :, :, 1:] - tensor[:, :, :, :-1])
    # Return sum of average gradients in both directions
    return torch.mean(dx) + torch.mean(dy)


def emissivity_prior_loss(eps: torch.Tensor) -> torch.Tensor:
    """
    Emissivity prior loss combining smoothness and sparsity.
    
    This loss encourages the predicted emissivity to be piecewise smooth 
    (using total variation) while also promoting sparsity (using L1 norm).
    Both properties are physically meaningful for emissivity maps.
    
    Args:
        eps (torch.Tensor): Predicted emissivity tensor
        
    Returns:
        torch.Tensor: Combined smoothness and sparsity loss
    """
    # Total variation loss for smoothness
    tv = smoothness_loss(eps)
    # L1 loss for mild sparsity
    sparse = torch.mean(torch.abs(eps))
    # Combine both losses (with a weight of 0.1 for sparsity)
    return tv + 0.1 * sparse


def consistency_loss(
    t1: torch.Tensor,
    t2: torch.Tensor,
    eps1: torch.Tensor,
    eps2: torch.Tensor,
    n1: Optional[torch.Tensor] = None,
    n2: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Consistency loss for two-view training.
    
    This loss ensures that when the same image is processed through two 
    different augmented views, the predicted physical properties (temperature,
    emissivity, and optionally noise) remain consistent. This helps improve
    the robustness of the decomposition.
    
    Args:
        t1 (torch.Tensor): Temperature prediction from view 1
        t2 (torch.Tensor): Temperature prediction from view 2
        eps1 (torch.Tensor): Emissivity prediction from view 1
        eps2 (torch.Tensor): Emissivity prediction from view 2
        n1 (Optional[torch.Tensor]): Noise prediction from view 1 (if using noise)
        n2 (Optional[torch.Tensor]): Noise prediction from view 2 (if using noise)
        
    Returns:
        torch.Tensor: Scalar loss value representing the consistency between views
    """
    # Calculate consistency loss for temperature and emissivity
    loss = torch.mean(torch.abs(t1 - t2)) + torch.mean(torch.abs(eps1 - eps2))
    # Add consistency loss for noise if predictions are provided
    if n1 is not None and n2 is not None:
        loss = loss + torch.mean(torch.abs(n1 - n2))
    return loss


def corr_loss(tensor_a: torch.Tensor, tensor_b: torch.Tensor) -> torch.Tensor:
    """
    Correlation loss to minimize correlation between two tensors.
    
    This loss calculates the absolute correlation between two tensors and 
    encourages them to be uncorrelated. It's particularly useful for 
    decoupling temperature and emissivity, which should be independent
    physical properties.
    
    The correlation is calculated as Pearson correlation coefficient:
    corr = (a - mean(a)) · (b - mean(b)) / (||a - mean(a)|| * ||b - mean(b)||)
    
    Args:
        tensor_a (torch.Tensor): First tensor (e.g., temperature)
        tensor_b (torch.Tensor): Second tensor (e.g., emissivity)
        
    Returns:
        torch.Tensor: Scalar loss value representing the absolute correlation
    """
    # Flatten tensors to 2D (batch_size, features) for correlation calculation
    flat_a = tensor_a.flatten(start_dim=1)
    flat_b = tensor_b.flatten(start_dim=1)
    
    # Center the tensors by subtracting their means
    flat_a = flat_a - flat_a.mean(dim=1, keepdim=True)
    flat_b = flat_b - flat_b.mean(dim=1, keepdim=True)
    
    # Calculate the denominator for correlation (with small epsilon to avoid division by zero)
    denom = (
        torch.sqrt(torch.sum(flat_a ** 2, dim=1))
        * torch.sqrt(torch.sum(flat_b ** 2, dim=1))
        + 1e-6
    )
    
    # Calculate correlation: dot product normalized by the product of L2 norms
    corr = torch.sum(flat_a * flat_b, dim=1) / denom
    
    # Return mean of absolute correlations (to penalize both positive and negative correlations)
    return torch.mean(torch.abs(corr))


class PhysDecompLoss(nn.Module):
    """
    Combined reconstruction + consistency loss for two-view training of 
    physical decomposition networks.
    
    This loss function combines multiple components to train the network
    effectively:
    1. Reconstruction loss: Ensures the reconstructed image matches input
    2. Consistency loss: Ensures predictions are consistent across augmented views
    
    The two-view training approach helps improve the robustness and generalization
    of the physical decomposition.
    """

    def __init__(
        self,
        recon_weight: float = 1.0,
        consistency_weight: float = 0.1,
        recon_type: str = "l1",
    ) -> None:
        """
        Initialize the PhysDecompLoss module.
        
        Args:
            recon_weight (float): Weight for reconstruction loss. Defaults to 1.0.
            consistency_weight (float): Weight for consistency loss. Defaults to 0.1.
            recon_type (str): Type of reconstruction loss ('l1' or 'l2'). Defaults to 'l1'.
        
        Raises:
            ValueError: If recon_type is not 'l1' or 'l2'
        """
        super().__init__()
        self.recon_weight = float(recon_weight)
        self.consistency_weight = float(consistency_weight)
        if recon_type not in {"l1", "l2"}:
            raise ValueError("recon_type must be 'l1' or 'l2'")
        self.recon_type = recon_type

    def _recon(self, ir_hat: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute reconstruction loss using either L1 or L2 norm.
        
        Args:
            ir_hat (torch.Tensor): Reconstructed image
            target (torch.Tensor): Target image
            
        Returns:
            torch.Tensor: Reconstruction loss value
        """
        if self.recon_type == "l2":
            # Use MSE loss (L2 norm squared)
            return F.mse_loss(ir_hat, target)
        # Use the custom L1 reconstruction loss
        return recon_loss(ir_hat, target)

    def forward(
        self,
        ir_hat1: torch.Tensor,
        ir_hat2: torch.Tensor,
        ir1: torch.Tensor,
        ir2: torch.Tensor,
        t1: torch.Tensor,
        t2: torch.Tensor,
        eps1: torch.Tensor,
        eps2: torch.Tensor,
        n1: Optional[torch.Tensor] = None,
        n2: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass to compute the combined loss.
        
        Args:
            ir_hat1 (torch.Tensor): Reconstructed image from view 1
            ir_hat2 (torch.Tensor): Reconstructed image from view 2
            ir1 (torch.Tensor): Target image for view 1
            ir2 (torch.Tensor): Target image for view 2
            t1 (torch.Tensor): Temperature prediction from view 1
            t2 (torch.Tensor): Temperature prediction from view 2
            eps1 (torch.Tensor): Emissivity prediction from view 1
            eps2 (torch.Tensor): Emissivity prediction from view 2
            n1 (Optional[torch.Tensor]): Noise prediction from view 1
            n2 (Optional[torch.Tensor]): Noise prediction from view 2
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary containing total loss and individual components
        """
        # Calculate reconstruction loss for both views
        recon = self._recon(ir_hat1, ir1) + self._recon(ir_hat2, ir2)
        
        # Calculate consistency loss between the two views
        consistency = consistency_loss(t1, t2, eps1, eps2, n1, n2)
        
        # Combine losses with respective weights
        total = self.recon_weight * recon + self.consistency_weight * consistency
        
        # Return all loss components for monitoring
        return {"total": total, "recon": recon, "consistency": consistency}
