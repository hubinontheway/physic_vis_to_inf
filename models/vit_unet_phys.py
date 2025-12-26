from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
from torch import nn

from .physics_layer import PhysicsLayer

try:
    import timm
except ImportError:  # pragma: no cover - optional dependency
    timm = None


class PatchEmbedding(nn.Module):
    """
    Image to patch embedding with optional position encoding.
    
    This module converts an input image into a sequence of patch embeddings, 
    which can be processed by transformer architectures.
    """

    def __init__(self, in_channels: int, embed_dim: int, patch_size: int) -> None:
        """
        Initialize the PatchEmbedding module.
        
        Args:
            in_channels (int): Number of input channels
            embed_dim (int): Dimension of the patch embeddings
            patch_size (int): Size of each square patch
        """
        super().__init__()
        self.patch_size = int(patch_size)
        # Convolutional layer to extract patches and embed them
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.pos_embed: Optional[nn.Parameter] = None

    def _init_pos_embed(self, num_patches: int, device: torch.device) -> None:
        """
        Initialize positional embeddings if not already initialized or if the number of patches changed.
        
        Args:
            num_patches (int): Number of patches in the image
            device (torch.device): Device to initialize the embedding on
        """
        if self.pos_embed is None or self.pos_embed.shape[1] != num_patches:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, self.proj.out_channels, device=device)
            )
            # Initialize positional embeddings with truncated normal distribution
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Forward pass to convert image to patch embeddings.
        
        Args:
            x (torch.Tensor): Input image of shape (B, C, H, W)
            
        Returns:
            Tuple containing:
            - tokens (torch.Tensor): Patch embeddings of shape (B, num_patches, embed_dim)
            - (h, w) (Tuple[int, int]): Height and width of the patch grid
        """
        # Project image patches to embedding dimension
        x = self.proj(x)
        h, w = x.shape[-2:]
        # Reshape to (B, num_patches, embed_dim) format for transformer
        tokens = x.flatten(2).transpose(1, 2)
        # Initialize and add positional embeddings
        self._init_pos_embed(tokens.shape[1], tokens.device)
        tokens = tokens + self.pos_embed
        return tokens, (h, w)


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) used in Vision Transformers.
    
    This is a simple feed-forward network with GELU activation and dropout.
    """

    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        """
        Initialize the MLP module.
        
        Args:
            dim (int): Input and output dimension
            mlp_ratio (float): Hidden dimension will be dim * mlp_ratio. Defaults to 4.0.
            dropout (float): Dropout rate. Defaults to 0.0.
        """
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)  # First linear layer
        self.act = nn.GELU()  # GELU activation function
        self.fc2 = nn.Linear(hidden_dim, dim)  # Second linear layer
        self.drop = nn.Dropout(dropout)  # Dropout layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MLP.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, N, dim) or (B, dim)
            
        Returns:
            torch.Tensor: Output tensor of the same shape as input
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """
    A single transformer block with multi-head self-attention and MLP.
    
    This block implements the standard transformer architecture with layer normalization,
    residual connections, and feed-forward networks.
    """

    def __init__(
        self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0
    ) -> None:
        """
        Initialize the TransformerBlock.
        
        Args:
            dim (int): Dimension of the input features
            num_heads (int): Number of attention heads
            mlp_ratio (float): Ratio of hidden dimension in MLP. Defaults to 4.0.
            dropout (float): Dropout rate. Defaults to 0.0.
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)  # First layer normalization
        # Multihead attention layer
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)  # Second layer normalization
        # MLP for feed-forward processing
        self.mlp = MLP(dim, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the transformer block.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, N, dim)
            
        Returns:
            torch.Tensor: Output tensor of shape (B, N, dim)
        """
        # Self-attention with residual connection
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out
        # MLP with residual connection
        x = x + self.mlp(self.norm2(x))
        return x


class ViTEncoder(nn.Module):
    """
    Vision Transformer encoder that outputs a spatial feature map.
    
    This encoder can either use a custom implementation or leverage the timm library
    for pre-implemented ViT models. It processes input images through transformer blocks
    to extract high-level features while maintaining spatial structure.
    """

    def __init__(
        self,
        in_channels: int = 1,
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        patch_size: int = 16,
        image_size: Optional[int] = None,
        dropout: float = 0.0,
        use_timm: bool = False,
        timm_model: str = "vit_base_patch16_224",
        timm_pretrained: bool = False,
    ) -> None:
        """
        Initialize the ViTEncoder.
        
        Args:
            in_channels (int): Number of input channels. Defaults to 1.
            embed_dim (int): Embedding dimension for patches. Defaults to 256.
            depth (int): Number of transformer blocks. Defaults to 6.
            num_heads (int): Number of attention heads. Defaults to 8.
            patch_size (int): Size of image patches. Defaults to 16.
            image_size (Optional[int]): Input image size (for timm models). Defaults to None.
            dropout (float): Dropout rate. Defaults to 0.0.
            use_timm (bool): Whether to use timm library for ViT. Defaults to False.
            timm_model (str): Name of timm model to use. Defaults to "vit_base_patch16_224".
            timm_pretrained (bool): Whether to use pretrained timm weights. Defaults to False.
        
        Raises:
            ImportError: If use_timm=True but timm is not installed
        """
        super().__init__()
        self.use_timm = use_timm
        self.patch_size = int(patch_size)
        self.embed_dim = int(embed_dim)

        if use_timm:
            if timm is None:
                raise ImportError("timm is required for use_timm=True")
            # Prepare arguments for timm model creation
            timm_kwargs = {
                "pretrained": timm_pretrained,
                "num_classes": 0,  # No classification head
                "in_chans": in_channels,
            }
            if image_size is not None:
                timm_kwargs["img_size"] = int(image_size)
            self.timm_model = timm.create_model(timm_model, **timm_kwargs)
            self.embed_dim = self.timm_model.num_features
            patch = self.timm_model.patch_embed.patch_size
            self.patch_size = int(patch[0] if isinstance(patch, tuple) else patch)
        else:
            # Custom ViT implementation
            self.patch_embed = PatchEmbedding(in_channels, embed_dim, patch_size)
            # Create transformer blocks
            self.blocks = nn.ModuleList(
                [
                    TransformerBlock(
                        dim=embed_dim,
                        num_heads=num_heads,
                        mlp_ratio=4.0,
                        dropout=dropout,
                    )
                    for _ in range(depth)
                ]
            )
            self.norm = nn.LayerNorm(embed_dim)  # Final layer normalization

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ViT encoder.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)
            
        Returns:
            torch.Tensor: Output feature map of shape (B, embed_dim, H_out, W_out)
        """
        if self.use_timm:
            # Use timm model to extract features
            tokens = self.timm_model.forward_features(x)
            if tokens.dim() == 3:
                # Handle case where tokens are in (B, num_patches, embed_dim) format
                h = x.shape[2] // self.patch_size
                w = x.shape[3] // self.patch_size
                num_patches = h * w
                if tokens.shape[1] == num_patches + 1:
                    # Remove class token if present
                    tokens = tokens[:, 1:, :]
                elif tokens.shape[1] != num_patches:
                    raise ValueError("Unexpected timm token count for input size")
                # Reshape to (B, embed_dim, h, w) format
                return tokens.transpose(1, 2).reshape(x.shape[0], -1, h, w)
            if tokens.dim() == 4:
                # Return in (B, embed_dim, h, w) format directly
                return tokens
            raise ValueError("Unexpected timm ViT feature shape")

        # Validate input size is divisible by patch size
        if x.shape[2] % self.patch_size != 0 or x.shape[3] % self.patch_size != 0:
            raise ValueError("Input size must be divisible by patch_size")
        
        # Extract patch embeddings
        tokens, (h, w) = self.patch_embed(x)
        # Process through transformer blocks
        for block in self.blocks:
            tokens = block(tokens)
        # Apply final normalization
        tokens = self.norm(tokens)
        # Reshape back to spatial format (B, embed_dim, h, w)
        return tokens.transpose(1, 2).reshape(x.shape[0], -1, h, w)


class ConvStem(nn.Module):
    """
    CNN pyramid to provide multi-scale skip connections aligned with ViT patch grid.
    
    This module creates a CNN-based feature pyramid that provides skip connections
    to the U-Net decoder. It's designed to align with the patch structure of ViT
    to enable effective feature fusion.
    """

    def __init__(self, in_channels: int, base_channels: int, num_stages: int) -> None:
        """
        Initialize the ConvStem.
        
        Args:
            in_channels (int): Number of input channels
            base_channels (int): Base number of channels (doubled at each stage)
            num_stages (int): Number of downsampling stages
        """
        super().__init__()
        self.stages = nn.ModuleList()
        # First stage: no downsampling
        self.stages.append(
            nn.Sequential(
                nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )
        )
        in_ch = base_channels
        # Subsequent stages: downsampling by factor of 2 with strided conv
        for idx in range(1, num_stages + 1):
            out_ch = base_channels * (2**idx)
            self.stages.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),  # Stride 2 for downsampling
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                )
            )
            in_ch = out_ch

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass through the CNN pyramid.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)
            
        Returns:
            List[torch.Tensor]: List of feature maps at different scales
        """
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features


class UpBlock(nn.Module):
    """
    U-Net upsampling block that combines features from encoder and decoder paths.
    
    This block performs upsampling, concatenates skip connections, and applies
    convolutional operations to refine features.
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        """
        Initialize the UpBlock.
        
        Args:
            in_channels (int): Number of input channels from lower resolution
            skip_channels (int): Number of channels from skip connection
            out_channels (int): Number of output channels
        """
        super().__init__()
        # Upsample the input features
        self.up = nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False)
        # Convolutional layers after concatenation
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the UpBlock.
        
        Args:
            x (torch.Tensor): Input features from lower resolution (B, in_channels, H, W)
            skip (torch.Tensor): Skip connection features (B, skip_channels, 2*H, 2*W)
            
        Returns:
            torch.Tensor: Refined features (B, out_channels, 2*H, 2*W)
        """
        x = self.up(x)  # Upsample
        x = torch.cat([x, skip], dim=1)  # Concatenate with skip connection
        return self.conv(x)  # Apply convolutions


class UNetDecoder(nn.Module):
    """
    U-Net decoder with skip connections for feature refinement.
    
    This decoder takes features from the bottleneck and skip connections from
    the encoder path, progressively upsampling and refining features.
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: List[int],
        out_channels: int,
    ) -> None:
        """
        Initialize the UNetDecoder.
        
        Args:
            in_channels (int): Number of channels from bottleneck
            skip_channels (List[int]): List of channel counts for skip connections (in decoding order)
            out_channels (int): Number of output channels for the final prediction heads
        """
        super().__init__()
        # Bottleneck processing before upsampling
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Create upsampling blocks
        self.up_blocks = nn.ModuleList()
        current_channels = out_channels
        for skip_ch in skip_channels:
            # Calculate output channels for this block
            next_channels = max(out_channels // 2, 16)
            # Create UpBlock to process features and skip connection
            self.up_blocks.append(UpBlock(current_channels, skip_ch, next_channels))
            current_channels = next_channels
            out_channels = next_channels

        self.out_channels = current_channels  # Store final output channel count

    def forward(self, x: torch.Tensor, skips: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass of the U-Net decoder.
        
        Args:
            x (torch.Tensor): Bottleneck features (B, in_channels, H, W)
            skips (List[torch.Tensor]): Skip connection features (in decoding order)
            
        Returns:
            torch.Tensor: Refined features (B, self.out_channels, H_up, W_up)
        """
        x = self.bottleneck(x)  # Process bottleneck features
        # Process through upsampling blocks with skip connections
        for block, skip in zip(self.up_blocks, skips):
            x = block(x, skip)
        return x


class PhysDecompViTUNet(nn.Module):
    """
    ViT encoder + U-Net decoder for physical property decoupling.
    
    This architecture combines the global context capture capability of Vision 
    Transformers (ViT) with the spatial precision of U-Net. The ViT encoder 
    captures high-level semantic features, while the U-Net decoder with skip 
    connections preserves spatial details for accurate physical property estimation.
    
    The model decomposes an infrared image into temperature and emissivity maps
    using a physics-inspired approach that ensures physical consistency.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        vit_embed_dim: int = 256,
        vit_depth: int = 6,
        vit_heads: int = 8,
        patch_size: int = 16,
        image_size: Optional[int] = None,
        t_min: float = 1.0,
        t_max: float = 10.0,
        use_noise: bool = True,
        use_mlp: bool = False,
        use_timm: bool = False,
        timm_model: str = "vit_base_patch16_224",
        timm_pretrained: bool = False,
    ) -> None:
        """
        Initialize the PhysDecompViTUNet.
        
        Args:
            in_channels (int): Number of input channels. Defaults to 1.
            base_channels (int): Base number of channels for CNN stem. Defaults to 32.
            vit_embed_dim (int): Embedding dimension for ViT encoder. Defaults to 256.
            vit_depth (int): Depth of ViT encoder (number of transformer blocks). Defaults to 6.
            vit_heads (int): Number of attention heads in ViT. Defaults to 8.
            patch_size (int): Patch size for ViT (must be power of 2). Defaults to 16.
            image_size (Optional[int]): Input image size (for timm models). Defaults to None.
            t_min (float): Minimum temperature value for scaling. Defaults to 1.0.
            t_max (float): Maximum temperature value for scaling. Defaults to 10.0.
            use_noise (bool): Whether to predict and include noise. Defaults to True.
            use_mlp (bool): Whether to use MLP in PhysicsLayer. Defaults to False.
            use_timm (bool): Whether to use timm library for ViT. Defaults to False.
            timm_model (str): Name of timm model to use. Defaults to "vit_base_patch16_224".
            timm_pretrained (bool): Whether to use pretrained timm weights. Defaults to False.
        
        Raises:
            ValueError: If t_max <= t_min or patch_size is not a power of 2
        """
        super().__init__()
        if t_max <= t_min:
            raise ValueError("t_max must be greater than t_min")
        if patch_size <= 0 or patch_size & (patch_size - 1) != 0:
            raise ValueError("patch_size must be a power of 2")

        # Store temperature range and noise flag
        self.t_min = float(t_min)
        self.t_max = float(t_max)
        self.use_noise = use_noise
        # Calculate number of stages based on patch size (for CNN pyramid)
        num_stages = int(math.log2(patch_size))

        # CNN pyramid for multi-scale feature extraction and skip connections
        self.stem = ConvStem(in_channels, base_channels, num_stages)
        
        # ViT encoder for global feature extraction
        self.encoder = ViTEncoder(
            in_channels=in_channels,
            embed_dim=vit_embed_dim,
            depth=vit_depth,
            num_heads=vit_heads,
            patch_size=patch_size,
            image_size=image_size,
            use_timm=use_timm,
            timm_model=timm_model,
            timm_pretrained=timm_pretrained,
        )

        # Calculate channel sizes for different stages
        stem_channels = [base_channels * (2**idx) for idx in range(num_stages + 1)]
        # Bottleneck combines ViT features and the last stem feature map
        bottleneck_channels = stem_channels[-1] + self.encoder.embed_dim
        # Skip channels in reverse order for decoder
        skip_channels = list(reversed(stem_channels[:-1]))
        decoder_out = max(base_channels * 4, 32)
        
        # U-Net decoder with skip connections
        self.decoder = UNetDecoder(
            in_channels=bottleneck_channels,
            skip_channels=skip_channels,
            out_channels=decoder_out,
        )

        # Prediction heads for temperature, emissivity, and noise
        self.head_t = nn.Conv2d(self.decoder.out_channels, 1, kernel_size=1)  # Temperature head
        self.head_eps = nn.Conv2d(self.decoder.out_channels, 1, kernel_size=1)  # Emissivity head
        self.head_noise = nn.Conv2d(self.decoder.out_channels, 1, kernel_size=1)  # Noise head
        
        # Physics layer to enforce physical consistency
        self.physics = PhysicsLayer(use_mlp=use_mlp, channels=1)

    def _scale_temperature(self, raw: torch.Tensor) -> torch.Tensor:
        """
        Scale raw temperature predictions to the specified range [t_min, t_max].
        
        Args:
            raw (torch.Tensor): Raw temperature predictions
            
        Returns:
            torch.Tensor: Scaled temperature values in [t_min, t_max] range
        """
        scaled = torch.sigmoid(raw)
        return scaled * (self.t_max - self.t_min) + self.t_min

    def forward(
        self, ir: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        Forward pass of the PhysDecompViTUNet.
        
        Decomposes an infrared image into temperature and emissivity maps while
        ensuring physical consistency through the physics layer.
        
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
        # Extract multi-scale features using CNN pyramid
        skips = self.stem(ir)
        
        # Extract global features using ViT encoder
        vit_feat = self.encoder(ir)
        
        # Validate that ViT and stem feature maps have matching spatial dimensions
        if vit_feat.shape[-2:] != skips[-1].shape[-2:]:
            raise ValueError("ViT feature map resolution must match stem bottleneck")
        
        # Combine ViT features and the last stem feature map in bottleneck
        bottleneck = torch.cat([vit_feat, skips[-1]], dim=1)

        # Prepare skip connections for decoder (in reverse order)
        decoder_skips = list(reversed(skips[:-1]))
        # Process through U-Net decoder
        features = self.decoder(bottleneck, decoder_skips)

        # Generate raw predictions from decoder features
        t_raw = self.head_t(features)          # Raw temperature predictions
        eps_raw = self.head_eps(features)      # Raw emissivity predictions
        noise = self.head_noise(features) if self.use_noise else None  # Optional noise prediction

        # Apply scaling and activation functions
        temperature = self._scale_temperature(t_raw)  # Scale temperature to [t_min, t_max]
        emissivity = torch.sigmoid(eps_raw)           # Apply sigmoid to ensure [0, 1] range for emissivity
        
        # Use physics layer to synthesize reconstructed infrared image
        ir_hat = self.physics(temperature, emissivity, noise)
        
        return temperature, emissivity, noise, ir_hat
