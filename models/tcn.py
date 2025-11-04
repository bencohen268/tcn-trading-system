"""
Temporal Convolutional Network (TCN) implementation.

Based on: "An Empirical Evaluation of Generic Convolutional and Recurrent Networks
for Sequence Modeling" (Bai et al., 2018)

Key features:
- Causal convolutions (no future leakage)
- Dilated convolutions (exponentially growing receptive field)
- Residual connections (stable gradients)
- Weight normalization (training stability)
"""

from typing import List
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class TemporalBlock(nn.Module):
    """
    Single temporal block in the TCN.
    
    Structure:
        Input → Conv1d → ReLU → Dropout → Conv1d → ReLU → Dropout → + → Output
          ↓                                                            ↑
          └─────────────────── Residual (1x1 Conv if needed) ─────────┘
    
    Each block has:
    - Two dilated causal convolutions
    - ReLU activations
    - Dropout regularization
    - Residual connection
    """
    
    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.1,
    ):
        """
        Initialize temporal block.
        
        Args:
            n_inputs: Number of input channels
            n_outputs: Number of output channels
            kernel_size: Size of convolutional kernel
            dilation: Dilation factor
            dropout: Dropout probability
        """
        super(TemporalBlock, self).__init__()
        
        # Causal padding: pad only on the left (past)
        # For dilation d and kernel k, we need (k-1)*d padding
        self.padding = (kernel_size - 1) * dilation
        
        # First convolution
        self.conv1 = weight_norm(nn.Conv1d(
            n_inputs,
            n_outputs,
            kernel_size,
            stride=1,
            padding=0,  # We'll handle padding manually
            dilation=dilation,
        ))
        
        # Second convolution
        self.conv2 = weight_norm(nn.Conv1d(
            n_outputs,
            n_outputs,
            kernel_size,
            stride=1,
            padding=0,
            dilation=dilation,
        ))
        
        # Activations and regularization
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # Residual connection (1x1 conv if dimensions don't match)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        
        # Final activation after residual
        self.relu_final = nn.ReLU()
        
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights."""
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, channels, sequence_length)
            
        Returns:
            Output tensor (batch, channels, sequence_length)
        """
        # Apply causal padding (pad left side only)
        x_padded = nn.functional.pad(x, (self.padding, 0))
        
        # First conv block
        out = self.conv1(x_padded)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        # Apply causal padding again
        out_padded = nn.functional.pad(out, (self.padding, 0))
        
        # Second conv block
        out = self.conv2(out_padded)
        out = self.relu2(out)
        out = self.dropout2(out)
        
        # Residual connection
        res = x if self.downsample is None else self.downsample(x)
        
        # Add residual and apply final activation
        out = self.relu_final(out + res)
        
        return out


class TemporalConvNet(nn.Module):
    """
    Temporal Convolutional Network.
    
    Stacks multiple TemporalBlocks with increasing dilation.
    Each block doubles the dilation: [1, 2, 4, 8, ...]
    
    This creates an exponentially growing receptive field.
    """
    
    def __init__(
        self,
        num_inputs: int,
        num_channels: List[int],
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        """
        Initialize TCN.
        
        Args:
            num_inputs: Number of input features
            num_channels: List of channel sizes per block (e.g., [32, 32, 64])
            kernel_size: Kernel size for all blocks
            dropout: Dropout probability
        """
        super(TemporalConvNet, self).__init__()
        
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation = 2 ** i  # Exponentially increasing dilation
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            
            layers.append(TemporalBlock(
                n_inputs=in_channels,
                n_outputs=out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                dropout=dropout,
            ))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, channels, sequence_length)
            
        Returns:
            Output tensor (batch, channels, sequence_length)
        """
        return self.network(x)


class TCNClassifier(nn.Module):
    """
    Complete TCN model for binary classification.
    
    Architecture:
        Input (batch, seq_len, features)
        → Transpose to (batch, features, seq_len)
        → TCN
        → Take last timestep
        → Linear layer
        → Output logit (batch, 1)
    """
    
    def __init__(
        self,
        num_inputs: int,
        num_channels: List[int],
        kernel_size: int = 3,
        dropout: float = 0.1,
        output_dim: int = 1,
    ):
        """
        Initialize TCN classifier.
        
        Args:
            num_inputs: Number of input features
            num_channels: List of channel sizes per TCN block
            kernel_size: Kernel size
            dropout: Dropout probability
            output_dim: Output dimension (1 for binary classification)
        """
        super(TCNClassifier, self).__init__()
        
        self.tcn = TemporalConvNet(
            num_inputs=num_inputs,
            num_channels=num_channels,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        
        # Output layer
        self.linear = nn.Linear(num_channels[-1], output_dim)
        
        self.num_inputs = num_inputs
        self.num_channels = num_channels
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, seq_len, features)
            
        Returns:
            Output logits (batch, output_dim)
        """
        # Transpose to (batch, features, seq_len) for Conv1d
        x = x.transpose(1, 2)
        
        # Pass through TCN
        y = self.tcn(x)  # (batch, channels, seq_len)
        
        # Take last timestep
        y = y[:, :, -1]  # (batch, channels)
        
        # Linear output layer
        out = self.linear(y)  # (batch, output_dim)
        
        return out
    
    def get_receptive_field(self, kernel_size: int) -> int:
        """
        Calculate the receptive field of the TCN.
        
        Args:
            kernel_size: Kernel size used in the network
            
        Returns:
            Receptive field size
        """
        num_levels = len(self.num_channels)
        receptive_field = 1
        
        for i in range(num_levels):
            dilation = 2 ** i
            receptive_field += 2 * (kernel_size - 1) * dilation
        
        return receptive_field
    
    def count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_tcn_from_config(config: dict) -> TCNClassifier:
    """
    Create TCN model from configuration dictionary.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        TCNClassifier instance
    """
    model_config = config['model']
    
    model = TCNClassifier(
        num_inputs=model_config['num_inputs'],
        num_channels=model_config['num_channels'],
        kernel_size=model_config['kernel_size'],
        dropout=model_config['dropout'],
        output_dim=model_config['output_dim'],
    )
    
    # Print model summary
    receptive_field = model.get_receptive_field(model_config['kernel_size'])
    num_params = model.count_parameters()
    
    print(f"\nTCN Model Summary:")
    print(f"  Input features: {model_config['num_inputs']}")
    print(f"  Channels: {model_config['num_channels']}")
    print(f"  Kernel size: {model_config['kernel_size']}")
    print(f"  Dropout: {model_config['dropout']}")
    print(f"  Receptive field: {receptive_field}")
    print(f"  Total parameters: {num_params:,}")
    
    return model

