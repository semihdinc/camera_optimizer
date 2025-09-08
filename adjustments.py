"""
Adjustments class for camera parameter optimization.

This module contains the Adjustments class that manages all learnable parameters
for correcting noisy camera intrinsics and extrinsics.
"""

import torch
import torch.nn as nn
from typing import List, Tuple
import numpy as np


class Adjustments(nn.Module):
    """
    Learnable adjustment parameters for camera optimization.
    
    This class manages all learnable parameters needed to correct noisy camera
    parameters including intrinsics (fx, fy, cx, cy) and extrinsics (rotation, translation).
    """
    
    def __init__(self, num_cameras: int, device: str = 'cpu'):
        """
        Initialize adjustment parameters for all cameras.
        
        Args:
            num_cameras (int): Number of cameras to optimize
            device (str): Device to store parameters on ('cpu' or 'cuda')
        """
        super(Adjustments, self).__init__()
        
        self.num_cameras = num_cameras
        self.device = device
        
        # Intrinsic adjustments: delta_fx, delta_fy, delta_cx, delta_cy for each camera
        self.intrinsic_deltas = nn.Parameter(torch.zeros(num_cameras, 4, device=device))
        
        # Extrinsic adjustments: delta_R (rotation) and delta_t (translation) for each camera
        # For rotation, we use axis-angle representation (3 params per camera)
        self.rotation_deltas = nn.Parameter(torch.zeros(num_cameras, 3, device=device))
        self.translation_deltas = nn.Parameter(torch.zeros(num_cameras, 3, device=device))
        
    def get_intrinsic_adjustments(self, camera_idx: int) -> torch.Tensor:
        """Get intrinsic adjustments for a specific camera."""
        return self.intrinsic_deltas[camera_idx]
    
    def get_extrinsic_adjustments(self, camera_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get extrinsic adjustments for a specific camera."""
        return self.rotation_deltas[camera_idx], self.translation_deltas[camera_idx]

    def get_total_parameters(self) -> int:
        """Get total number of parameters."""
        return self.intrinsic_deltas.numel() + self.rotation_deltas.numel() + self.translation_deltas.numel()
    
    def get_adjustment_summary(self) -> dict:
        """Get summary statistics of adjustment parameters."""
        with torch.no_grad():
            return {
                'intrinsic_deltas': {
                    'mean': self.intrinsic_deltas.mean().item(),
                    'std': self.intrinsic_deltas.std().item()
                },
                'rotation_deltas': {
                    'mean': self.rotation_deltas.mean().item(),
                    'std': self.rotation_deltas.std().item()
                },
                'translation_deltas': {
                    'mean': self.translation_deltas.mean().item(),
                    'std': self.translation_deltas.std().item()
                }
            }