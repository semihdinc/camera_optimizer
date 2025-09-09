"""
Adjustments class for camera parameter optimization.

This module contains the Adjustments class that manages all learnable parameters
for correcting noisy camera intrinsics and extrinsics.
"""

import torch
import torch.nn as nn
from typing import List
from cameras.camera import Camera

class CameraCorrector(nn.Module):
    """   
    Learn intrinsics and extrinsics deltas to correct the noisy camera parameters.
    """  
    def __init__(self, noisy_cameras: List[Camera], device='cpu'):
        super().__init__()
        self.device = device
        self.num_cameras = len(noisy_cameras)
        
        # Store original noisy parameters
        self.noisy_cameras = noisy_cameras
        
        # Intrinsic deltas: delta_fx, delta_fy, delta_cx, delta_cy for each camera
        self.intrinsic_deltas = nn.Parameter(torch.zeros(self.num_cameras, 4, device=self.device, dtype=torch.double))
        
        # Extrinsic deltas: delta_R (rotation, 3 angles per camera) and delta_t (translation) for each camera
        self.rotation_deltas = nn.Parameter(torch.zeros(self.num_cameras, 3, device=self.device, dtype=torch.double))
        self.translation_deltas = nn.Parameter(torch.zeros(self.num_cameras, 3, device=self.device, dtype=torch.double))

    def forward(self, X_world: torch.Tensor, camera_indices: torch.Tensor) -> torch.Tensor:
        """Project a 3D point using corrected camera parameters."""
        projected_pixels = []
        for id, cam in enumerate(camera_indices):
            
            # Get deltas for this camera
            intrinsic_deltas = self.intrinsic_deltas[cam]
            rotation_deltas = self.rotation_deltas[cam]
            translation_deltas = self.translation_deltas[cam]
            
            # Use the camera's projection method with corrections
            camera = self.noisy_cameras[cam]
            px = camera.project_to_image_with_corrections(
                X_world[id], intrinsic_deltas, rotation_deltas, translation_deltas
            )
            projected_pixels.append(px)
        
        return torch.stack(projected_pixels, dim=0)

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