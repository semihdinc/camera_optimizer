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
        self.intrinsic_deltas = nn.Parameter(
            torch.zeros(num_cameras, 4, device=device)
        )
        
        # Extrinsic adjustments: delta_R (rotation) and delta_t (translation) for each camera
        # For rotation, we use axis-angle representation (3 params per camera)
        self.rotation_deltas = nn.Parameter(
            torch.zeros(num_cameras, 3, device=device)
        )
        self.translation_deltas = nn.Parameter(
            torch.zeros(num_cameras, 3, device=device)
        )
        
        # Optional: Learnable distortion parameters (radial and tangential)
        # Uncomment if you want to also optimize lens distortion
        # self.distortion_deltas = nn.Parameter(
        #     torch.zeros(num_cameras, 5, device=device)  # k1, k2, p1, p2, k3
        # )
        
    def get_intrinsic_adjustments(self, camera_idx: int) -> torch.Tensor:
        """
        Get intrinsic adjustments for a specific camera.
        
        Args:
            camera_idx (int): Index of the camera
            
        Returns:
            torch.Tensor: Intrinsic adjustments [delta_fx, delta_fy, delta_cx, delta_cy]
        """
        return self.intrinsic_deltas[camera_idx]
    
    def get_extrinsic_adjustments(self, camera_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get extrinsic adjustments for a specific camera.
        
        Args:
            camera_idx (int): Index of the camera
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (rotation_delta, translation_delta)
        """
        return self.rotation_deltas[camera_idx], self.translation_deltas[camera_idx]
    
    def get_all_intrinsic_adjustments(self) -> torch.Tensor:
        """Get intrinsic adjustments for all cameras."""
        return self.intrinsic_deltas
    
    def get_all_extrinsic_adjustments(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get extrinsic adjustments for all cameras."""
        return self.rotation_deltas, self.translation_deltas
    
    def apply_intrinsic_adjustments(self, camera_idx: int, original_intrinsics: dict) -> torch.Tensor:
        """
        Apply learned adjustments to original intrinsic parameters.
        
        Args:
            camera_idx (int): Index of the camera
            original_intrinsics (dict): Original intrinsic parameters {'fx': ..., 'fy': ..., 'cx': ..., 'cy': ...}
            
        Returns:
            torch.Tensor: Corrected intrinsic matrix K (3x3)
        """
        # Get adjustments
        delta = self.get_intrinsic_adjustments(camera_idx)
        
        # Apply adjustments to original parameters
        fx_corrected = original_intrinsics['fx'] + delta[0]
        fy_corrected = original_intrinsics['fy'] + delta[1]
        cx_corrected = original_intrinsics['cx'] + delta[2]
        cy_corrected = original_intrinsics['cy'] + delta[3]
        
        # Build corrected intrinsic matrix
        K = torch.zeros(3, 3, device=self.device)
        K[0, 0] = fx_corrected
        K[1, 1] = fy_corrected
        K[0, 2] = cx_corrected
        K[1, 2] = cy_corrected
        K[2, 2] = 1.0
        
        return K
    
    def apply_extrinsic_adjustments(self, camera_idx: int, original_R: torch.Tensor, original_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply learned adjustments to original extrinsic parameters.
        
        Args:
            camera_idx (int): Index of the camera
            original_R (torch.Tensor): Original rotation matrix (3x3)
            original_t (torch.Tensor): Original translation vector (3,)
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (corrected_R, corrected_t)
        """
        # Get adjustments
        delta_R, delta_t = self.get_extrinsic_adjustments(camera_idx)
        
        # Convert rotation delta from axis-angle to rotation matrix
        delta_R_matrix = self.axis_angle_to_rotation_matrix(delta_R)
        
        # Apply corrections
        R_corrected = delta_R_matrix @ original_R
        t_corrected = original_t + delta_t
        
        return R_corrected, t_corrected
    
    def axis_angle_to_rotation_matrix(self, axis_angle: torch.Tensor) -> torch.Tensor:
        """
        Convert axis-angle representation to rotation matrix using Rodrigues' formula.
        
        Args:
            axis_angle (torch.Tensor): Axis-angle representation (3,)
            
        Returns:
            torch.Tensor: Rotation matrix (3x3)
        """
        angle = torch.norm(axis_angle)
        if angle < 1e-8:
            return torch.eye(3, device=self.device)
        
        axis = axis_angle / angle
        cos_angle = torch.cos(angle)
        sin_angle = torch.sin(angle)
        
        # Rodrigues' formula
        K = torch.tensor([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ], device=self.device)
        
        R = torch.eye(3, device=self.device) + sin_angle * K + (1 - cos_angle) * (K @ K)
        return R
    
    def get_total_parameters(self) -> int:
        """Get total number of learnable parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def get_parameters_per_camera(self) -> int:
        """Get number of learnable parameters per camera."""
        return self.get_total_parameters() // self.num_cameras
    
    def reset_parameters(self, std: float = 0.01):
        """
        Reset all parameters to small random values.
        
        Args:
            std (float): Standard deviation for random initialization
        """
        with torch.no_grad():
            self.intrinsic_deltas.normal_(0, std)
            self.rotation_deltas.normal_(0, std)
            self.translation_deltas.normal_(0, std)
    
    def get_adjustment_summary(self) -> dict:
        """
        Get a summary of current adjustment values.
        
        Returns:
            dict: Summary of adjustment statistics
        """
        with torch.no_grad():
            return {
                'intrinsic_deltas': {
                    'mean': self.intrinsic_deltas.mean().item(),
                    'std': self.intrinsic_deltas.std().item(),
                    'min': self.intrinsic_deltas.min().item(),
                    'max': self.intrinsic_deltas.max().item()
                },
                'rotation_deltas': {
                    'mean': self.rotation_deltas.mean().item(),
                    'std': self.rotation_deltas.std().item(),
                    'min': self.rotation_deltas.min().item(),
                    'max': self.rotation_deltas.max().item()
                },
                'translation_deltas': {
                    'mean': self.translation_deltas.mean().item(),
                    'std': self.translation_deltas.std().item(),
                    'min': self.translation_deltas.min().item(),
                    'max': self.translation_deltas.max().item()
                }
            }
    
    def save_adjustments(self, filepath: str):
        """
        Save adjustment parameters to file.
        
        Args:
            filepath (str): Path to save the parameters
        """
        torch.save(self.state_dict(), filepath)
    
    def load_adjustments(self, filepath: str):
        """
        Load adjustment parameters from file.
        
        Args:
            filepath (str): Path to load the parameters from
        """
        self.load_state_dict(torch.load(filepath, map_location=self.device))
    
    def __str__(self):
        """String representation of the adjustments."""
        return f"Adjustments(num_cameras={self.num_cameras}, total_params={self.get_total_parameters()})"
    
    def __repr__(self):
        """Detailed string representation of the adjustments."""
        summary = self.get_adjustment_summary()
        return (f"Adjustments(num_cameras={self.num_cameras}, "
                f"total_params={self.get_total_parameters()}, "
                f"intrinsic_std={summary['intrinsic_deltas']['std']:.4f}, "
                f"rotation_std={summary['rotation_deltas']['std']:.4f}, "
                f"translation_std={summary['translation_deltas']['std']:.4f})")
