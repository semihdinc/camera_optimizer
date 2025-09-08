"""
Data loading utilities for camera parameter optimization.

This module contains the ProjectionDataset class and data loading functions
for training camera parameter optimization models.
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from cameras.camera import Camera
from typing import Dict, List


class ProjectionDataset(Dataset):
    """Dataset for camera projection optimization."""
    
    def __init__(self, projection_data: List[Dict], camera_id_to_idx: Dict[str, int], device='cpu'):
        self.data = projection_data
        self.camera_id_to_idx = camera_id_to_idx
        self.device = device
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        world_points = torch.tensor(item['world_points'], device=self.device, dtype=torch.double)
        pixel_coords = torch.tensor(item['pixels'], device=self.device, dtype=torch.double)
        camera_idx = self.camera_id_to_idx[item['camera_id']]
        return world_points, pixel_coords, camera_idx


def load_data(noisy_cameras_file: str, projection_results_file: str):
    """Load noisy cameras and projection results."""
    # Load noisy cameras
    with open(noisy_cameras_file, 'r') as f:
        cameras_data = json.load(f)
    noisy_cameras = [Camera(cam_dict) for cam_dict in cameras_data['cameras']]
    
    # Load projection results
    with open(projection_results_file, 'r') as f:
        projection_data = json.load(f)
    
    # Create camera ID to index mapping
    camera_id_to_idx = {cam.id: i for i, cam in enumerate(noisy_cameras)}
    
    return noisy_cameras, projection_data, camera_id_to_idx


def create_dataloader(projection_data: List[Dict], camera_id_to_idx: Dict[str, int], 
                     batch_size: int = 1, device: str = 'cpu') -> DataLoader:
    """Create a DataLoader for the projection dataset."""
    dataset = ProjectionDataset(projection_data, camera_id_to_idx, device)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
