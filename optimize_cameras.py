"""
PyTorch-based camera parameter optimization script.

This script learns adjustment parameters for noisy cameras using ground truth projections.
It optimizes intrinsic and extrinsic parameters to minimize projection errors.
"""

import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from camera import Camera
from adjustments import Adjustments
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import argparse


class CameraOptimizer(nn.Module):
    """
    PyTorch module for optimizing camera parameters.
    
    This module uses the Adjustments class to learn correction parameters
    for both intrinsics and extrinsics to correct the noisy camera parameters.
    """
    
    def __init__(self, noisy_cameras: List[Camera], device='cpu'):
        super(CameraOptimizer, self).__init__()
        self.device = device
        self.num_cameras = len(noisy_cameras)
        
        # Store original noisy parameters
        self.noisy_cameras = noisy_cameras
        
        # Create adjustments module
        self.adjustments = Adjustments(self.num_cameras, device)
           
    def project_points(self, X_world: torch.Tensor, camera_indices: torch.Tensor) -> torch.Tensor:
        """Project a 3D point using corrected camera parameters."""
        projected_pixels = []
        for id, cam in enumerate(camera_indices):
            # Get adjustments for this camera
            intrinsic_deltas = self.adjustments.get_intrinsic_adjustments(cam)
            rotation_deltas, translation_deltas = self.adjustments.get_extrinsic_adjustments(cam)
            
            # Use the camera's projection method with corrections
            camera = self.noisy_cameras[cam]
            px = camera.project_to_image_with_corrections(
                X_world[id], intrinsic_deltas, rotation_deltas, translation_deltas
            )
            projected_pixels.append(px)
        
        return torch.stack(projected_pixels, dim=0)


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
        world_points = torch.tensor(item['world_points'], device=self.device, dtype=torch.float32)
        pixel_coords = torch.tensor(item['pixels'], device=self.device, dtype=torch.float32)
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


def train_model(model: CameraOptimizer, dataloader: DataLoader, num_epochs: int = 1000, lr: float = 0.01):
    """Train the camera optimization model."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    losses = []
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0
        
        for world_points, pixels, camera_indices in dataloader:
            optimizer.zero_grad()
                        
            # Forward pass
            predicted_pixels = model.project_points(world_points, camera_indices)
            
            # Compute loss
            loss = criterion(predicted_pixels, pixels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        losses.append(avg_loss)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {avg_loss:.6f}")
    
    return losses


def evaluate_model(model: CameraOptimizer, dataloader: DataLoader):
    """Evaluate the trained model."""
    model.eval()
    total_error = 0.0
    num_samples = 0
    
    with torch.no_grad():
        for world_points, pixel_coords, camera_indices in dataloader:
            predicted_pixels = model.project_points(world_points, camera_indices)
            error = torch.norm(predicted_pixels - pixel_coords, dim=1)
            total_error += error.sum().item()
            num_samples += len(error)
    
    avg_error = total_error / num_samples if num_samples > 0 else 0
    return avg_error


def main():
    parser = argparse.ArgumentParser(description='Optimize camera parameters using PyTorch')
    parser.add_argument('--noisy_cameras', default='cameras_noisy.json', help='Noisy cameras JSON file')
    parser.add_argument('--projection_results', default='projection_results.json', help='Projection results JSON file')
    parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--device', default='cpu', help='Device to use (cpu/cuda)')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    noisy_cameras, projection_data, camera_id_to_idx = load_data(args.noisy_cameras, args.projection_results)
    print(f"Loaded {len(noisy_cameras)} cameras and {len(projection_data)} projection samples")
    
    # Create dataset and dataloader
    dataset = ProjectionDataset(projection_data, camera_id_to_idx, device)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Create model
    model = CameraOptimizer(noisy_cameras, device)
    print(f"Model created with {model.adjustments.get_total_parameters()} parameters")
    print(f"Parameters per camera: {model.adjustments.get_total_parameters() / len(noisy_cameras)}")
    
    # Train model
    print("Training model...")
    losses = train_model(model, dataloader, args.epochs, args.lr)
    
    # Evaluate model
    print("Evaluating model...")
    avg_error = evaluate_model(model, dataloader)
    print(f"Average projection error: {avg_error:.4f} pixels")
    
    # Print adjustment statistics
    print("\nAdjustment Statistics:")
    summary = model.adjustments.get_adjustment_summary()
    print(f"Intrinsic adjustments - Mean: {summary['intrinsic_deltas']['mean']:.4f}, Std: {summary['intrinsic_deltas']['std']:.4f}")
    print(f"Rotation adjustments - Mean: {summary['rotation_deltas']['mean']:.4f}, Std: {summary['rotation_deltas']['std']:.4f}")
    print(f"Translation adjustments - Mean: {summary['translation_deltas']['mean']:.4f}, Std: {summary['translation_deltas']['std']:.4f}")
    
    # Plot training curve
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.yscale('log')
    plt.grid(True)
    plt.savefig('training_curve.png')
    plt.show()
    
    print("Training completed!")


if __name__ == "__main__":
    main()
