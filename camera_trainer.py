"""
PyTorch-based camera parameter optimization script.

This script learns adjustment parameters for noisy cameras using ground truth projections.
It optimizes intrinsic and extrinsic parameters to minimize projection errors.
"""

import torch
import torch.nn as nn
import torch.optim as optim

from cameras.adjustments import CameraOptimizer
from data.dataloader import load_data, create_dataloader

import argparse
import matplotlib.pyplot as plt

class Trainer:
    """
    Trainer class for camera parameter optimization.
    
    This class handles data loading, model creation, optimization setup, training, and evaluation.
    """
    
    def __init__(self, noisy_cameras_file: str, projection_results_file: str, 
                 device: str = 'cpu', lr: float = 0.001, batch_size: int = 1, seed: int = 42):

        self.device = device
        self.lr = lr
        self.batch_size = batch_size
        self.seed = seed
        
        # Set random seeds for reproducibility
        self.set_seed(seed)
        
        # Load data
        print("Loading data...")
        self.noisy_cameras, self.projection_data, self.camera_id_to_idx = load_data(
            noisy_cameras_file, projection_results_file
        )
        print(f"Loaded {len(self.noisy_cameras)} cameras and {len(self.projection_data)} projection samples")
        
        # Create dataloader
        self.dataloader = create_dataloader(
            self.projection_data, self.camera_id_to_idx, batch_size, device
        )
        
        # Create model
        self.model = CameraOptimizer(self.noisy_cameras, device)
        
        # Create optimizer and loss function
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
    
    def set_seed(self, seed: int):
        """Set random seeds for reproducible results."""
        torch.manual_seed(seed)
        
    def train(self, num_epochs: int = 1000):
        """
        Train the camera optimization model.
        
        Args:
            num_epochs: Number of training epochs
            
        Returns:
            List of average losses per epoch
        """
        self.model.train()
        losses = []
        
        for epoch in range(num_epochs):
            total_loss = 0.0
            num_batches = 0
            
            for world_points, pixels, camera_indices in self.dataloader:
                self.optimizer.zero_grad()
                            
                # Forward pass
                predicted_pixels = self.model(world_points, camera_indices)
                
                # Compute loss
                loss = self.criterion(predicted_pixels, pixels)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            losses.append(avg_loss)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {avg_loss:.6f}")
        
        return losses
    
    def evaluate(self):
        """
        Evaluate the trained model.
            
        Returns:
            Average projection error in pixels
        """
        self.model.eval()
        total_error = 0.0
        num_samples = 0
        
        with torch.no_grad():
            for world_points, pixel_coords, camera_indices in self.dataloader:
                predicted_pixels = self.model(world_points, camera_indices)
                error = torch.norm(predicted_pixels - pixel_coords, dim=1)
                total_error += error.sum().item()
                num_samples += len(error)
        
        avg_error = total_error / num_samples if num_samples > 0 else 0
        return avg_error
    
    def get_adjustment_summary(self):
        """Get summary statistics of adjustment parameters."""
        return self.model.adjustments.get_adjustment_summary()
    
    def get_total_parameters(self):
        """Get total number of parameters."""
        return self.model.adjustments.get_total_parameters()


def main():
    parser = argparse.ArgumentParser(description='Optimize camera parameters using PyTorch')
    parser.add_argument('--noisy_cameras', default='data/cameras_noisy.json', help='Noisy cameras JSON file')
    parser.add_argument('--projection_results', default='data/projection_results.json', help='Projection results JSON file')
    parser.add_argument('--epochs', type=int, default=5000, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--device', default='cpu', help='Device to use (cpu/cuda)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducible results')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Using seed: {args.seed}")
    
    # Create trainer (handles data loading internally)
    trainer = Trainer(
        noisy_cameras_file=args.noisy_cameras,
        projection_results_file=args.projection_results,
        lr=args.lr,
        batch_size=args.batch_size,
        device=device,
        seed=args.seed,
    )

    print(f"Model created with {trainer.get_total_parameters()} parameters")
    print(f"Parameters per camera: {trainer.get_total_parameters() / len(trainer.noisy_cameras)}")
    
    # Train model
    print("Training model...")
    losses = trainer.train(args.epochs)
    
    # Evaluate model
    print("Evaluating model...")
    avg_error = trainer.evaluate()
    print(f"Average projection error: {avg_error:.4f} pixels")
    
    # Print adjustment statistics
    print("\nAdjustment Statistics:")
    summary = trainer.get_adjustment_summary()
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
    print("Training completed!")

if __name__ == "__main__":
    main()
