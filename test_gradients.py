#!/usr/bin/env python3
"""
Test script to verify that gradients flow properly through intrinsic_deltas.
"""

import torch
import json
from camera import Camera
from adjustments import Adjustments

def test_gradient_flow():
    """Test that gradients flow through intrinsic_deltas."""
    device = 'cpu'
    
    # Create a simple camera
    camera_dict = {
        'id': 'test_camera',
        'intrinsics': {'fx': 800.0, 'fy': 800.0, 'cx': 320.0, 'cy': 240.0},
        'pose': {'R': [[1, 0, 0], [0, 1, 0], [0, 0, 1]], 't': [0, 0, 0]}
    }
    camera = Camera(camera_dict, device=device)
    
    # Create adjustments
    adjustments = Adjustments(1, device=device)
    
    # Test data
    X_world = torch.tensor([[1.0, 2.0, 10.0]], device=device, dtype=torch.double)
    
    # Get adjustments
    intrinsic_deltas = adjustments.get_intrinsic_adjustments(0)
    rotation_deltas, translation_deltas = adjustments.get_extrinsic_adjustments(0)
    
    print(f"Initial intrinsic_deltas: {intrinsic_deltas}")
    print(f"intrinsic_deltas.requires_grad: {intrinsic_deltas.requires_grad}")
    
    # Project with corrections
    projected = camera.project_to_image_with_corrections(
        X_world, intrinsic_deltas, rotation_deltas, translation_deltas
    )
    
    print(f"Projected pixels: {projected}")
    print(f"Projected pixels.requires_grad: {projected.requires_grad}")
    
    # Test gradient flow
    target = torch.tensor([[400.0, 300.0]], device=device, dtype=torch.double)
    loss = torch.nn.functional.mse_loss(projected, target)
    
    print(f"Loss: {loss}")
    print(f"Loss.requires_grad: {loss.requires_grad}")
    
    # Backward pass
    loss.backward()
    
    print(f"Gradients for intrinsic_deltas: {adjustments.intrinsic_deltas.grad}")
    print(f"Gradients for rotation_deltas: {adjustments.rotation_deltas.grad}")
    print(f"Gradients for translation_deltas: {adjustments.translation_deltas.grad}")
    
    # Check if gradients are non-zero
    intrinsic_grad_norm = torch.norm(adjustments.intrinsic_deltas.grad[0])
    print(f"Intrinsic gradient norm: {intrinsic_grad_norm}")
    
    if intrinsic_grad_norm > 1e-8:
        print("✅ Gradients are flowing through intrinsic_deltas!")
    else:
        print("❌ No gradients flowing through intrinsic_deltas!")

if __name__ == "__main__":
    test_gradient_flow()
