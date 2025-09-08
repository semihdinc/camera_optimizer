"""
Camera Projection and Coordinate System Convention

This script implements 3D point projection using the OpenCV coordinate system convention.

COORDINATE SYSTEM CONVENTION (OpenCV):
- Origin: Camera center (0, 0, 0)
- X-axis: Points to the right (positive X goes right)
- Y-axis: Points down (positive Y goes down) 
- Z-axis: Points forward (positive Z goes away from camera, into the scene)
- Camera looks along the +Z direction

IMAGE COORDINATE SYSTEM:
- u (horizontal): Increases to the right
- v (vertical): Increases downward
- Origin (0,0): Top-left corner of image

TRANSFORMATION PIPELINE:
1. World coordinates (X_world) -> Camera coordinates (X_cam)
   X_cam = R * X_world + t
2. Camera coordinates -> Image coordinates (u, v)
   [u]   [fx  0  cx] [X_cam]
   [v] = [0  fy  cy] [Y_cam]
   [1]   [0   0   1] [Z_cam]
   u = (fx * X_cam + cx * Z_cam) / Z_cam
   v = (fy * Y_cam + cy * Z_cam) / Z_cam

This differs from OpenGL convention where:
- Y-axis points up (not down)
- Camera looks along -Z direction
- Image coordinates have origin at bottom-left
"""

import numpy as np

class Camera:
    
    def __init__(self, camera_dict):
        """
        Initialize camera from dictionary containing intrinsics and pose.
        
        Args:
            camera_dict (dict): Dictionary containing:
                - 'id': Camera identifier
                - 'name': Camera name
                - 'intrinsics': Dictionary with fx, fy, cx, cy
                - 'pose': Dictionary with R (rotation matrix) and t (translation vector)
        """
        self.id = camera_dict['id']
        
        # Extract intrinsics
        intrinsics = camera_dict['intrinsics']
        self.fx = intrinsics['fx']
        self.fy = intrinsics['fy']
        self.cx = intrinsics['cx']
        self.cy = intrinsics['cy']
        
        # Create intrinsic matrix K
        self.K = np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ])
        
        # Extract pose
        pose = camera_dict['pose']
        self.R = np.array(pose['R'])
        self.t = np.array(pose['t']).reshape(3, 1)
        
    def project_to_image(self, X_world):
        # Controls: check input shape and type
        if X_world.ndim != 2 or X_world.shape[1] != 3:
            raise ValueError("X_world must be of shape (N, 3) where N is the number of points")
        if X_world.shape[0] == 0:
            return np.empty((0, 2))

        # Project points
        X_world_T = X_world.T  # (3, N)
        X_cam = self.R @ X_world_T + self.t  # (3, N)
        # Check for points behind the camera (z <= 0)
        z = X_cam[2, :]
        if np.any(z <= 0):
            raise ValueError("Some points are behind the camera (z <= 0) after transformation.")

        x_proj = self.K @ X_cam  # (3, N)
        u = x_proj[0, :] / x_proj[2, :]
        v = x_proj[1, :] / x_proj[2, :]
        return np.stack([u, v], axis=1)  # (N, 2)

    

