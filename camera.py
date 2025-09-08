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

import torch

class Camera:
    
    def __init__(self, camera_dict, device='cpu'):
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
        self.device = device

        # Extract intrinsics
        intrinsics = camera_dict['intrinsics']
        self.fx = intrinsics['fx']
        self.fy = intrinsics['fy']
        self.cx = intrinsics['cx']
        self.cy = intrinsics['cy']
        
        # Create intrinsic matrix K
        self.K = torch.tensor([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], device=self.device, dtype=torch.double)
        
        # Extract pose
        pose = camera_dict['pose']
        R_matrix = torch.tensor(pose['R'], device=self.device)
        # Convert rotation matrix to axis-angle (Rodrigues vector)
        self.rotation_angles = self.rotation_matrix_to_axis_angle(R_matrix)
        self.t = torch.tensor(pose['t'], device=self.device).reshape(3,1)


    def axis_angle_to_rotation_matrix(self, axis_angle):
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
    
    def rotation_matrix_to_axis_angle(self, R):
        """
        Convert rotation matrix to axis-angle representation.
        
        Args:
            R (torch.Tensor): Rotation matrix (3x3)
            
        Returns:
            torch.Tensor: Axis-angle representation (3,)
        """
        # Trace of rotation matrix
        trace = torch.trace(R)
        angle = torch.acos(torch.clamp((trace - 1) / 2, -1, 1))
        
        if angle < 1e-8:
            return torch.zeros(3, device=self.device)
        
        # Axis of rotation
        axis = torch.tensor([
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1]
        ], device=self.device) / (2 * torch.sin(angle))
        
        return angle * axis

    def project_to_image(self, X_world):
        # Controls: check input shape and type
        if X_world.ndim != 2 or X_world.shape[1] != 3:
            raise ValueError("X_world must be of shape (N, 3) where N is the number of points")
        if X_world.shape[0] == 0:
            return torch.empty((0, 2), device=self.device)

        rot_matrix = self.axis_angle_to_rotation_matrix(self.rotation_angles)

        # Project points
        X_cam = rot_matrix @ X_world.T + self.t  # (3, N)
        # Check for points behind the camera (z <= 0)
        z = X_cam[2, :]
        if torch.any(z <= 0):
            raise ValueError("Some points are behind the camera (z <= 0) after transformation.")

        x_proj = self.K @ X_cam  # (3, N)
        u = x_proj[0, :] / x_proj[2, :]
        v = x_proj[1, :] / x_proj[2, :]
        return torch.stack([u, v], axis=1)  # (N, 2)
    
    def project_to_image_with_corrections(self, X_world, intrinsic_deltas, rotation_deltas, translation_deltas):
        """Project points using corrected camera parameters without modifying the camera state."""
        # Controls: check input shape and type
        if X_world.ndim != 2 or X_world.shape[1] != 3:
            raise ValueError("X_world must be of shape (N, 3) where N is the number of points")
        if X_world.shape[0] == 0:
            return torch.empty((0, 2), device=self.device)

        # Apply corrections to intrinsics
        fx_corrected = self.fx + intrinsic_deltas[0]
        fy_corrected = self.fy + intrinsic_deltas[1]
        cx_corrected = self.cx + intrinsic_deltas[2]
        cy_corrected = self.cy + intrinsic_deltas[3]
        
        # Create corrected intrinsic matrix (preserve computational graph)
        K_corrected = torch.stack([
            torch.stack([fx_corrected, torch.zeros_like(fx_corrected), cx_corrected]),
            torch.stack([torch.zeros_like(fy_corrected), fy_corrected, cy_corrected]),
            torch.stack([torch.zeros_like(fx_corrected), torch.zeros_like(fx_corrected), torch.ones_like(fx_corrected)])
        ]).to(dtype=fx_corrected.dtype)
        
        # Apply corrections to extrinsics
        rotation_angles_corrected = self.rotation_angles + rotation_deltas
        t_corrected = self.t + translation_deltas.unsqueeze(1)
        
        # Convert to rotation matrix
        rot_matrix = self.axis_angle_to_rotation_matrix(rotation_angles_corrected).to(dtype=X_world.dtype)

        # Project points
        X_cam = rot_matrix @ X_world.T + t_corrected  # (3, N)
        # Check for points behind the camera (z <= 0)
        z = X_cam[2, :]
        if torch.any(z <= 0):
            raise ValueError("Some points are behind the camera (z <= 0) after transformation.")

        x_proj = K_corrected @ X_cam  # (3, N)
        u = x_proj[0, :] / x_proj[2, :]
        v = x_proj[1, :] / x_proj[2, :]
        return torch.stack([u, v], axis=1)  # (N, 2)

    

