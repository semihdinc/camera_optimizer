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

# Camera intrinsic parameters
intrinsics = {
    'fx': 800,  # focal length in pixels (x)
    'fy': 800,  # focal length in pixels (y)
    'cx': 320,  # principal point x
    'cy': 240   # principal point y
}

K = np.array([
    [intrinsics['fx'], 0, intrinsics['cx']],
    [0, intrinsics['fy'], intrinsics['cy']],
    [0, 0, 1]
])

# Camera pose (extrinsics): rotation (R) and translation (t)
# Let's assume the camera is at (0,0,0) looking along +Z, so R=I, t=0
pose = {
    'R': np.eye(3),
    't': np.zeros((3, 1))
}

def project_point(X_world, K, R, t):
    """
    Projects a 3D point in world coordinates to image pixel coordinates.
    X_world: (3,) numpy array
    K: (3,3) intrinsic matrix
    R: (3,3) rotation matrix
    t: (3,1) translation vector
    Returns: (u, v) pixel coordinates
    """
    # Convert to camera coordinates
    X_world = np.array(X_world).reshape(3, 1)
    X_cam = R @ X_world + t  # (3,1)
    # Project to image plane
    x_proj = K @ X_cam  # (3,1)
    u = x_proj[0,0] / x_proj[2,0]
    v = x_proj[1,0] / x_proj[2,0]
    return u, v

# Sample 3D point in world coordinates
X_world = [1, 2, 10]  # (x, y, z) in meters

u, v = project_point(X_world, K, pose['R'], pose['t'])

print("Camera Intrinsics (K):\n", K)
print("Camera Pose (R):\n", pose['R'])
print("Camera Pose (t):\n", pose['t'])
print("3D World Point:", X_world)
print("Projected pixel coordinates: (u, v) = ({:.2f}, {:.2f})".format(u, v))
