import numpy as np
import json
from camera import Camera

def load_cameras(json_file):
    """Load camera definitions from JSON file and create Camera objects."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    return [Camera(camera_dict) for camera_dict in data['cameras']]

# Load cameras from JSON file
cameras = load_cameras('cameras.json')

# Sample 3D point in world coordinates
X_world = np.array([
    [1, 2, 10],
    [0, 0, 5],
    [2, -1, 8],
    [-3, 4, 12],
    [5, 5, 15]
])  # 5 world points, each row is (x, y, z) in meters

print("=" * 60)
print("3D POINT PROJECTION TO MULTIPLE CAMERAS")
print("=" * 60)
print(f"3D World Point: {X_world}")
print()

# Project the 3D point to all cameras
for camera in cameras:
    print(f"Camera: {camera.name} (ID: {camera.id})")
    print("-" * 40)
    
    # Project the point using the camera's method
    uv = camera.project_to_image(X_world)
    
    print("Projected pixel coordinates (u, v):")
    print(uv)
