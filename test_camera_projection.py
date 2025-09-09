import torch
from data.dataloader import load_cameras

# Load cameras from JSON file
cameras = load_cameras('data/cameras.json')
noisy_cameras = load_cameras('data/cameras_noisy.json')

# Sample 3D point in world coordinates
X_world = torch.tensor([
    [1, 2, 10],
    [0, 0, 5],
    [2, -1, 8],
    [-3, 4, 12],
    [5, 5, 15]
], dtype=torch.double)  # 5 world points, each row is (x, y, z) in meters

print("3D World Points:")
print("-" * 60)
print(f"{X_world}")
print()

# Project the 3D point to all cameras
for cam_id in range(len(cameras)):  
    print(f"Camera ID: {cam_id}")
    print("-" * 60)
    
    # Project the point using the camera's method
    uv = cameras[cam_id].project_to_image(X_world)
    uv_noisy = noisy_cameras[cam_id].project_to_image(X_world)
    
    print("Projected pixel coordinates (u, v):")
    error = torch.norm(uv - uv_noisy, dim=1)
    print(f"Error: {error.mean().item()}")
    print(torch.cat([uv, uv_noisy], dim=1))
    print()

    # # Save all world points and their projections for this camera in a single dict
    # result = {
    #     'camera_id': camera.id,
    #     'world_points': X_world.tolist(),
    #     'pixels': uv.tolist()
    # }
    # projection_results.append(result)

# # Save projection results to a JSON file
# with open('projection_results.json', 'w') as f:
#     json.dump(projection_results, f, indent=4)