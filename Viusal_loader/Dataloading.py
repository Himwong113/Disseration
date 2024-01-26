import numpy as np
import open3d as o3d

# Load data from an .npz file
loaded_data = np.load('./Own topic/mbs-shapepart/data/000001.npz')

# Get the list of keys (array names) in the .npz file
array_keys = loaded_data.files

# Print the list of keys
for key in array_keys:
    print(key)

# Access and print the data associated with a specific key
if 'pc' in array_keys:
    pc_data = loaded_data['pc']
    print("Data for 'pc':")
    print(pc_data.shape)

#print(dir(loaded_data['pc']))

# Access the point cloud data using the correct key (e.g., 'pc' if that's the key in your file)
point_cloud_data = loaded_data['pc']

for i in range(4):

    temp = point_cloud_data[i]
    # Close the .npz file when you're done with it
    loaded_data.close()

    # Create an Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()

    # Set the point cloud data as the points in the Open3D PointCloud object
    pcd.points = o3d.utility.Vector3dVector(point_cloud_data.reshape(-1, 3))
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd,axes])