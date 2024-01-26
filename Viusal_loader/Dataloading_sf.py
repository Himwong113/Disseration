import numpy as np
import open3d as o3d
from utils.visual_util import build_pointcloud


loaded_data = np.load('D:\\MasterAIBDDisseration\\RawDataset\\processed\\000000\\pc1.npy')
seg_data = np.load('D:\\MasterAIBDDisseration\\RawDataset\\processed\\000000\\segm.npy')

print(loaded_data.shape)
print(seg_data)


pcd2 = o3d.geometry.PointCloud()
pcd2= build_pointcloud(loaded_data,seg_data,with_background=True)

vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(pcd2)

# Get the view control from the Visualizer
view_control = vis.get_view_control()

# Set the view control parameters
view_control.set_front([0, 0, -1])
view_control.set_up([0, 1, -32])
view_control.set_lookat([0,5, 15])


# Start the visualization
vis.run()
vis.destroy_window()
print(loaded_data)

"""
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(loaded_data.reshape(-1, 3))

grey_color = np.asarray([0.5, 0.5, 0.5])
seg_color = np.asarray([0.4, 0.3, 0.4])
pcd.colors = o3d.utility.Vector3dVector(np.tile(grey_color, (loaded_data.shape[0], 1)))  # Corrected the shape argument
pcd.colors = o3d.utility.Vector3dVector(np.tile(seg_color, (seg_data.shape[0], 1)))
axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
o3d.visualization.draw_geometries([pcd,axes])



# Load data from an .npz file
loaded_data = np.load('D:\\MasterAIBDDisseration\\RawDataset\\processed\\000000\\pc1.npy')
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(loaded_data.reshape(-1, 3))

grey_color = np.asarray([0.5, 0.5, 0.5])
pcd.colors = o3d.utility.Vector3dVector(np.tile(grey_color, (loaded_data[0], 1)))
o3d.visualization.draw_geometries([pcd])
print (loaded_data)

"""

"""
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

"""
