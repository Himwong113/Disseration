import numpy as np
import open3d as o3d
import pandas as pd
from utils.visual_util import build_pointcloud,build_bbox3d
from data_prepare.kittidet import kittidet_util as utils
import tqdm
def box_to_segm(points, objects, relax=0.01):
    """
    :param points: (N, 3).
    :param objects: list of Object3d.
    :return:
        segm: (N,).
    """
    n_point = points.shape[0]
    segm = np.zeros(n_point, dtype=np.int32)

    pc = np.copy(points)
    pc[:, :2] *= -1.

    for sid, obj in enumerate(objects):
        #if obj.type != 'Car':
        #    continue

        R = utils.roty(-obj.ry)
        transl = obj.t
        l, w, h = obj.l, obj.w, obj.h

        pc_tr = pc - transl
        pc_tr = np.einsum('ij,nj->ni', R, pc_tr)

        # Select points within bounding box
        within_box_x = np.logical_and(pc_tr[:, 0] > (-l / 2 - relax), pc_tr[:, 0] < (l / 2 + relax))
        within_box_y = np.logical_and(pc_tr[:, 1] > (-h - relax), pc_tr[:, 1] < relax)
        within_box_z = np.logical_and(pc_tr[:, 2] > (-w / 2 - relax), pc_tr[:, 2] < (w / 2 + relax))
        within_box = np.logical_and(np.logical_and(within_box_x, within_box_y), within_box_z)

        # Grant segmentation ID (Foreground objects start from 1)
        segm[within_box] = sid + 1
    return segm

def get_label_objects(idx):
    label_filename = label_root+ "%06d.txt" % (idx)
    #print(f'label filename={label_filename}')
    return utils.read_label(label_filename)

def get_calibration(idx):
    calib_filename = calib_root+"%06d.txt" % (idx)
    return utils.Calibration(calib_filename)

source_root="f:\\Master\\pointcloud\\training\\"
label_root=source_root+"label_2\\"
calib_root=source_root+"calib\\"

n_sample=8192
#pbar=tqdm.tqdm(total=n_sample)

#sid =8
def show3dpolt(sid):
    source_root="f:\\Master\\pointcloud\\training\\"
    label_root=source_root+"label_2\\"
    calib_root=source_root+"calib\\"

    root = 'f:\\Master\\pointcloud\\downsampled\\'+str(sid).zfill(6)
    loaded_data_path=str(root +'\\pc.npy')
    seg_data_path=str(root +'\\segm.npy')
    pc = np.load(loaded_data_path)
    seg_data = np.load(seg_data_path)

    n_sample =8192


    pcd2 = o3d.geometry.PointCloud()
    pcd2= build_pointcloud(pc,seg_data,with_background=True)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd2)

    # Get the view control from the Visualizer
    view_control = vis.get_view_control()

    # Set the view control parameters
    view_control.set_front([0, 0, -1])
    view_control.set_up([0, 1, 10])
    view_control.set_lookat([0, 2, 0])


    # Start the visualization
    vis.run()
    vis.destroy_window()

"""
### download dataset 

for i in range(8192):
    root = 'f:\\Master\\pointcloud\\downsampled\\'+str(i).zfill(6)
    loaded_data_path=str(root +'\\pc.npy')
    
    pc = np.load(loaded_data_path)
    
    label_objects = get_label_objects(i)
    calib = get_calibration(i)
    boxes_3d =[]
    for obj in label_objects:
            if obj.type == "DontCare":
                continue
            # Draw 3d bounding box
            _, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
            box3d_pts_3d[:, :2] *= -1.
            boxes_3d.append(box3d_pts_3d)
    segm = box_to_segm(pc, label_objects)
    np.save( 'f:\\Master\\pointcloud\\downsampled\\'+str(i).zfill(6)+'\\segm.npy',segm)
    pbar.update(1)

"""

"""
sid =0
source_root="f:\\Master\\pointcloud\\training\\"
label_root=source_root+"label_2\\"
calib_root=source_root+"calib\\"

root = 'f:\\Master\\pointcloud\\downsampled\\'+str(sid).zfill(6)
loaded_data_path=str(root +'\\pc.npy')
seg_data_path=str(root +'\\segment.npy')
pc = np.load(loaded_data_path)
seg_data = np.load(seg_data_path)

n_sample =8192


pcd2 = o3d.geometry.PointCloud()
pcd2= build_pointcloud(pc,seg_data,with_background=True)

vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(pcd2)

# Get the view control from the Visualizer
view_control = vis.get_view_control()

# Set the view control parameters
view_control.set_front([0, 0, -1])
view_control.set_up([0, 1, 0])
view_control.set_lookat([0,5, 15])


# Start the visualization
vis.run()
vis.destroy_window()


for i in range(8192):
    root = 'f:\\Master\\pointcloud\\downsampled\\'+str(i).zfill(6)
    loaded_data_path=str(root +'\\pc.npy')
    
    pc = np.load(loaded_data_path)
    
    label_objects = get_label_objects(i)
    calib = get_calibration(i)
    boxes_3d =[]
    for obj in label_objects:
            if obj.type == "DontCare":
                continue
            # Draw 3d bounding box
            _, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
            box3d_pts_3d[:, :2] *= -1.
            boxes_3d.append(box3d_pts_3d)
    segm = box_to_segm(pc, label_objects)
    np.save( 'f:\\Master\\pointcloud\\downsampled\\'+str(i).zfill(6)+'\\segm.npy',segm)
    pbar.update(1)

pcd2 = o3d.geometry.PointCloud()
pcd2= build_pointcloud(pc,segm,with_background=True)

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

for i in range(8192):
    label_objects = get_label_objects(sid)
    calib = get_calibration(sid)
    boxes_3d =[]
    for obj in label_objects:
            if obj.type == "DontCare":
                continue
            # Draw 3d bounding box
            _, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
            box3d_pts_3d[:, :2] *= -1.
            boxes_3d.append(box3d_pts_3d)
    segm = box_to_segm(pc, label_objects)
    np.save( 'f:\\Master\\pointcloud\\downsampled\\'+str(sid).zfill(6)+"\\segment.npy",segm)
    pbar.update(1)
"""

#(19199, 3)
#(19199, 3)

"""
df =pd.DataFrame(seg_data)
df.to_csv(root+'seg.csv')

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




"""
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(seg_data.reshape(-1, 3))

grey_color = np.asarray([0.5, 0.5, 0.5])
pcd.colors = o3d.utility.Vector3dVector(np.tile(grey_color, (seg_data[0], 1)))
o3d.visualization.draw_geometries([pcd])
"""


"""
pcd2 = o3d.geometry.PointCloud()
pcd2= build_pointcloud(pc=loaded_data,segm=seg_data,with_background=True)

vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(pcd2)

# Get the view control from the Visualizer
view_control = vis.get_view_control()

# Set the view control parameters
view_control.set_front([0, 0, -1])
view_control.set_up([0, 0.1, 0.1])
view_control.set_lookat([0,0.01, 0])


# Start the visualization
vis.run()
vis.destroy_window()
print(loaded_data)
"""



"""

loaded_data = np.load(loaded_data_path)
seg_data = np.load(seg_data_path)
print(seg_data)
df =pd.DataFrame(seg_data)
df.to_csv(root+'seg.csv')
print (f'seg_data.shape={seg_data.shape}')
print (f'load_data.shape={loaded_data.shape}')


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
