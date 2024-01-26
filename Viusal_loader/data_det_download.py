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

sid =8
source_root="f:\\Master\\pointcloud\\training\\"
label_root=source_root+"label_2\\"
calib_root=source_root+"calib\\"

root = 'f:\\Master\\pointcloud\\downsampled\\'+str(sid).zfill(6)
loaded_data_path=str(root +'\\pc.npy')
seg_data_path=str(root +'\\segm.npy')
pc = np.load(loaded_data_path)
seg_data = np.load(seg_data_path)

### download dataset 
pbar = tqdm.tqdm(total=n_sample)
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
    np.save( 'f:\\Master\\pointcloud\\downsampled_bk\\'+str(i).zfill(6)+'\\segm2.npy',segm)
    pbar.update(1)