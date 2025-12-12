import numpy as np
import open3d as o3d
import torch
import SPOps
from collections import defaultdict

pcd = o3d.io.read_triangle_mesh("GrowSP_w7/data_prepare/demo2.ply")
points = np.array(pcd.vertices)
colors = np.array(pcd.vertex_colors)
faces = np.array(pcd.triangles)

print("##points.shape:", points.shape)
print("supervoxel partition ...")

raw_label = SPOps.super_voxel(points, colors, vtype="raw")
vccs_label = SPOps.super_voxel(points, colors, vtype="vccs")
vccsknn_label = SPOps.super_voxel(points, colors, vtype="vccsknn")
print("total superpoints: raw {}, vccs {}, vccsknn {}".format(raw_label.max(), vccs_label.max(), vccsknn_label.max()))


print("cutpursuit partition ...")
cp_label = SPOps.cut_pursuit(points, colors, 0.08)
print("total superpoints: {}".format(cp_label.max()))

print("mesh segmentor ...")
ms_label = SPOps.mesh_segmentor(points, faces, 0.1, 40)
print("total superpoints: {}".format(ms_label.max()))

max_label = np.max([raw_label.max(), vccs_label.max(), vccsknn_label.max(), cp_label.max(), ms_label.max()])

colormap = np.random.random_sample((max_label + 1, 3))


print("##raw_label.shape:",raw_label.shape)

pcd_raw = o3d.geometry.PointCloud()
pcd_raw.points = o3d.utility.Vector3dVector(points)
pcd_raw.colors = o3d.utility.Vector3dVector(colormap[raw_label])
o3d.io.write_point_cloud("voxel.ply", pcd_raw)

pcd_raw = o3d.geometry.PointCloud()
pcd_raw.points = o3d.utility.Vector3dVector(points)
pcd_raw.colors = o3d.utility.Vector3dVector(colormap[vccs_label])
o3d.io.write_point_cloud("vccs.ply", pcd_raw)

pcd_raw = o3d.geometry.PointCloud()
pcd_raw.points = o3d.utility.Vector3dVector(points)
pcd_raw.colors = o3d.utility.Vector3dVector(colormap[vccsknn_label])
o3d.io.write_point_cloud("vccsknn.ply", pcd_raw)

pcd_raw = o3d.geometry.PointCloud()
pcd_raw.points = o3d.utility.Vector3dVector(points)
pcd_raw.colors = o3d.utility.Vector3dVector(colormap[cp_label])
o3d.io.write_point_cloud("cp.ply", pcd_raw)


pcd_raw = o3d.geometry.PointCloud()
pcd_raw.points = o3d.utility.Vector3dVector(points)
pcd_raw.colors = o3d.utility.Vector3dVector(colormap[ms_label])
o3d.io.write_point_cloud("ms.ply", pcd_raw)



#sub_xyz, sub_rgb, _ = SPOps.grid_sample(points.astype(np.float32), colors.astype(np.float32), np.zeros(points.shape[0]).astype(np.int), 0.02)


### TODO: test case for cutpursuit2
print("cutpursuit2 partition ...")
# exp_scene_dir = './27_shizilin_taduo_1.pth'
# shizilin_points = torch.load(exp_scene_dir)['coord']
# shizilin_rgb = torch.load(exp_scene_dir)['color']
# shizilin_labels = torch.load(exp_scene_dir)['semantic_gt']
# n_cls = 16
# np.random.seed(0)
# colormap = np.random.random_sample((n_cls + 1, 3))
# encoded_arr = np.zeros([shizilin_labels.size, n_cls+1], dtype=int)
# encoded_arr[np.arange(shizilin_labels.size), shizilin_labels]=1 #one hot labels with shape [N, n_cls+1]
# onehot_labels = encoded_arr
# dump = SPOps.cut_pursuit2(shizilin_points, onehot_labels, knn_adj=15, knn_local=45)
# n_objects = len(np.unique(dump))
# indices_dict = defaultdict(list)
# for i, label in enumerate(dump): # reverse indexing
#     indices_dict[label].append(i)  
# print('visualizing scene semantic labels...')
# pcd_raw = o3d.geometry.PointCloud()
# pcd_raw.points = o3d.utility.Vector3dVector(shizilin_points)
# pcd_raw.colors = o3d.utility.Vector3dVector(colormap[shizilin_labels])
# o3d.visualization.draw_geometries([pcd_raw])
# print('done')
# print('visualizing objects extracted from scene semantic labels...')
# print(f'number of objects:{n_objects}')
# for i in range(n_objects):
#     tmp_pcd = o3d.geometry.PointCloud()
#     tmp_pcd.points = o3d.utility.Vector3dVector(shizilin_points[indices_dict[i]])
#     tmp_pcd.colors = o3d.utility.Vector3dVector(colormap[shizilin_labels[indices_dict[i]]])

#     o3d.visualization.draw_geometries([tmp_pcd])
# print('done')