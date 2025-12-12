### Introduction

We have integrated the superpoint partition methods appearing in the following four articles, for a total of five methods.

1. Voxel cloud connectivity segmentation-supervoxels for point clouds
2. Toward better boundary preserved supervoxel segmentation for 3D point clouds
3. Large-scale point cloud semantic segmentation with superpoint graphs
4. Scannet: Richly-annotated 3d reconstructions of indoor scenes

![](./teaser.png)

### install

```shell
pip install pybind11
conda install -c anaconda boost
conda install -c omnia eigen3 
conda install eigen
python setup.py install --conda_home={Your Anaconda Home} --env_name={Your Virtual Environment Name}
```

please specify the anaconda home and the name of your environment since we need to decide the include path of eigen3.

### Definition

```python
# supervoxel partition
def super_voxel(
    points, 
    colors,
    knn=15,
    resolution=1.0,
    voxel_resolution=0.03,
    vtype="vccsknn"
):
    '''
    points: input point cloud with shape [N, 3]
    colors: input point cloud colors with shape [N, 3], and value should be normalized to [0,1]
    knn: k nearest neighbor
    resolution: resolution used in the octree, larger value will lead to less superpoints
    voxel_resolution: resolution used to seed the supervoxels, only useful when vtype=="vccs", voxel_resolution < 2.0 * resolution, larger value will lead to less superpoints
    vtype: choose from "raw", "vccs" and "vccsknn"
    '''
    pass

# cut pursuit partition
def cut_pursuit(
    points,
    colors,
    reg_strength,
    knn_adj=10,
    knn_geof=45,
    lambda_edge_weight=1
):
    '''
    points: input point cloud with shape [N, 3]
    colors: input point cloud colors with shape [N, 3], and value should be normalized to [0,1]
    reg_strength: larger value will lead to less superpoints
    knn_adj: k neighbors used for building graph
    knn_geof: k neighbors used for compute handcrafted features
    '''
    pass

# superpoint generation for mesh data
def mesh_segmentor(
    points,
    faces,
    kthr=0.01,
    minverts=30
):
    '''
    points: input point cloud with shape [N, 3]
    faces: input faces of a mesh, shape [M, 3]
    kthr & minverts: larger value will lead to less superpoints
    '''
    pass
```

### Usage

we provide a use case in `test.py`, once installed, you can run with `python test.py`