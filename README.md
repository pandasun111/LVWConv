# lvwConv_csdattn
![](./fig/Framework_6.jpg)

**code of An Unsupervised Point Cloud Segmentation Model Guided by Controllable 3D Priors Convolution**

## Setup
Setting up for this project involves installing dependencies. 

To install all the dependencies, please run the following:
```shell script
sudo apt install build-essential python3-dev libopenblas-dev
conda env create -f env.yml
conda activate growsp
pip install -U MinkowskiEngine --install-option="--blas=openblas" -v --no-deps
```

## Running codes

### Preparing the dataset
```shell script
python data_prepare/data_prepare_S3DIS.py --data_path ${your_S3DIS}
```

### Construct initial superpoints:
```shell script
python data_prepare/initialSP_prepare_S3DIS.py
```

### Training:
```shell script
CUDA_VISIBLE_DEVICES=0 python3.8 train_S3DIS.py
```



**Our code builds on [GrowSP](https://github.com/vLAR-group/GrowSP/). Many thanks to GrowSP for a fantastic framework.**

