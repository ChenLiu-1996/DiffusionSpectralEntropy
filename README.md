# Manifold Topology

### The repository is structured in the following manner.
```
ManifoldTopology
    ├── external_src
    |   └── `${MODEL_NAME}/...`: git repos for external models
    ├── setup: Instructions for proper setup of environments?
    └── src
        ├── `investigate.py`: centralized place for our contribution.
        ├── `${MODEL_NAME}_model.py`: wrapper code to interface with external models.
        ├── `base.py`: a class with common methods to be inherited by `${MODEL_NAME}_model.py`.
        ├── unit_test
        |   └── `test_run_${MODEL_NAME}.py`: scripts to check validity of `${MODEL_NAME}_model.py`.
        └── utils: folder of utility functions.
```

### Environment
We developed the codebase in a miniconda environment.
Tested on Python 3.9.13 + PyTorch 1.12.1.
How we created the conda environment:
```
conda create --name $OUR_CONDA_ENV pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
conda activate $OUR_CONDA_ENV
conda install -c anaconda scikit-image pillow matplotlib seaborn tqdm
python -m pip install -U giotto-tda
cd diffusion_curvature
python -m pip install .
```

### Usage.
<details> <summary>Unit Test. Run the pretrained models.</summary>

```
$OUR_CONDA_ENV
cd src/unit_test/
python test_run_model.py --model barlowtwins
python test_run_model.py --model moco
python test_run_model.py --model simsiam
python test_run_model.py --model swav
```
</details>

### Preparing pretrained weights of external models.
<details> <summary>Barlow Twins</summary>

```
cd external_src/barlowtwins/
mkdir checkpoints && cd checkpoints
mkdir ImageNet && cd ImageNet
wget -O barlowtwins_bs2048_ep1000.pth.tar https://dl.fbaipublicfiles.com/barlowtwins/ljng/resnet50.pth
```
</details>

<details> <summary>MoCo</summary>

```
cd external_src/moco/
mkdir checkpoints && cd checkpoints
mkdir ImageNet && cd ImageNet
wget -O moco_v1_ep200.pth.tar https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v1_200ep/moco_v1_200ep_pretrain.pth.tar
wget -O moco_v2_ep200.pth.tar https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_200ep/moco_v2_200ep_pretrain.pth.tar
wget -O moco_v2_ep800.pth.tar https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_800ep/moco_v2_800ep_pretrain.pth.tar
```
</details>

<details> <summary>SimSiam</summary>

```
cd external_src/simsiam/
mkdir checkpoints && cd checkpoints
mkdir ImageNet && cd ImageNet
wget -O simsiam_bs512_ep100.pth.tar https://dl.fbaipublicfiles.com/simsiam/models/100ep/pretrain/checkpoint_0099.pth.tar
wget -O simsiam_bs256_ep100.pth.tar https://dl.fbaipublicfiles.com/simsiam/models/100ep-256bs/pretrain/checkpoint_0099.pth.tar
```
</details>

<details> <summary>Swav</summary>

```
cd external_src/swav/
mkdir checkpoints && cd checkpoints
mkdir ImageNet && cd ImageNet
wget -O swav_bs4096_ep800.pth.tar https://dl.fbaipublicfiles.com/deepcluster/swav_800ep_pretrain.pth.tar
wget -O swav_bs4096_ep400.pth.tar https://dl.fbaipublicfiles.com/deepcluster/swav_400ep_pretrain.pth.tar
wget -O swav_bs4096_ep200.pth.tar https://dl.fbaipublicfiles.com/deepcluster/swav_200ep_pretrain.pth.tar
wget -O swav_bs4096_ep100.pth.tar https://dl.fbaipublicfiles.com/deepcluster/swav_100ep_pretrain.pth.tar
wget -O swav_bs256_ep200.pth.tar https://dl.fbaipublicfiles.com/deepcluster/swav_200ep_bs256_pretrain.pth.tar
wget -O swav_bs256_ep400.pth.tar https://dl.fbaipublicfiles.com/deepcluster/swav_400ep_bs256_pretrain.pth.tar
```
</details>
