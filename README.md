# Manifold Topology

### The repository is structured in the following manner.
```
mondi-image-gen
    ├── external_src
    |   └── `${MODEL_NAME}/...`: git repos for external models
    ├── setup: Instructions for proper setup of environments?
    ├── src
    |   ├── configs
    |   |   └── `${MODEL_NAME}/xxx.yaml`: config files for external models (1 folder per model)
    |   ├── unit_test
    |   |   └── `test_run_${MODEL_NAME}.py`: scripts to check validity of `${MODEL_NAME}_model.py`.
    |   ├── `investigate.py`: centralized place for our contribution.
    |   └── `${MODEL_NAME}_model.py`: wrapper code to interface with external models.
    └── utils: folder of utility functions.
```


### Preparing pretrained weights of external models.
#### MoCo
```
cd external_src/moco/
mkdir checkpoints && cd checkpoints
mkdir ImageNet && cd ImageNet
wget https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v1_200ep/moco_v1_200ep_pretrain.pth.tar
wget https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_200ep/moco_v2_200ep_pretrain.pth.tar
wget https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_800ep/moco_v2_800ep_pretrain.pth.tar
```

#### SimSiam
```
cd external_src/simsiam/
mkdir checkpoints && cd checkpoints
mkdir ImageNet && cd ImageNet
wget https://dl.fbaipublicfiles.com/simsiam/models/100ep/pretrain/checkpoint_0099.pth.tar
mv checkpoint_0099.pth.tar simsiam_bs512_ep100.pth.tar
wget https://dl.fbaipublicfiles.com/simsiam/models/100ep-256bs/pretrain/checkpoint_0099.pth.tar
mv checkpoint_0099.pth.tar simsiam_bs256_ep100.pth.tar
```
