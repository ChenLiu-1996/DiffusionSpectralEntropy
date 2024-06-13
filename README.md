# Diffusion Spectral Entropy and Mutual Information
**Krishnaswamy Lab, Yale University**

[![Twitter](https://img.shields.io/twitter/follow/KrishnaswamyLab.svg?style=social&label=Follow)](https://twitter.com/KrishnaswamyLab)
[![Github Stars](https://img.shields.io/github/stars/ChenLiu-1996/DiffusionSpectralEntropy.svg?style=social&label=Stars)](https://github.com/ChenLiu-1996/DiffusionSpectralEntropy/)


This is the **official** implementation of

[**Assessing Neural Network Representations During Training Using Noise-Resilient Diffusion Spectral Entropy**](https://arxiv.org/abs/2312.04823)

<img src="assets/logos/Yale_logo.png" height="96"/> &emsp; <img src="assets/logos/Mila_logo.png" height="96"/> &emsp; <img src="assets/logos/MetaAI_logo.png" height="96"/>

## Announcement
**Due to certain internal policies, we removed the codebase from public access. However, for the benefit of future researchers, we hereby provide the DSE/DSMI implementations.**

## Citation
```
@inproceedings{DSE2023,
  title={Assessing Neural Network Representations During Training Using Noise-Resilient Diffusion Spectral Entropy},
  author={Liao, Danqi and Liu, Chen and Christensen, Ben and Tong, Alexander and Huguet, Guillaume and Wolf, Guy and Nickel, Maximilian and Adelstein, Ian and Krishnaswamy, Smita},
  booktitle={ICML 2023 Workshop on Topology, Algebra and Geometry in Machine Learning (TAG-ML)},
  year={2023},
}
@inproceedings{DSE2024,
  title={Assessing neural network representations during training using noise-resilient diffusion spectral entropy},
  author={Liao, Danqi and Liu, Chen and Christensen, Benjamin W and Tong, Alexander and Huguet, Guillaume and Wolf, Guy and Nickel, Maximilian and Adelstein, Ian and Krishnaswamy, Smita},
  booktitle={2024 58th Annual Conference on Information Sciences and Systems (CISS)},
  pages={1--6},
  year={2024},
  organization={IEEE}
}
```


## API: Your One-Stop Shop
Here we present the refactored and reorganized APIs for this project.

### Diffusion Spectral Entropy
[Go to function](./api/dse.py/#L7)
```
api > dse.py > diffusion_spectral_entropy
```

### Diffusion Spectral Mutual Information
[Go to function](./api/dsmi.py/#L7)
```
api > dsmi.py > diffusion_spectral_mutual_information
```

### Unit Tests for DSE and DSMI
You can directly run the following lines for built-in unit tests.
```
python dse.py
python dsmi.py
```

## Overview
> We proposed a framework to measure the **entropy** and **mutual information** in high dimensional data and thus applicable to modern neural networks.

We can measure, with respect to a given set of data samples, (1) the entropy of the neural representation at a specific layer and (2) the mutual information between a random variable (e.g., model input or output) and the neural representation at a specific layer.

Compared to the classic Shannon formulation using the binning method, e.g. as in the famous paper **_Deep Learning and the Information Bottleneck Principle_** [[PDF]](https://arxiv.org/abs/1503.02406) [[Github1]](https://github.com/stevenliuyi/information-bottleneck) [[Github2]](https://github.com/artemyk/ibsgd), our proposed method is more robust and expressive.

## Main Advantage
No binning and hence **no curse of dimensionality**. Therefore, **it works on modern deep neural networks** (e.g., ResNet-50), not just on toy models with double digit layer width. See "Limitations of the Classic Shannon Entropy and Mutual Information" in our paper for details.

<img src="assets/curse_of_dim.png" width="600">

## A One-Minute Explanation of the Methods
Conceptually, we build a data graph from the neural network representations of all data points in a dataset, and compute the diffusion matrix of the data graph. This matrix is a condensed representation of the diffusion geometry of the neural representation manifold. Our proposed **Diffusion Spectral Entropy (DSE)** and **Diffusion Spectral Mutual Information (DSMI)** can be computed from this diffusion matrix.

<img src="assets/procedure.png" width="600">

## Quick Flavors of the Results

### Definition
<img src="assets/def_DSE.png" width="400"> <img src="assets/def_DSMI.png" width="400">

### Theoretical Results
One major statement to make is that the proposed DSE and DSMI are "not conceptually the same as" the classic Shannon counterparts. They are defined differently and while they maintain the gist of "entropy" and "mutual information" measures, they have their own unique properties. For example, DSE is *more sensitive to the underlying dimension and structures (e.g., number of branches or clusters) than to the spread or noise in the data itself, which is contracted to the manifold by raising the diffusion operator to the power of $t$*.

In the theoretical results, we upper- and lower-bounded the proposed DSE and DSMI. More interestingly, we showed that if a data distribution originates as a single Gaussian blob but later evolves into $k$ distinct Gaussian blobs, the upper bound of the expected DSE will increase. This has implication for the training process of classification networks.

### Empirical Results
We first use toy experiments to showcase that DSE and DSMI "behave properly" as measures of entropy and mutual information. We also demonstrate they are more robust to high dimensions than the classic counterparts.

Then, we also look at how well DSE and DSMI behave at higher dimensions. In the figure below, we show how DSMI outperforms other mutual information estimators when the dimension is high. Besides, the runtime comparison shows DSMI scales better with respect to dimension.

<img src="assets/method_comparison.png" width="800">

</br>

Finally, it's time to put them in practice! We use DSE and DSMI to visualize the training dynamics of classification networks of 6 backbones (3 ConvNets and 3 Transformers) under 3 training conditions and 3 random seeds. We are evaluating the penultimate layer of the neural network --- the second-to-last layer where people believe embeds the rich representation of the data and are often used for visualization, linear-probing evaluation, etc.

<img src="assets/main_figure_DSE(Z).png" width="600">

DSE(Z) increasese during training. This happens for both generalizable training and overfitting. The former case coincides with our theoretical finding that DSE(Z) shall increase as the model learns to separate data representation into clusters.

<img src="assets/main_figure_DSMI(Z;Y).png" width="600">

DSMI(Z; Y) increases during generalizable training but stays stagnant during overfitting. This is very much expected.

<img src="assets/main_figure_DSMI(Z;X).png" width="600">

DSMI(Z; X) shows quite intriguing trends. On MNIST, it keeps increasing. On CIFAR-10 and STL-10, it peaks quickly and gradually decreases. Recall that IB [Tishby et al.] suggests that I(Z; X) shall decrease while [Saxe et al. ICLR'18] believes the opposite. We find that both of them could be correct since the trend we observe is dataset-dependent. One possibility is that MNIST features are too easy to learn (and perhaps the models all overfit?) --- and we leave this to future explorations.


## Utility Studies: How can we use DSE and DSMI?
One may ask, besides just peeking into the training dynamics of neural networks, how can we _REALLY_ use DSE and DSMI? Here comes the utility studies.

### Guiding network initialization with DSE
We sought to assess the effects of network initialization in terms of DSE. We were motivated by two observations: (1) the initial DSEs for different models are not always the same despite using the same method for random initialization; (2) if DSE starts low, it grows monotonically; if DSE starts high, it first decreases and then increases.

We found that if we initialize the convolutional layers with weights $\sim \mathcal{N}(0, \sigma)$, DSE $S_D(Z)$ is affected by $\sigma$. We then trained ResNet models with networks initialized at high ($\approx$ log(n)) versus low ($\approx 0$) DSE by setting $\sigma=0.1$ and $\sigma=0.01$, respectively. The training history suggests that initializing the network at a lower $S_D(Z)$ can improve the convergence speed and final performance. We believe this is because the high initial DSE from random initialization corresponds to an undesirable high-entropy state, which the network needs to get away from (causing the DSE decrease) before it migrates to the desirable high-entropy state (causing the DSE increase).

<img src="assets/compare-cifar10-supervised-resnet-ConvInitStd-1e-2-1e-1-seed1-2-3.png" width="300"> <img src="assets/compare-stl10-supervised-resnet-ConvInitStd-1e-2-1e-1-seed1-2-3.png" width="300">

### ImageNet cross-model correlation
By far, we have monitored DSE and DSMI **along the training process of the same model**. Now we will show how DSMI correlates with downstream classification accuracy **across many different pre-trained models**. The following result demonstrates the potential in using DSMI for pre-screening potentially competent models for your specialized dataset.

<img src="assets/vs_imagenet_acc.png" width="600">


## Reproducing Results in the ongoing submission.
Removed due to internal policies.

## Preparation

### Environment
We developed the codebase in a miniconda environment.
Tested on Python 3.9.13 + PyTorch 1.12.1.
How we created the conda environment:
**Some packages may no longer be required.**
```
conda create --name dse pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
conda activate dse
conda install -c anaconda scikit-image pillow matplotlib seaborn tqdm
python -m pip install -U giotto-tda
python -m pip install POT torch-optimizer
python -m pip install tinyimagenet
python -m pip install natsort
python -m pip install phate
python -m pip install DiffusionEMD
python -m pip install magic-impute
python -m pip install timm
python -m pip install pytorch-lightning
```



