

# DS-AGC







# Self-Supervised Graph Convolution Autoencoder for Social Networks Clustering

- This is repository contains the code for the paper (IEEE TRANSACTIONS ON COMPUTATIONAL SOCIAL SYSTEMS  , 2022)
- Chen, C., Lu, H. Self-Supervised Graph Convolution Autoencoder for Social Networks Clustering . *(2022). 



# Abstract:

In recent years, graph-based deep learning algorithms have attracted widespread attention. Still, most of the current graph neural networks are based on supervised learning or semi-supervised learning, which often relies on the actual labels given by the samples as auxiliary information, thus limiting the robustness and generalization ability of the model. To solve this problem, we propose a neural network model based on the selfsupervised graph convolutional autoencoder structure and use it for social networks clustering. We divide the proposed model into two parts: a pretext task and a downstream task. In the pretext task, the model generates pseudo label by graph attention autoencoder structure, then adopts the proposed reliable sample filtering mechanism to gain a high confidential sample. This sample and corresponding pseudo label are selected as input of downstream task for helping downstream task to finish learning task. The model obtains the predictive labels of the sample from the downstream task. This model does not depend on the actual label sample for training and uses the pseudo labels produced by self-supervised learning to improve the clustering performance. We apply the proposed model to three commonly used public social network datasets to test its performance. Compared with the presented unsupervised clustering algorithms and other deep graph neural network algorithms, various metrics have shown that the proposed model consistently outperforms the state-ofthe-art.  



# Platform

This code was developed and tested with:

```python
networkx == 2.6.3
Pillow==8.4.0
scikit-learn==0.22
torch==1.7.0+cu101
torchvision==0.8.1+cu101
torch-cluster==1.5.8
torch-geometric==2.0.2
torch-scatter==2.0.5
torch-sparse==0.6.8
torch-spline-conv==1.2.0
```



# citation

if you use this code for your research, please cite our paper:

@article{DS-AGC,
title={ Self-Supervised Graph Convolution Autoencoder for Social Networks Clustering },
author={Chao Chen, Hu Lu, Haotian Hong, Hai Wang, Shaohua Wan   },
journal={IEEE TRANSACTIONS ON COMPUTATIONAL SOCIAL SYSTEMS},
year={2022},
}







