# Advancing Audio-Visual Transformer-Based Spiking Neural Networks via Semantic-Aware Cross-Modal Residual Learning
Here is the PyTorch implementation of our paper. 

**Paper Title: "Advancing Audio-Visual Transformer-Based Spiking Neural Networks via Semantic-Aware Cross-Modal Residual Learning"**

**Authors: Xiang He\*, Dongcheng Zhao\*, Yiting Dong, Guobin Shen,  Xin Yang, Yi Zeng**

\[[arxiv]()\] \[[paper]()\] \[[code](https://github.com/Brain-Cog-Lab/MCF)\]



## Method

We construct a semantic-aware cross-modal residual learning framework, comprising a *cross-modal complementary spatiotemporal spiking attention mechanism* and a *semantic-enhanced optimization mechanism*, which provides an efficient feature fusion method for multimodal spiking neural networks. 

![method](./figs/method.jpg)



## Training Script

All experimental scripts can be found in `[run_classification.sh](./SNN/run_classification.sh)`

A sample script for our method on the CRMEA-D dataset is as follows：

```
CUDA_VISIBLE_DEVICES=0 python train_snn.py --model AVspikformer --dataset CREMAD --epoch 100 --batch-size 128 --num-classes 6 --step 4 --modality audio-visual --cross-attn --attn-method SpatialTemporal --alpha 1.0 --contrastive
```



## Datasets

CREMA-D datasets：[CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D)

UrbanSound8K-AV datasets: [UrbanSound8K-AV](https://github.com/Guo-Lingyue/SMMT)



## Citation
If our paper is useful for your research, please consider citing it:
```
arxiv here
```





## Acknowledgements

The UrbanSound8K-AV datasets used can be found in [SMMT](https://github.com/Guo-Lingyue/SMMT), thanks to their excellent work!  The SNN implementation is based on [Brain-Cog](https://github.com/BrainCog-X/Brain-Cog).  

If you are confused about using it or have other feedback and comments, please feel free to contact us via hexiang2021@ia.ac.cn. Have a good day!