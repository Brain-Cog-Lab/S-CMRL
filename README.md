# Advancing Audio-Visual Spiking Neural Networks via Semantic-Alignment  Cross-Modal Residual Learning
Here is the PyTorch implementation of our paper. 

**Paper Title: "Advancing Audio-Visual Spiking Neural Networks via Semantic-Alignment  Cross-Modal Residual Learning"**

**Authors: Xiang He\*, Dongcheng Zhao\*, Yiting Dong, Guobin Shen,  Xin Yang, Yi Zeng**

\[[arxiv](https://arxiv.org/abs/2502.12488)\] \[[paper]()\] \[[code](https://github.com/Brain-Cog-Lab/S-CMRL)\]



## Method

We construct a semantic-alignment cross-modal residual learning framework for multimodal SNNs. This framework provides an efficient feature fusion strategy and achieves state-of-the-art performance on three public datasets, demonstrating superior accuracy and robustness compared to existing methods. 

<img src="./figs/method.jpg" alt="method" style="zoom:80%;" />

Comparison of S-CMRL with state-of-the-art methods on three datasets:

<img src="./figs/results.jpg" alt="results" style="zoom:80%;" />



## Training Script

All experimental scripts can be found in [run_classification.sh](./SNN/run_classification.sh)

A sample script for our method on the CRMEA-D dataset is as follows：

```
CUDA_VISIBLE_DEVICES=0 python train_snn.py --model AVspikformer --dataset CREMAD --epoch 100 --batch-size 128 --num-classes 6 --step 4 --modality audio-visual --cross-attn --attn-method SpatialTemporal --alpha 1.5 --contrastive --temperature 0.07
```



The well-trained model weights and training logs are available [here]() to reproduce the results from the paper.



## Datasets

CREMA-D datasets：[CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D)

UrbanSound8K-AV datasets: [UrbanSound8K-AV](https://github.com/Guo-Lingyue/SMMT)



## Citation
If our paper is useful for your research, please consider citing it:
```
@misc{he2025enhancingaudiovisualspikingneural,
      title={Enhancing Audio-Visual Spiking Neural Networks through Semantic-Alignment and Cross-Modal Residual Learning}, 
      author={Xiang He and Dongcheng Zhao and Yiting Dong and Guobin Shen and Xin Yang and Yi Zeng},
      year={2025},
      eprint={2502.12488},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2502.12488}, 
}
```





## Acknowledgements

The UrbanSound8K-AV datasets used can be found in [SMMT](https://github.com/Guo-Lingyue/SMMT), thanks to their excellent work!  The SNN implementation is based on [Brain-Cog](https://github.com/BrainCog-X/Brain-Cog).  

If you are confused about using it or have other feedback and comments, please feel free to contact us via hexiang2021@ia.ac.cn. Have a good day!
