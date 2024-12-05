# Enhancing Audio-Visual Learning: Incorporating Multimodal Cue Fusion in Transformer-Based Spiking Neural Networks
Here is the PyTorch implementation of our paper. 

**Paper Title: "Enhancing Audio-Visual Learning: Incorporating Multimodal Cue Fusion in Transformer-Based Spiking Neural Networks"**

**Authors: Xiang He\*, Dongcheng Zhao\*, Yiting Dong, Guobin Shen,  Xin Yang, Yi Zeng**

\[[arxiv]()\] \[[paper]()\] \[[code](https://github.com/Brain-Cog-Lab/MCF)\]

##### 1. CREMAD

```
CUDA_VISIBLE_DEVICES=0 python train_snn_cl.py --model spikformer --dataset CREMAD --epoch 100 --batch-size 128 --num-classes 6 --step 4 --modality audio --class_num_per_step 6
```



```
CUDA_VISIBLE_DEVICES=1 python train_snn_cl.py --model spikformer --dataset CREMAD --epoch 100 --batch-size 128 --num-classes 6 --step 4 --modality visual --class_num_per_step 6
```



```
CUDA_VISIBLE_DEVICES=2 python train_snn_cl.py --model AVspikformer --dataset CREMAD --epoch 100 --batch-size 128 --num-classes 6 --step 4 --modality audio-visual --class_num_per_step 6
```



```
CUDA_VISIBLE_DEVICES=3 python train_snn_cl.py --model AVspikformer --dataset CREMAD --epoch 100 --batch-size 128 --num-classes 6 --step 4 --modality audio-visual --class_num_per_step 6 --cross-attn --attn-method Spatial
```



```
CUDA_VISIBLE_DEVICES=4 python train_snn_cl.py --model AVspikformer --dataset CREMAD --epoch 100 --batch-size 128 --num-classes 6 --step 4 --modality audio-visual --class_num_per_step 6 --cross-attn --attn-method Temporal
```



```
CUDA_VISIBLE_DEVICES=5 python train_snn_cl.py --model AVspikformer --dataset CREMAD --epoch 100 --batch-size 128 --num-classes 6 --step 4 --modality audio-visual --class_num_per_step 6 --cross-attn --attn-method SpatialTemporal --alpha 0.0
```



```
CUDA_VISIBLE_DEVICES=6 python train_snn_cl.py --model AVspikformer --dataset CREMAD --epoch 100 --batch-size 128 --num-classes 6 --step 4 --modality audio-visual --class_num_per_step 6 --cross-attn --attn-method SpatialTemporal --alpha 1.0
```



```
CUDA_VISIBLE_DEVICES=7 python train_snn_cl.py --model AVspikformer --dataset CREMAD --epoch 100 --batch-size 128 --num-classes 6 --step 4 --modality audio-visual --class_num_per_step 6 --cross-attn --attn-method SpatialTemporal --alpha 1.0 --contrastive
```

##### 2. UrbanSound8K




##### 3. AVmnistdvs

```
CUDA_VISIBLE_DEVICES=0 python train_snn_cl.py --model spikformer --dataset AVmnistdvs --epoch 20 --batch-size 32 --num-classes 10 --step 4 --modality audio --shallow-sps --event-size 28
```



```\
CUDA_VISIBLE_DEVICES=0 python train_snn_cl.py --model spikformer --dataset AVmnistdvs --epoch 20 --batch-size 32 --num-classes 10 --step 4 --modality visual --shallow-sps --event-size 28
```



```
CUDA_VISIBLE_DEVICES=0 python train_snn_cl.py --model AVspikformer --dataset AVmnistdvs --epoch 20 --batch-size 32 --num-classes 10 --step 4 --modality audio-visual --shallow-sps --event-size 28
```



```
CUDA_VISIBLE_DEVICES=0 python train_snn_cl.py --model AVspikformer --dataset AVmnistdvs --epoch 20 --batch-size 32 --num-classes 10 --step 4 --modality audio-visual --shallow-sps --event-size 28 --cross-attn
```



## Datasets

CREMA-D datasets：[CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D)

UrbanSound8K-AV datasets: [UrbanSound8K-AV](https://github.com/Guo-Lingyue/SMMT)




## Acknowledgements

The UrbanSound8K-AV datasets used can be found in [SMMT](https://github.com/Guo-Lingyue/SMMT), thanks to their excellent work!  The SNN implementation is based on [Brain-Cog](https://github.com/BrainCog-X/Brain-Cog).  

If you are confused about using it or have other feedback and comments, please feel free to contact us via hexiang2021@ia.ac.cn. Have a good day!