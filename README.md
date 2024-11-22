# Audio-Visual Prompts in Class-Incremental Learning- Pytorch
Here is the PyTorch implementation of our paper. （Prompt middle version）

**Paper Title: "Better Multimodal Cue Fusion: Incorporating Audio-Visual Prompts in Class-Incremental Learning"**



### Continual Learning

##### 1. UrbanSound8K

```
CUDA_VISIBLE_DEVICES=0 python train_snn_cl.py --model spikformer --dataset UrbanSound8K --epoch 100 --batch-size 64 --num-classes 10 --step 4 --modality audio
```



```
CUDA_VISIBLE_DEVICES=1 python train_snn_cl.py --model spikformer --dataset UrbanSound8K --epoch 100 --batch-size 64 --num-classes 10 --step 4 --modality visual
```



```
CUDA_VISIBLE_DEVICES=2 python train_snn_cl.py --model AVspikformer --dataset UrbanSound8K --epoch 100 --batch-size 64 --num-classes 10 --step 4 --modality audio-visual
```



```
CUDA_VISIBLE_DEVICES=3 python train_snn_cl.py --model AVspikformer --dataset UrbanSound8K --epoch 100 --batch-size 64 --num-classes 10 --step 4 --modality audio-visual --cross-attn
```



```
CUDA_VISIBLE_DEVICES=4 python train_snn_cl.py --model AVspikformer --dataset UrbanSound8K --epoch 100 --batch-size 64 --num-classes 10 --step 4 --modality audio-visual --cross-attn --av-attn
```



##### 2. AvCifar10

```
CUDA_VISIBLE_DEVICES=5 python train_snn_cl.py --model spikformer --dataset AvCifar10 --epoch 100 --batch-size 64 --num-classes 10 --step 4 --modality audio --shallow-sps --event-size 32
```



```
CUDA_VISIBLE_DEVICES=6 python train_snn_cl.py --model spikformer --dataset AvCifar10 --epoch 100 --batch-size 64 --num-classes 10 --step 4 --modality visual --shallow-sps --event-size 32
```



```
CUDA_VISIBLE_DEVICES=7 python train_snn_cl.py --model AVspikformer --dataset AvCifar10 --epoch 100 --batch-size 64 --num-classes 10 --step 4 --modality audio-visual --shallow-sps --event-size 32
```



```
CUDA_VISIBLE_DEVICES=0 python train_snn_cl.py --model AVspikformer --dataset AvCifar10 --epoch 100 --batch-size 64 --num-classes 10 --step 4 --modality audio-visual --shallow-sps --event-size 32 --cross-attn
```



```
CUDA_VISIBLE_DEVICES=0 python train_snn_cl.py --model AVspikformer --dataset AvCifar10 --epoch 100 --batch-size 64 --num-classes 10 --step 4 --modality audio-visual --shallow-sps --event-size 32 --cross-attn --av-attn
```





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






## Acknowledgements

Our code was built based on [https://github.com/weiguoPian/AV-CIL_ICCV2023](https://github.com/weiguoPian/AV-CIL_ICCV2023), and the audio-visual datasets used can all be found in that project as well, thanks to their excellent work!  The SNN implementation is based on [Brain-Cog](https://github.com/BrainCog-X/Brain-Cog).  

If you are confused about using it or have other feedback and comments, please feel free to contact us via hexiang2021@ia.ac.cn. Have a good day!