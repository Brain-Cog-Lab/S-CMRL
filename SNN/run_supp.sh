#CUDA_VISIBLE_DEVICES=4 python train_snn.py --model AVspikformer --dataset CREMAD --epoch 100 --batch-size 128 --num-classes 6 --step 4 --modality audio-visual --cross-attn --attn-method SpatialTemporal --alpha 1.5 --contrastive --temperature 0.01 &
#PID5=$!;
#
#CUDA_VISIBLE_DEVICES=5 python train_snn.py --model AVspikformer --dataset CREMAD --epoch 100 --batch-size 128 --num-classes 6 --step 4 --modality audio-visual --cross-attn --attn-method SpatialTemporal --alpha 1.5 --contrastive --temperature 0.07 &
#PID6=$!;
#
#CUDA_VISIBLE_DEVICES=6 python train_snn.py --model AVspikformer --dataset CREMAD --epoch 100 --batch-size 128 --num-classes 6 --step 4 --modality audio-visual --cross-attn --attn-method SpatialTemporal --alpha 1.5 --contrastive --temperature 0.5 &
#PID7=$!;
#
#CUDA_VISIBLE_DEVICES=7 python train_snn.py --model AVspikformer --dataset CREMAD --epoch 100 --batch-size 128 --num-classes 6 --step 4 --modality audio-visual --cross-attn --attn-method SpatialTemporal --alpha 1.5 --contrastive --temperature 1.0 &
#PID8=$!;
#
#CUDA_VISIBLE_DEVICES=0 python train_snn.py --model AVspikformer --dataset CREMAD --epoch 100 --batch-size 128 --num-classes 6 --step 4 --modality audio-visual --cross-attn --attn-method SpatialTemporal --alpha 2.0&
#
#
#CUDA_VISIBLE_DEVICES=1 python train_snn.py --model AVspikformer --embed_dims 64 --dataset UrbanSound8K --epoch 100 --batch-size 128 --num-classes 10 --step 4 --modality audio-visual --cross-attn --attn-method SpatialTemporal --alpha 2.0&


CUDA_VISIBLE_DEVICES=0 python train_snn.py --model AVspikformer --embed_dims 64 --dataset UrbanSound8K --epoch 100 --batch-size 128 --num-classes 10 --step 4 --modality audio-visual --cross-attn --attn-method Spatial --alpha 1.5&
PID1=$!;

CUDA_VISIBLE_DEVICES=1 python train_snn.py --model AVspikformer --embed_dims 64 --dataset UrbanSound8K --epoch 100 --batch-size 128 --num-classes 10 --step 4 --modality audio-visual --cross-attn --attn-method Temporal --alpha 1.5&
PID2=$!;