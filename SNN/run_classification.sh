#CUDA_VISIBLE_DEVICES=0 python train_snn_cl.py --model spikformer --dataset CREMAD --epoch 100 --batch-size 128 --num-classes 6 --step 4 --modality audio --class_num_per_step 6 &
#PID1=$!;
#
#CUDA_VISIBLE_DEVICES=1 python train_snn_cl.py --model spikformer --dataset CREMAD --epoch 100 --batch-size 128 --num-classes 6 --step 4 --modality visual --class_num_per_step 6 &
#PID2=$!;
#
#CUDA_VISIBLE_DEVICES=2 python train_snn_cl.py --model AVspikformer --dataset CREMAD --epoch 100 --batch-size 128 --num-classes 6 --step 4 --modality audio-visual --class_num_per_step 6 &
#PID3=$!;
#
#CUDA_VISIBLE_DEVICES=3 python train_snn_cl.py --model AVspikformer --dataset CREMAD --epoch 100 --batch-size 128 --num-classes 6 --step 4 --modality audio-visual --class_num_per_step 6 --cross-attn --attn-method Spatial &
#PID4=$!;
#
#CUDA_VISIBLE_DEVICES=4 python train_snn_cl.py --model AVspikformer --dataset CREMAD --epoch 100 --batch-size 128 --num-classes 6 --step 4 --modality audio-visual --class_num_per_step 6 --cross-attn --attn-method Temporal &
#PID5=$!;
#
#CUDA_VISIBLE_DEVICES=5 python train_snn_cl.py --model AVspikformer --dataset CREMAD --epoch 100 --batch-size 128 --num-classes 6 --step 4 --modality audio-visual --class_num_per_step 6 --cross-attn --attn-method SpatialTemporal &
#PID6=$!;
#
#CUDA_VISIBLE_DEVICES=6 python train_snn_cl.py --model AVspikformer --dataset CREMAD --epoch 100 --batch-size 128 --num-classes 6 --step 4 --modality audio-visual --class_num_per_step 6 --cross-attn --attn-method Spatial --av-attn &
#PID7=$!;
#
#CUDA_VISIBLE_DEVICES=7 python train_snn_cl.py --model AVspikformer --dataset CREMAD --epoch 100 --batch-size 128 --num-classes 6 --step 4 --modality audio-visual --class_num_per_step 6 --cross-attn --attn-method SpatialTemporal --av-attn &
#PID8=$!;
#
#wait ${PID1} && wait ${PID2} && wait ${PID3} && wait ${PID4} && wait ${PID5} && wait ${PID6} && wait ${PID7} && wait ${PID8}


#CUDA_VISIBLE_DEVICES=0 python train_snn_cl.py --model spikformer --embed_dims 64 --dataset UrbanSound8K --epoch 100 --batch-size 128 --num-classes 10 --step 2 --modality audio --class_num_per_step 10 &
#PID1=$!;
#
#CUDA_VISIBLE_DEVICES=1 python train_snn_cl.py --model spikformer --embed_dims 64 --dataset UrbanSound8K --epoch 100 --batch-size 128 --num-classes 10 --step 2 --modality visual --class_num_per_step 10 &
#PID2=$!;
#
#CUDA_VISIBLE_DEVICES=2 python train_snn_cl.py --model AVspikformer --embed_dims 64 --dataset UrbanSound8K --epoch 100 --batch-size 128 --num-classes 10 --step 2 --modality audio-visual --class_num_per_step 10 &
#PID3=$!;
#
#CUDA_VISIBLE_DEVICES=3 python train_snn_cl.py --model AVspikformer --embed_dims 64 --dataset UrbanSound8K --epoch 100 --batch-size 128 --num-classes 10 --step 2 --modality audio-visual --class_num_per_step 10 --cross-attn --attn-method Spatial &
#PID4=$!;
#
#CUDA_VISIBLE_DEVICES=4 python train_snn_cl.py --model AVspikformer --embed_dims 64 --dataset UrbanSound8K --epoch 100 --batch-size 128 --num-classes 10 --step 2 --modality audio-visual --class_num_per_step 10 --cross-attn --attn-method Temporal &
#PID5=$!;
#
#CUDA_VISIBLE_DEVICES=5 python train_snn_cl.py --model AVspikformer --embed_dims 64 --dataset UrbanSound8K --epoch 100 --batch-size 128 --num-classes 10 --step 2 --modality audio-visual --class_num_per_step 10 --cross-attn --attn-method SpatialTemporal &
#PID6=$!;
#
#CUDA_VISIBLE_DEVICES=6 python train_snn_cl.py --model AVspikformer --embed_dims 64 --dataset UrbanSound8K --epoch 100 --batch-size 128 --num-classes 10 --step 2 --modality audio-visual --class_num_per_step 10 --cross-attn --attn-method Spatial --av-attn &
#PID7=$!;
#
#CUDA_VISIBLE_DEVICES=7 python train_snn_cl.py --model AVspikformer --embed_dims 64 --dataset UrbanSound8K --epoch 100 --batch-size 128 --num-classes 10 --step 2 --modality audio-visual --class_num_per_step 10 --cross-attn --attn-method SpatialTemporal --av-attn &
#PID8=$!;


CUDA_VISIBLE_DEVICES=0 python train_snn_cl.py --model AVspikformer --dataset CREMAD --epoch 100 --batch-size 128 --num-classes 6 --step 4 --modality audio-visual --class_num_per_step 6 --cross-attn --attn-method SpatialTemporal --alpha 0.0&
PID1=$!;

CUDA_VISIBLE_DEVICES=1 python train_snn_cl.py --model AVspikformer --dataset CREMAD --epoch 100 --batch-size 128 --num-classes 6 --step 4 --modality audio-visual --class_num_per_step 6 --cross-attn --attn-method SpatialTemporal --alpha 0.5&
PID2=$!;

CUDA_VISIBLE_DEVICES=2 python train_snn_cl.py --model AVspikformer --dataset CREMAD --epoch 100 --batch-size 128 --num-classes 6 --step 4 --modality audio-visual --class_num_per_step 6 --cross-attn --attn-method SpatialTemporal --alpha 1.0&
PID3=$!;

CUDA_VISIBLE_DEVICES=3 python train_snn_cl.py --model AVspikformer --dataset CREMAD --epoch 100 --batch-size 128 --num-classes 6 --step 4 --modality audio-visual --class_num_per_step 6 --cross-attn --attn-method SpatialTemporal --alpha 1.5&
PID4=$!;

CUDA_VISIBLE_DEVICES=4 python train_snn_cl.py --model AVspikformer --dataset CREMAD --epoch 100 --batch-size 128 --num-classes 6 --step 4 --modality audio-visual --class_num_per_step 6 --cross-attn --attn-method SpatialTemporal --alpha 2.0&
PID5=$!;

CUDA_VISIBLE_DEVICES=5 python train_snn_cl.py --model AVspikformer --dataset CREMAD --epoch 100 --batch-size 128 --num-classes 6 --step 4 --modality audio-visual --class_num_per_step 6 --cross-attn --attn-method SpatialTemporal --alpha 3.0&
PID6=$!;

CUDA_VISIBLE_DEVICES=6 python train_snn_cl.py --model AVspikformer --dataset CREMAD --epoch 100 --batch-size 128 --num-classes 6 --step 4 --modality audio-visual --class_num_per_step 6 --cross-attn --attn-method SpatialTemporal --alpha 5.0&
PID7=$!;

CUDA_VISIBLE_DEVICES=7 python train_snn_cl.py --model AVspikformer --dataset CREMAD --epoch 100 --batch-size 128 --num-classes 6 --step 4 --modality audio-visual --class_num_per_step 6 --cross-attn --attn-method SpatialTemporal --alpha 10.0&
PID8=$!;

wait ${PID1} && wait ${PID2} && wait ${PID3} && wait ${PID4} && wait ${PID5} && wait ${PID6} && wait ${PID7} && wait ${PID8}


CUDA_VISIBLE_DEVICES=0 python train_snn_cl.py --model AVspikformer --embed_dims 64 --dataset UrbanSound8K --epoch 100 --batch-size 128 --num-classes 10 --step 4 --modality audio-visual --class_num_per_step 10 --cross-attn --attn-method SpatialTemporal --alpha 0.0 &
PID1=$!;

CUDA_VISIBLE_DEVICES=1 python train_snn_cl.py --model AVspikformer --embed_dims 64 --dataset UrbanSound8K --epoch 100 --batch-size 128 --num-classes 10 --step 4 --modality audio-visual --class_num_per_step 10 --cross-attn --attn-method SpatialTemporal --alpha 0.3 &
PID2=$!;

CUDA_VISIBLE_DEVICES=2 python train_snn_cl.py --model AVspikformer --embed_dims 64 --dataset UrbanSound8K --epoch 100 --batch-size 128 --num-classes 10 --step 4 --modality audio-visual --class_num_per_step 10 --cross-attn --attn-method SpatialTemporal --alpha 0.5 &
PID3=$!;

CUDA_VISIBLE_DEVICES=3 python train_snn_cl.py --model AVspikformer --embed_dims 64 --dataset UrbanSound8K --epoch 100 --batch-size 128 --num-classes 10 --step 4 --modality audio-visual --class_num_per_step 10 --cross-attn --attn-method SpatialTemporal --alpha 0.6 &
PID4=$!;

CUDA_VISIBLE_DEVICES=4 python train_snn_cl.py --model AVspikformer --embed_dims 64 --dataset UrbanSound8K --epoch 100 --batch-size 128 --num-classes 10 --step 4 --modality audio-visual --class_num_per_step 10 --cross-attn --attn-method SpatialTemporal --alpha 0.8 &
PID5=$!;

CUDA_VISIBLE_DEVICES=5 python train_snn_cl.py --model AVspikformer --embed_dims 64 --dataset UrbanSound8K --epoch 100 --batch-size 128 --num-classes 10 --step 4 --modality audio-visual --class_num_per_step 10 --cross-attn --attn-method SpatialTemporal --alpha 1.0 &
PID6=$!;

CUDA_VISIBLE_DEVICES=6 python train_snn_cl.py --model AVspikformer --embed_dims 64 --dataset UrbanSound8K --epoch 100 --batch-size 128 --num-classes 10 --step 4 --modality audio-visual --class_num_per_step 10 --cross-attn --attn-method SpatialTemporal --alpha 1.5 &
PID7=$!;

CUDA_VISIBLE_DEVICES=7 python train_snn_cl.py --model AVspikformer --embed_dims 64 --dataset UrbanSound8K --epoch 100 --batch-size 128 --num-classes 10 --step 4 --modality audio-visual --class_num_per_step 10 --cross-attn --attn-method SpatialTemporal --alpha 2.0 &
PID8=$!;

wait ${PID1} && wait ${PID2} && wait ${PID3} && wait ${PID4} && wait ${PID5} && wait ${PID6} && wait ${PID7} && wait ${PID8}