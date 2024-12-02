###---------------- UrbanSound8K --------------#
##CUDA_VISIBLE_DEVICES=0 python train_snn_cl.py --model spikformer --dataset UrbanSound8K --epoch 100 --batch-size 64 --num-classes 10 --step 4 --modality audio&
##PID1=$!;
##
##CUDA_VISIBLE_DEVICES=1 python train_snn_cl.py --model spikformer --dataset UrbanSound8K --epoch 100 --batch-size 64 --num-classes 10 --step 4 --modality visual&
##PID2=$!;
##
##CUDA_VISIBLE_DEVICES=2 python train_snn_cl.py --model AVspikformer --dataset UrbanSound8K --epoch 100 --batch-size 64 --num-classes 10 --step 4 --modality audio-visual&
##PID3=$!;
##
##CUDA_VISIBLE_DEVICES=3 python train_snn_cl.py --model AVspikformer --dataset UrbanSound8K --epoch 100 --batch-size 64 --num-classes 10 --step 4 --modality audio-visual --cross-attn &
##PID4=$!;
##
##CUDA_VISIBLE_DEVICES=4 python train_snn_cl.py --model AVspikformer --dataset UrbanSound8K --epoch 100 --batch-size 64 --num-classes 10 --step 4 --modality audio-visual --cross-attn --av-attn &
##PID5=$!;
##
##CUDA_VISIBLE_DEVICES=5 python train_snn_cl.py --model AVspikformer --dataset UrbanSound8K --epoch 100 --batch-size 64 --num-classes 10 --step 4 --modality audio-visual --contrastive &
##PID6=$!;
##
##CUDA_VISIBLE_DEVICES=6 python train_snn_cl.py --model AVspikformer --dataset UrbanSound8K --epoch 100 --batch-size 64 --num-classes 10 --step 4 --modality audio-visual --cross-attn --contrastive&
##PID7=$!;
##
##CUDA_VISIBLE_DEVICES=7 python train_snn_cl.py --model AVspikformer --dataset UrbanSound8K --epoch 100 --batch-size 64 --num-classes 10 --step 4 --modality audio-visual --cross-attn --av-attn --contrastive&
##PID8=$!;
##
##wait ${PID1} && wait ${PID2} && wait ${PID3} && wait ${PID4} && wait ${PID5} && wait ${PID6} && wait ${PID7} && wait ${PID8}
#
##---------------- CREMAD --------------#
##CUDA_VISIBLE_DEVICES=0 python train_snn_cl.py --model spikformer --dataset CREMAD --epoch 100 --batch-size 64 --num-classes 6 --step 4 --modality audio&
##PID1=$!;
##
##CUDA_VISIBLE_DEVICES=1 python train_snn_cl.py --model spikformer --dataset CREMAD --epoch 100 --batch-size 64 --num-classes 6 --step 4 --modality visual&
##PID2=$!;
##
##CUDA_VISIBLE_DEVICES=2 python train_snn_cl.py --model AVspikformer --dataset CREMAD --epoch 100 --batch-size 64 --num-classes 6 --step 4 --modality audio-visual&
##PID3=$!;
#
#CUDA_VISIBLE_DEVICES=3 python train_snn_cl.py --model AVspikformer --dataset CREMAD --epoch 100 --batch-size 64 --num-classes 6 --step 4 --modality audio-visual --cross-attn&
#PID4=$!;
#
#CUDA_VISIBLE_DEVICES=4 python train_snn_cl.py --model AVspikformer --dataset CREMAD --epoch 100 --batch-size 64 --num-classes 6 --step 4 --modality audio-visual --cross-attn --av-attn&
#PID5=$!;
#
#CUDA_VISIBLE_DEVICES=5 python train_snn_cl.py --model AVspikformer --dataset CREMAD --epoch 100 --batch-size 64 --num-classes 6 --step 4 --modality audio-visual --contrastive&
#PID6=$!;
#
#CUDA_VISIBLE_DEVICES=6 python train_snn_cl.py --model AVspikformer --dataset CREMAD --epoch 100 --batch-size 64 --num-classes 6 --step 4 --modality audio-visual --cross-attn --contrastive&
#PID7=$!;
#
#CUDA_VISIBLE_DEVICES=7 python train_snn_cl.py --model AVspikformer --dataset CREMAD --epoch 100 --batch-size 64 --num-classes 6 --step 4 --modality audio-visual --cross-attn --av-attn --contrastive&
#PID8=$!;
#
##wait ${PID1} && wait ${PID2} && wait ${PID3} && wait ${PID4} && wait ${PID5} && wait ${PID6} && wait ${PID7} && wait ${PID8}
##
###---------------- AvCifar10 --------------#
##CUDA_VISIBLE_DEVICES=0 python train_snn_cl.py --model spikformer --dataset AvCifar10 --epoch 100 --batch-size 64 --num-classes 10 --step 4 --modality audio --shallow-sps&
##PID1=$!;
##
##CUDA_VISIBLE_DEVICES=1 python train_snn_cl.py --model spikformer --dataset AvCifar10 --epoch 100 --batch-size 64 --num-classes 10 --step 4 --modality visual --shallow-sps&
##PID2=$!;
##
##CUDA_VISIBLE_DEVICES=2 python train_snn_cl.py --model AVspikformer --dataset AvCifar10 --epoch 100 --batch-size 64 --num-classes 10 --step 4 --modality audio-visual --shallow-sps&
##PID3=$!;
##
##CUDA_VISIBLE_DEVICES=3 python train_snn_cl.py --model AVspikformer --dataset AvCifar10 --epoch 100 --batch-size 64 --num-classes 10 --step 4 --modality audio-visual --shallow-sps --cross-attn&
##PID4=$!;
##
##CUDA_VISIBLE_DEVICES=4 python train_snn_cl.py --model AVspikformer --dataset AvCifar10 --epoch 100 --batch-size 64 --num-classes 10 --step 4 --modality audio-visual --shallow-sps --cross-attn --av-attn&
##PID5=$!;
##
##CUDA_VISIBLE_DEVICES=5 python train_snn_cl.py --model AVspikformer --dataset AvCifar10 --epoch 100 --batch-size 64 --num-classes 10 --step 4 --modality audio-visual --shallow-sps --contrastive&
##PID6=$!;
##
##CUDA_VISIBLE_DEVICES=6 python train_snn_cl.py --model AVspikformer --dataset AvCifar10 --epoch 100 --batch-size 64 --num-classes 10 --step 4 --modality audio-visual --shallow-sps --cross-attn --contrastive&
##PID7=$!;
##
##CUDA_VISIBLE_DEVICES=7 python train_snn_cl.py --model AVspikformer --dataset AvCifar10 --epoch 100 --batch-size 64 --num-classes 10 --step 4 --modality audio-visual --shallow-sps --cross-attn --av-attn --contrastive&
##PID8=$!;
##
##wait ${PID1} && wait ${PID2} && wait ${PID3} && wait ${PID4} && wait ${PID5} && wait ${PID6} && wait ${PID7} && wait ${PID8}
#
##
##CUDA_VISIBLE_DEVICES=2 python train_snn_cl.py --model spikformer --dataset AVmnistdvs --epoch 100 --batch-size 64 --num-classes 10 --step 4 --modality audio --shallow-sps --event-size 28 &
##PID3=$!;
##
##CUDA_VISIBLE_DEVICES=3 python train_snn_cl.py --model spikformer --dataset AVmnistdvs --epoch 100 --batch-size 64 --num-classes 10 --step 4 --modality visual --shallow-sps --event-size 28 &
##PID4=$!;
##
##CUDA_VISIBLE_DEVICES=4 python train_snn_cl.py --model AVspikformer --dataset AVmnistdvs --epoch 100 --batch-size 64 --num-classes 10 --step 4 --modality audio-visual --shallow-sps --event-size 28 &
##PID5=$!;
##
##CUDA_VISIBLE_DEVICES=5 python train_snn_cl.py --model AVspikformer --dataset AVmnistdvs --epoch 100 --batch-size 64 --num-classes 10 --step 4 --modality audio-visual --shallow-sps --event-size 28 --cross-attn &
##PID6=$!;
##
##CUDA_VISIBLE_DEVICES=6 python train_snn_cl.py --model AVspikformer --dataset AVmnistdvs --epoch 100 --batch-size 64 --num-classes 10 --step 4 --modality audio-visual --shallow-sps --event-size 28 --cross-attn --av-attn &
##PID7=$!;
#
#
#
#



CUDA_VISIBLE_DEVICES=0 python train_snn_cl.py --model spikformer --dataset CREMAD --epoch 100 --batch-size 64 --num-classes 6 --step 4 --modality audio &
PID1=$!;

CUDA_VISIBLE_DEVICES=1 python train_snn_cl.py --model spikformer --dataset CREMAD --epoch 100 --batch-size 64 --num-classes 6 --step 4 --modality visual&
PID2=$!;

CUDA_VISIBLE_DEVICES=2 python train_snn_cl.py --model AVspikformer --dataset CREMAD --epoch 100 --batch-size 64 --num-classes 6 --step 4 --modality audio-visual &
PID3=$!;

CUDA_VISIBLE_DEVICES=3 python train_snn_cl.py --model AVspikformer --dataset CREMAD --epoch 100 --batch-size 64 --num-classes 6 --step 4 --modality audio-visual --cross-attn --attn-method Spatial &
PID4=$!;

CUDA_VISIBLE_DEVICES=4 python train_snn_cl.py --model AVspikformer --dataset CREMAD --epoch 100 --batch-size 64 --num-classes 6 --step 4 --modality audio-visual --cross-attn --attn-method Temporal &
PID5=$!;

CUDA_VISIBLE_DEVICES=5 python train_snn_cl.py --model AVspikformer --dataset CREMAD --epoch 100 --batch-size 64 --num-classes 6 --step 4 --modality audio-visual --cross-attn --attn-method SpatialTemporal &
PID6=$!;

CUDA_VISIBLE_DEVICES=6 python train_snn_cl.py --model AVspikformer --dataset CREMAD --epoch 100 --batch-size 64 --num-classes 6 --step 4 --modality audio-visual --cross-attn --attn-method Spatial --av-attn &
PID7=$!;

CUDA_VISIBLE_DEVICES=7 python train_snn_cl.py --model AVspikformer --dataset CREMAD --epoch 100 --batch-size 64 --num-classes 6 --step 4 --modality audio-visual --cross-attn --attn-method SpatialTemporal --av-attn &
PID8=$!;

wait ${PID1} && wait ${PID2} && wait ${PID3} && wait ${PID4} && wait ${PID5} && wait ${PID6} && wait ${PID7} && wait ${PID8}

CUDA_VISIBLE_DEVICES=0 python train_snn_cl.py --model spikformer --dataset UrbanSound8K --epoch 100 --batch-size 64 --num-classes 10 --step 4 --modality audio &
PID1=$!;

CUDA_VISIBLE_DEVICES=1 python train_snn_cl.py --model spikformer --dataset UrbanSound8K --epoch 100 --batch-size 64 --num-classes 10 --step 4 --modality visual &
PID2=$!;

CUDA_VISIBLE_DEVICES=2 python train_snn_cl.py --model AVspikformer --dataset UrbanSound8K --epoch 100 --batch-size 64 --num-classes 10 --step 4 --modality audio-visual &
PID3=$!;

CUDA_VISIBLE_DEVICES=3 python train_snn_cl.py --model AVspikformer --dataset UrbanSound8K --epoch 100 --batch-size 64 --num-classes 10 --step 4 --modality audio-visual --cross-attn --attn-method Spatial &
PID4=$!;

CUDA_VISIBLE_DEVICES=4 python train_snn_cl.py --model AVspikformer --dataset UrbanSound8K --epoch 100 --batch-size 64 --num-classes 10 --step 4 --modality audio-visual --cross-attn --attn-method Temporal &
PID5=$!;

CUDA_VISIBLE_DEVICES=5 python train_snn_cl.py --model AVspikformer --dataset UrbanSound8K --epoch 100 --batch-size 64 --num-classes 10 --step 4 --modality audio-visual --cross-attn --attn-method SpatialTemporal &
PID6=$!;

CUDA_VISIBLE_DEVICES=6 python train_snn_cl.py --model AVspikformer --dataset UrbanSound8K --epoch 100 --batch-size 64 --num-classes 10 --step 4 --modality audio-visual --cross-attn --attn-method Spatial --av-attn &
PID7=$!;

CUDA_VISIBLE_DEVICES=7 python train_snn_cl.py --model AVspikformer --dataset UrbanSound8K --epoch 100 --batch-size 64 --num-classes 10 --step 4 --modality audio-visual --cross-attn --attn-method SpatialTemporal --av-attn &
PID8=$!;