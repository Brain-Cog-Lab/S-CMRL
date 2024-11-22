#CUDA_VISIBLE_DEVICES=0 python train_snn_cl.py --model spikformer --dataset UrbanSound8K --epoch 100 --batch-size 64 --num-classes 10 --step 4 --modality audio &
#PID1=$!;
#
#CUDA_VISIBLE_DEVICES=1 python train_snn_cl.py --model spikformer --dataset UrbanSound8K --epoch 100 --batch-size 64 --num-classes 10 --step 4 --modality visual &
#PID2=$!;
#
#CUDA_VISIBLE_DEVICES=2 python train_snn_cl.py --model AVspikformer --dataset UrbanSound8K --epoch 100 --batch-size 64 --num-classes 10 --step 4 --modality audio-visual &
#PID3=$!;
#
#CUDA_VISIBLE_DEVICES=3 python train_snn_cl.py --model AVspikformer --dataset UrbanSound8K --epoch 100 --batch-size 64 --num-classes 10 --step 4 --modality audio-visual --cross-attn &
#PID4=$!;
#
#CUDA_VISIBLE_DEVICES=4 python train_snn_cl.py --model AVspikformer --dataset UrbanSound8K --epoch 100 --batch-size 64 --num-classes 10 --step 4 --modality audio-visual --cross-attn --av-attn &
#PID5=$!;
#
#CUDA_VISIBLE_DEVICES=5 python train_snn_cl.py --model spikformer --dataset AvCifar10 --epoch 100 --batch-size 64 --num-classes 10 --step 4 --modality audio --shallow-sps --event-size 32 &
#PID6=$!;
#
#CUDA_VISIBLE_DEVICES=6 python train_snn_cl.py --model spikformer --dataset AvCifar10 --epoch 100 --batch-size 64 --num-classes 10 --step 4 --modality visual --shallow-sps --event-size 32 &
#PID7=$!;
#
#CUDA_VISIBLE_DEVICES=7 python train_snn_cl.py --model AVspikformer --dataset AvCifar10 --epoch 100 --batch-size 64 --num-classes 10 --step 4 --modality audio-visual --shallow-sps --event-size 32 &
#PID8=$!;
#
#wait ${PID1} && wait ${PID2} && wait ${PID3} && wait ${PID4} && wait ${PID5} && wait ${PID6} && wait ${PID7} && wait ${PID8}

CUDA_VISIBLE_DEVICES=0 python train_snn_cl.py --model AVspikformer --dataset AvCifar10 --epoch 100 --batch-size 64 --num-classes 10 --step 4 --modality audio-visual --shallow-sps --event-size 32 --cross-attn &
PID1=$!;

CUDA_VISIBLE_DEVICES=1 python train_snn_cl.py --model AVspikformer --dataset AvCifar10 --epoch 100 --batch-size 64 --num-classes 10 --step 4 --modality audio-visual --shallow-sps --event-size 32 --cross-attn --av-attn &
PID2=$!;

CUDA_VISIBLE_DEVICES=2 python train_snn_cl.py --model spikformer --dataset AVmnistdvs --epoch 100 --batch-size 64 --num-classes 10 --step 4 --modality audio --shallow-sps --event-size 28 &
PID3=$!;

CUDA_VISIBLE_DEVICES=3 python train_snn_cl.py --model spikformer --dataset AVmnistdvs --epoch 100 --batch-size 64 --num-classes 10 --step 4 --modality visual --shallow-sps --event-size 28 &
PID4=$!;

CUDA_VISIBLE_DEVICES=4 python train_snn_cl.py --model AVspikformer --dataset AVmnistdvs --epoch 100 --batch-size 64 --num-classes 10 --step 4 --modality audio-visual --shallow-sps --event-size 28 &
PID5=$!;

CUDA_VISIBLE_DEVICES=5 python train_snn_cl.py --model AVspikformer --dataset AVmnistdvs --epoch 100 --batch-size 64 --num-classes 10 --step 4 --modality audio-visual --shallow-sps --event-size 28 --cross-attn &
PID6=$!;

CUDA_VISIBLE_DEVICES=6 python train_snn_cl.py --model AVspikformer --dataset AVmnistdvs --epoch 100 --batch-size 64 --num-classes 10 --step 4 --modality audio-visual --shallow-sps --event-size 28 --cross-attn --av-attn &
PID7=$!;

CUDA_VISIBLE_DEVICES=7 python train_snn_cl.py --model spikformer --dataset AvCifar10 --epoch 100 --batch-size 64 --num-classes 10 --step 4 --modality audio --shallow-sps --event-size 32 &  #候补