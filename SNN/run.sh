#CUDA_VISIBLE_DEVICES=0 python train_snn_cl.py --model spikformer --dataset UrbanSound8K --epoch 100 --batch-size 32 --num-classes 10 --step 2 --modality audio&
#
#CUDA_VISIBLE_DEVICES=1 python train_snn_cl.py --model spikformer --dataset UrbanSound8K --epoch 100 --batch-size 32 --num-classes 10 --step 2 --modality visual&
#
#CUDA_VISIBLE_DEVICES=2 python train_snn_cl.py --model AVspikformer --dataset UrbanSound8K --epoch 100 --batch-size 32 --num-classes 10 --step 2 --modality audio-visual&
#
#CUDA_VISIBLE_DEVICES=3 python train_snn_cl.py --model AVspikformer --dataset UrbanSound8K --epoch 100 --batch-size 32 --num-classes 10 --step 2 --modality audio-visual --cross-attn&


CUDA_VISIBLE_DEVICES=4 python train_snn_cl.py --model spikformer --dataset AVmnistdvs --epoch 100 --batch-size 64 --num-classes 10 --step 2 --modality audio --shallow-sps --event-size 28&

CUDA_VISIBLE_DEVICES=5 python train_snn_cl.py --model spikformer --dataset AVmnistdvs --epoch 100 --batch-size 64 --num-classes 10 --step 2 --modality visual --shallow-sps --event-size 28&

CUDA_VISIBLE_DEVICES=6 python train_snn_cl.py --model AVspikformer --dataset AVmnistdvs --epoch 100 --batch-size 64 --num-classes 10 --step 2 --modality audio-visual --shallow-sps --event-size 28&

CUDA_VISIBLE_DEVICES=7 python train_snn_cl.py --model AVspikformer --dataset AVmnistdvs --epoch 100 --batch-size 64 --num-classes 10 --step 2 --modality audio-visual --shallow-sps --event-size 28 --cross-attn&