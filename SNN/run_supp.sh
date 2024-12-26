CUDA_VISIBLE_DEVICES=0 python train_snn.py --model AVspikformer --dataset CREMAD --epoch 100 --batch-size 128 --num-classes 6 --step 4 --modality audio-visual --cross-attn --attn-method SpatialTemporal --alpha 1.0 --contrastive --snr 5 --output ./exp_results_snr --tensorboard-dir ./exp_results_snr&
PID1=$!;

CUDA_VISIBLE_DEVICES=1 python train_snn.py --model AVspikformer --dataset CREMAD --epoch 100 --batch-size 128 --num-classes 6 --step 4 --modality audio-visual --cross-attn --attn-method SpatialTemporal --alpha 1.0 --contrastive --snr 15 --output ./exp_results_snr --tensorboard-dir ./exp_results_snr&
PID2=$!;

CUDA_VISIBLE_DEVICES=2 python train_snn.py --model AVspikformer --dataset CREMAD --epoch 100 --batch-size 128 --num-classes 6 --step 4 --modality audio-visual --cross-attn --attn-method SpatialTemporal --alpha 1.0 --contrastive --snr 25 --output ./exp_results_snr --tensorboard-dir ./exp_results_snr&
PID3=$!;


CUDA_VISIBLE_DEVICES=3 python train_snn.py --model AVspikformer --dataset CREMAD --epoch 100 --batch-size 128 --num-classes 6 --step 4 --modality audio-visual --cross-attn --attn-method WeightAttention --snr 5 --output ./exp_results_snr --tensorboard-dir ./exp_results_snr&
PID4=$!;

CUDA_VISIBLE_DEVICES=4 python train_snn.py --model AVspikformer --dataset CREMAD --epoch 100 --batch-size 128 --num-classes 6 --step 4 --modality audio-visual --cross-attn --attn-method WeightAttention --snr 15 --output ./exp_results_snr --tensorboard-dir ./exp_results_snr&
PID5=$!;

CUDA_VISIBLE_DEVICES=5 python train_snn.py --model AVspikformer --dataset CREMAD --epoch 100 --batch-size 128 --num-classes 6 --step 4 --modality audio-visual --cross-attn --attn-method WeightAttention --snr 25 --output ./exp_results_snr --tensorboard-dir ./exp_results_snr&
PID6=$!;

wait ${PID1} && wait ${PID2} && wait ${PID3} && wait ${PID4} && wait ${PID5} && wait ${PID6}


CUDA_VISIBLE_DEVICES=0 python train_snn.py --model AVspikformer --dataset CREMAD --epoch 100 --batch-size 128 --num-classes 6 --step 4 --modality audio-visual --cross-attn --attn-method SCA --snr 5 --output ./exp_results_snr --tensorboard-dir ./exp_results_snr&
PID1=$!;

CUDA_VISIBLE_DEVICES=1 python train_snn.py --model AVspikformer --dataset CREMAD --epoch 100 --batch-size 128 --num-classes 6 --step 4 --modality audio-visual --cross-attn --attn-method SCA --snr 15 --output ./exp_results_snr --tensorboard-dir ./exp_results_snr&
PID2=$!;

CUDA_VISIBLE_DEVICES=2 python train_snn.py --model AVspikformer --dataset CREMAD --epoch 100 --batch-size 128 --num-classes 6 --step 4 --modality audio-visual --cross-attn --attn-method SCA --snr 25 --output ./exp_results_snr --tensorboard-dir ./exp_results_snr&
PID3=$!;

CUDA_VISIBLE_DEVICES=3 python train_snn.py --model AVspikformer --dataset CREMAD --epoch 100 --batch-size 128 --num-classes 6 --step 4 --modality audio-visual --cross-attn --attn-method CMCI --snr 5 --output ./exp_results_snr --tensorboard-dir ./exp_results_snr&
PID4=$!;

CUDA_VISIBLE_DEVICES=4 python train_snn.py --model AVspikformer --dataset CREMAD --epoch 100 --batch-size 128 --num-classes 6 --step 4 --modality audio-visual --cross-attn --attn-method CMCI --snr 15 --output ./exp_results_snr --tensorboard-dir ./exp_results_snr&
PID5=$!;

CUDA_VISIBLE_DEVICES=5 python train_snn.py --model AVspikformer --dataset CREMAD --epoch 100 --batch-size 128 --num-classes 6 --step 4 --modality audio-visual --cross-attn --attn-method CMCI --snr 25 --output ./exp_results_snr --tensorboard-dir ./exp_results_snr&
PID6=$!;

wait ${PID1} && wait ${PID2} && wait ${PID3} && wait ${PID4} && wait ${PID5} && wait ${PID6}