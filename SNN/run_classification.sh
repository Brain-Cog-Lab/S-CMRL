## --------A. CREMA-D数据集；单模态和多模态的baseline, 不同alpha----------
#CUDA_VISIBLE_DEVICES=0 python train_snn.py --model spikformer --dataset CREMAD --epoch 100 --batch-size 128 --num-classes 6 --step 4 --modality audio &
#PID1=$!;
#
#CUDA_VISIBLE_DEVICES=1 python train_snn.py --model spikformer --dataset CREMAD --epoch 100 --batch-size 128 --num-classes 6 --step 4 --modality visual &
#PID2=$!;
#
#CUDA_VISIBLE_DEVICES=2 python train_snn.py --model AVspikformer --dataset CREMAD --epoch 100 --batch-size 128 --num-classes 6 --step 4 --modality audio-visual &
#PID3=$!;
#
#CUDA_VISIBLE_DEVICES=3 python train_snn.py --model AVspikformer --dataset CREMAD --epoch 100 --batch-size 128 --num-classes 6 --step 4 --modality audio-visual --cross-attn --attn-method SpatialTemporal --alpha 0.0&
#PID4=$!;
#
#CUDA_VISIBLE_DEVICES=4 python train_snn.py --model AVspikformer --dataset CREMAD --epoch 100 --batch-size 128 --num-classes 6 --step 4 --modality audio-visual --cross-attn --attn-method SpatialTemporal --alpha 0.5&
#PID5=$!;
#
#CUDA_VISIBLE_DEVICES=5 python train_snn.py --model AVspikformer --dataset CREMAD --epoch 100 --batch-size 128 --num-classes 6 --step 4 --modality audio-visual --cross-attn --attn-method SpatialTemporal --alpha 1.0&
#PID6=$!;
#
#CUDA_VISIBLE_DEVICES=6 python train_snn.py --model AVspikformer --dataset CREMAD --epoch 100 --batch-size 128 --num-classes 6 --step 4 --modality audio-visual --cross-attn --attn-method SpatialTemporal --alpha 1.5&
#PID7=$!;
#
#CUDA_VISIBLE_DEVICES=7 python train_snn.py --model AVspikformer --dataset CREMAD --epoch 100 --batch-size 128 --num-classes 6 --step 4 --modality audio-visual --cross-attn --attn-method SpatialTemporal --alpha 2.0&
#PID8=$!;
#
#wait ${PID1} && wait ${PID2} && wait ${PID3} && wait ${PID4} && wait ${PID5} && wait ${PID6} && wait ${PID7} && wait ${PID8}
#
# --------A. UrbanSound8K-AV数据集；单模态和多模态的baseline, 不同alpha----------
#CUDA_VISIBLE_DEVICES=0 python train_snn.py --model spikformer --embed_dims 64 --dataset UrbanSound8K --epoch 100 --batch-size 128 --num-classes 10 --step 4 --modality audio &
#PID1=$!;
#
#CUDA_VISIBLE_DEVICES=1 python train_snn.py --model spikformer --embed_dims 64 --dataset UrbanSound8K --epoch 100 --batch-size 128 --num-classes 10 --step 4 --modality visual &
#PID2=$!;
#
#CUDA_VISIBLE_DEVICES=2 python train_snn.py --model AVspikformer --embed_dims 64 --dataset UrbanSound8K --epoch 100 --batch-size 128 --num-classes 10 --step 4 --modality audio-visual &
#PID3=$!;
#
#CUDA_VISIBLE_DEVICES=3 python train_snn.py --model AVspikformer --embed_dims 64 --dataset UrbanSound8K --epoch 100 --batch-size 128 --num-classes 10 --step 4 --modality audio-visual --cross-attn --attn-method SpatialTemporal --alpha 0.0&
#PID4=$!;
#
#CUDA_VISIBLE_DEVICES=4 python train_snn.py --model AVspikformer --embed_dims 64 --dataset UrbanSound8K --epoch 100 --batch-size 128 --num-classes 10 --step 4 --modality audio-visual --cross-attn --attn-method SpatialTemporal --alpha 0.5&
#PID5=$!;
#
#CUDA_VISIBLE_DEVICES=5 python train_snn.py --model AVspikformer --embed_dims 64 --dataset UrbanSound8K --epoch 100 --batch-size 128 --num-classes 10 --step 4 --modality audio-visual --cross-attn --attn-method SpatialTemporal --alpha 1.0&
#PID6=$!;
#
#CUDA_VISIBLE_DEVICES=6 python train_snn.py --model AVspikformer --embed_dims 64 --dataset UrbanSound8K --epoch 100 --batch-size 128 --num-classes 10 --step 4 --modality audio-visual --cross-attn --attn-method SpatialTemporal --alpha 1.5&
#PID7=$!;
#
#CUDA_VISIBLE_DEVICES=7 python train_snn.py --model AVspikformer --embed_dims 64 --dataset UrbanSound8K --epoch 100 --batch-size 128 --num-classes 10 --step 4 --modality audio-visual --cross-attn --attn-method SpatialTemporal --alpha 2.0&
#PID8=$!;
#
#wait ${PID1} && wait ${PID2} && wait ${PID3} && wait ${PID4} && wait ${PID5} && wait ${PID6} && wait ${PID7} && wait ${PID8}

# --------B. CREMA-D数据集；对比增强以及与其他sotas的比较----------
#CUDA_VISIBLE_DEVICES=0 python train_snn.py --model AVspikformer --dataset CREMAD --epoch 100 --batch-size 128 --num-classes 6 --step 4 --modality audio-visual --cross-attn --attn-method Spatial &
#PID1=$!;
#
#CUDA_VISIBLE_DEVICES=1 python train_snn.py --model AVspikformer --dataset CREMAD --epoch 100 --batch-size 128 --num-classes 6 --step 4 --modality audio-visual --cross-attn --attn-method Temporal &
#PID2=$!;
#
#CUDA_VISIBLE_DEVICES=2 python train_snn.py --model AVspikformer --dataset CREMAD --epoch 100 --batch-size 128 --num-classes 6 --step 4 --modality audio-visual --interaction Concat&
#PID3=$!;
#
#CUDA_VISIBLE_DEVICES=3 python train_snn.py --model AVspikformer --dataset CREMAD --epoch 100 --batch-size 128 --num-classes 6 --step 4 --modality audio-visual --cross-attn --attn-method SpatialTemporal --alpha 0.0 --contrastive&
#PID4=$!;
#
#CUDA_VISIBLE_DEVICES=4 python train_snn.py --model AVspikformer --dataset CREMAD --epoch 100 --batch-size 128 --num-classes 6 --step 4 --modality audio-visual --cross-attn --attn-method SpatialTemporal --alpha 1.5 --contrastive&
#PID5=$!;
#
#CUDA_VISIBLE_DEVICES=5 python train_snn.py --model AVspikformer --dataset CREMAD --epoch 100 --batch-size 128 --num-classes 6 --step 4 --modality audio-visual --cross-attn --attn-method WeightAttention &
#PID6=$!;
#
#CUDA_VISIBLE_DEVICES=6 python train_snn.py --model AVspikformer --dataset CREMAD --epoch 100 --batch-size 128 --num-classes 6 --step 4 --modality audio-visual --cross-attn --attn-method SCA &
#PID7=$!;
#
#CUDA_VISIBLE_DEVICES=7 python train_snn.py --model AVspikformer --dataset CREMAD --epoch 100 --batch-size 128 --num-classes 6 --step 4 --modality audio-visual --cross-attn --attn-method CMCI &
#PID8=$!;
#
#wait ${PID1} && wait ${PID2} && wait ${PID3} && wait ${PID4} && wait ${PID5} && wait ${PID6} && wait ${PID7} && wait ${PID8}


## --------B. UrbanSound8k数据集；对比增强以及与其他sotas的比较----------
#CUDA_VISIBLE_DEVICES=0 python train_snn.py --model AVspikformer --embed_dims 64 --dataset UrbanSound8K --epoch 100 --batch-size 128 --num-classes 10 --step 4 --modality audio-visual --cross-attn --attn-method Spatial --alpha 1.5&
#PID1=$!;
#
#CUDA_VISIBLE_DEVICES=1 python train_snn.py --model AVspikformer --embed_dims 64 --dataset UrbanSound8K --epoch 100 --batch-size 128 --num-classes 10 --step 4 --modality audio-visual --cross-attn --attn-method Temporal --alpha 1.5&
#PID2=$!;
#
#CUDA_VISIBLE_DEVICES=2 python train_snn.py --model AVspikformer --embed_dims 64 --dataset UrbanSound8K --epoch 100 --batch-size 128 --num-classes 10 --step 4 --modality audio-visual --interaction Concat&
#PID3=$!;
#
#CUDA_VISIBLE_DEVICES=3 python train_snn.py --model AVspikformer --embed_dims 64 --dataset UrbanSound8K --epoch 100 --batch-size 128 --num-classes 10 --step 4 --modality audio-visual --cross-attn --attn-method SpatialTemporal --alpha 0.0 --contrastive&
#PID4=$!;
#
#CUDA_VISIBLE_DEVICES=4 python train_snn.py --model AVspikformer --embed_dims 64 --dataset UrbanSound8K --epoch 100 --batch-size 128 --num-classes 10 --step 4 --modality audio-visual --cross-attn --attn-method SpatialTemporal --alpha 1.5 --contrastive&
#PID5=$!;
#
#CUDA_VISIBLE_DEVICES=5 python train_snn.py --model AVspikformer --embed_dims 64 --dataset UrbanSound8K --epoch 100 --batch-size 128 --num-classes 10 --step 4 --modality audio-visual --cross-attn --attn-method WeightAttention &
#PID6=$!;
#
#CUDA_VISIBLE_DEVICES=6 python train_snn.py --model AVspikformer --embed_dims 64 --dataset UrbanSound8K --epoch 100 --batch-size 128 --num-classes 10 --step 4 --modality audio-visual --cross-attn --attn-method SCA &
#PID7=$!;
#
#CUDA_VISIBLE_DEVICES=7 python train_snn.py --model AVspikformer --embed_dims 64 --dataset UrbanSound8K --epoch 100 --batch-size 128 --num-classes 10 --step 4 --modality audio-visual --cross-attn --attn-method CMCI &
#PID8=$!;
#
#wait ${PID1} && wait ${PID2} && wait ${PID3} && wait ${PID4} && wait ${PID5} && wait ${PID6} && wait ${PID7} && wait ${PID8}


# --------C. CREMA-D数据集；在不同snr下的表现----------
#CUDA_VISIBLE_DEVICES=0 python train_snn.py --model AVspikformer --dataset CREMAD --epoch 100 --batch-size 128 --num-classes 6 --step 4 --modality audio-visual --cross-attn --attn-method SpatialTemporal --alpha 1.5 --contrastive --temperature 0.07 --snr 0 --output ./exp_results_snr --tensorboard-dir ./exp_results_snr&
#PID1=$!;
#
#CUDA_VISIBLE_DEVICES=1 python train_snn.py --model AVspikformer --dataset CREMAD --epoch 100 --batch-size 128 --num-classes 6 --step 4 --modality audio-visual --cross-attn --attn-method SpatialTemporal --alpha 1.5 --contrastive --temperature 0.07 --snr 5 --output ./exp_results_snr --tensorboard-dir ./exp_results_snr&
#PID2=$!;
#
#CUDA_VISIBLE_DEVICES=2 python train_snn.py --model AVspikformer --dataset CREMAD --epoch 100 --batch-size 128 --num-classes 6 --step 4 --modality audio-visual --cross-attn --attn-method SpatialTemporal --alpha 1.5 --contrastive --temperature 0.07 --snr 10 --output ./exp_results_snr --tensorboard-dir ./exp_results_snr&
#PID3=$!;
#
#CUDA_VISIBLE_DEVICES=3 python train_snn.py --model AVspikformer --dataset CREMAD --epoch 100 --batch-size 128 --num-classes 6 --step 4 --modality audio-visual --cross-attn --attn-method SpatialTemporal --alpha 1.5 --contrastive --temperature 0.07 --snr 15 --output ./exp_results_snr --tensorboard-dir ./exp_results_snr&
#PID4=$!;
#
#CUDA_VISIBLE_DEVICES=4 python train_snn.py --model AVspikformer --dataset CREMAD --epoch 100 --batch-size 128 --num-classes 6 --step 4 --modality audio-visual --cross-attn --attn-method SpatialTemporal --alpha 1.5 --contrastive --temperature 0.07 --snr 20 --output ./exp_results_snr --tensorboard-dir ./exp_results_snr&
#PID5=$!;
#
#CUDA_VISIBLE_DEVICES=5 python train_snn.py --model AVspikformer --dataset CREMAD --epoch 100 --batch-size 128 --num-classes 6 --step 4 --modality audio-visual --cross-attn --attn-method SpatialTemporal --alpha 1.5 --contrastive --temperature 0.07 --snr 25 --output ./exp_results_snr --tensorboard-dir ./exp_results_snr&
#PID6=$!;
#
#CUDA_VISIBLE_DEVICES=6 python train_snn.py --model AVspikformer --dataset CREMAD --epoch 100 --batch-size 128 --num-classes 6 --step 4 --modality audio-visual --cross-attn --attn-method SpatialTemporal --alpha 1.5 --contrastive --temperature 0.07 --snr 30 --output ./exp_results_snr --tensorboard-dir ./exp_results_snr&
#PID7=$!;
#
#CUDA_VISIBLE_DEVICES=7 python train_snn.py --model AVspikformer --dataset CREMAD --epoch 100 --batch-size 128 --num-classes 6 --step 4 --modality audio-visual --cross-attn --attn-method SpatialTemporal --alpha 1.5 --contrastive --temperature 0.07 --output ./exp_results_snr --tensorboard-dir ./exp_results_snr&
#PID8=$!;
#
#wait ${PID1} && wait ${PID2} && wait ${PID3} && wait ${PID4} && wait ${PID5} && wait ${PID6} && wait ${PID7} && wait ${PID8}

#CUDA_VISIBLE_DEVICES=0 python train_snn.py --model AVspikformer --dataset CREMAD --epoch 100 --batch-size 128 --num-classes 6 --step 4 --modality audio-visual --cross-attn --attn-method WeightAttention --snr 0 --output ./exp_results_snr --tensorboard-dir ./exp_results_snr&
#PID1=$!;

#CUDA_VISIBLE_DEVICES=1 python train_snn.py --model AVspikformer --dataset CREMAD --epoch 100 --batch-size 128 --num-classes 6 --step 4 --modality audio-visual --cross-attn --attn-method WeightAttention --snr 5 --output ./exp_results_snr --tensorboard-dir ./exp_results_snr&
#PID2=$!;

#CUDA_VISIBLE_DEVICES=2 python train_snn.py --model AVspikformer --dataset CREMAD --epoch 100 --batch-size 128 --num-classes 6 --step 4 --modality audio-visual --cross-attn --attn-method WeightAttention --snr 10 --output ./exp_results_snr --tensorboard-dir ./exp_results_snr&
#PID3=$!;

#CUDA_VISIBLE_DEVICES=3 python train_snn.py --model AVspikformer --dataset CREMAD --epoch 100 --batch-size 128 --num-classes 6 --step 4 --modality audio-visual --cross-attn --attn-method WeightAttention --snr 15 --output ./exp_results_snr --tensorboard-dir ./exp_results_snr&
#PID4=$!;

#CUDA_VISIBLE_DEVICES=4 python train_snn.py --model AVspikformer --dataset CREMAD --epoch 100 --batch-size 128 --num-classes 6 --step 4 --modality audio-visual --cross-attn --attn-method WeightAttention --snr 20 --output ./exp_results_snr --tensorboard-dir ./exp_results_snr&
#PID5=$!;

#CUDA_VISIBLE_DEVICES=5 python train_snn.py --model AVspikformer --dataset CREMAD --epoch 100 --batch-size 128 --num-classes 6 --step 4 --modality audio-visual --cross-attn --attn-method WeightAttention --snr 25 --output ./exp_results_snr --tensorboard-dir ./exp_results_snr&
#PID6=$!;

#CUDA_VISIBLE_DEVICES=6 python train_snn.py --model AVspikformer --dataset CREMAD --epoch 100 --batch-size 128 --num-classes 6 --step 4 --modality audio-visual --cross-attn --attn-method WeightAttention --snr 30 --output ./exp_results_snr --tensorboard-dir ./exp_results_snr&
#

#CUDA_VISIBLE_DEVICES=7 python train_snn.py --model AVspikformer --dataset CREMAD --epoch 100 --batch-size 128 --num-classes 6 --step 4 --modality audio-visual --cross-attn --attn-method WeightAttention --output ./exp_results_snr --tensorboard-dir ./exp_results_snr&
#
#wait ${PID1} && wait ${PID2} && wait ${PID3} && wait ${PID4} && wait ${PID5} && wait ${PID6} && wait ${PID7} && wait ${PID8}


#CUDA_VISIBLE_DEVICES=0 python train_snn.py --model AVspikformer --dataset CREMAD --epoch 100 --batch-size 128 --num-classes 6 --step 4 --modality audio-visual --cross-attn --attn-method SCA --snr 0 --output ./exp_results_snr --tensorboard-dir ./exp_results_snr&
#PID1=$!;

#CUDA_VISIBLE_DEVICES=1 python train_snn.py --model AVspikformer --dataset CREMAD --epoch 100 --batch-size 128 --num-classes 6 --step 4 --modality audio-visual --cross-attn --attn-method SCA --snr 5 --output ./exp_results_snr --tensorboard-dir ./exp_results_snr&
#PID2=$!;

#CUDA_VISIBLE_DEVICES=2 python train_snn.py --model AVspikformer --dataset CREMAD --epoch 100 --batch-size 128 --num-classes 6 --step 4 --modality audio-visual --cross-attn --attn-method SCA --snr 10 --output ./exp_results_snr --tensorboard-dir ./exp_results_snr&
#PID3=$!;

#CUDA_VISIBLE_DEVICES=3 python train_snn.py --model AVspikformer --dataset CREMAD --epoch 100 --batch-size 128 --num-classes 6 --step 4 --modality audio-visual --cross-attn --attn-method SCA --snr 15 --output ./exp_results_snr --tensorboard-dir ./exp_results_snr&
#PID4=$!;

#CUDA_VISIBLE_DEVICES=4 python train_snn.py --model AVspikformer --dataset CREMAD --epoch 100 --batch-size 128 --num-classes 6 --step 4 --modality audio-visual --cross-attn --attn-method SCA --snr 20 --output ./exp_results_snr --tensorboard-dir ./exp_results_snr&
#PID5=$!;

#CUDA_VISIBLE_DEVICES=5 python train_snn.py --model AVspikformer --dataset CREMAD --epoch 100 --batch-size 128 --num-classes 6 --step 4 --modality audio-visual --cross-attn --attn-method SCA --snr 25 --output ./exp_results_snr --tensorboard-dir ./exp_results_snr&
#PID6=$!;

#CUDA_VISIBLE_DEVICES=6 python train_snn.py --model AVspikformer --dataset CREMAD --epoch 100 --batch-size 128 --num-classes 6 --step 4 --modality audio-visual --cross-attn --attn-method SCA --snr 30 --output ./exp_results_snr --tensorboard-dir ./exp_results_snr&
#PID7=$!;

#CUDA_VISIBLE_DEVICES=7 python train_snn.py --model AVspikformer --dataset CREMAD --epoch 100 --batch-size 128 --num-classes 6 --step 4 --modality audio-visual --cross-attn --attn-method SCA --output ./exp_results_snr --tensorboard-dir ./exp_results_snr&
#PID8=$!;

#wait ${PID1} && wait ${PID2} && wait ${PID3} && wait ${PID4} && wait ${PID5} && wait ${PID6} && wait ${PID7} && wait ${PID8}


#CUDA_VISIBLE_DEVICES=0 python train_snn.py --model AVspikformer --dataset CREMAD --epoch 100 --batch-size 128 --num-classes 6 --step 4 --modality audio-visual --cross-attn --attn-method CMCI --snr 0 --output ./exp_results_snr --tensorboard-dir ./exp_results_snr&
#PID1=$!;

#CUDA_VISIBLE_DEVICES=1 python train_snn.py --model AVspikformer --dataset CREMAD --epoch 100 --batch-size 128 --num-classes 6 --step 4 --modality audio-visual --cross-attn --attn-method CMCI --snr 5 --output ./exp_results_snr --tensorboard-dir ./exp_results_snr&
#PID2=$!;

#CUDA_VISIBLE_DEVICES=2 python train_snn.py --model AVspikformer --dataset CREMAD --epoch 100 --batch-size 128 --num-classes 6 --step 4 --modality audio-visual --cross-attn --attn-method CMCI --snr 10 --output ./exp_results_snr --tensorboard-dir ./exp_results_snr&
#PID3=$!;

#CUDA_VISIBLE_DEVICES=3 python train_snn.py --model AVspikformer --dataset CREMAD --epoch 100 --batch-size 128 --num-classes 6 --step 4 --modality audio-visual --cross-attn --attn-method CMCI --snr 15 --output ./exp_results_snr --tensorboard-dir ./exp_results_snr&
#PID4=$!;

#CUDA_VISIBLE_DEVICES=4 python train_snn.py --model AVspikformer --dataset CREMAD --epoch 100 --batch-size 128 --num-classes 6 --step 4 --modality audio-visual --cross-attn --attn-method CMCI --snr 20 --output ./exp_results_snr --tensorboard-dir ./exp_results_snr&
#PID5=$!;

#CUDA_VISIBLE_DEVICES=5 python train_snn.py --model AVspikformer --dataset CREMAD --epoch 100 --batch-size 128 --num-classes 6 --step 4 --modality audio-visual --cross-attn --attn-method CMCI --snr 25 --output ./exp_results_snr --tensorboard-dir ./exp_results_snr&
#PID6=$!;

#CUDA_VISIBLE_DEVICES=4 python train_snn.py --model AVspikformer --dataset CREMAD --epoch 100 --batch-size 128 --num-classes 6 --step 4 --modality audio-visual --cross-attn --attn-method CMCI --snr 30 --output ./exp_results_snr --tensorboard-dir ./exp_results_snr&
#PID7=$!;

#CUDA_VISIBLE_DEVICES=5 python train_snn.py --model AVspikformer --dataset CREMAD --epoch 100 --batch-size 128 --num-classes 6 --step 4 --modality audio-visual --cross-attn --attn-method CMCI --output ./exp_results_snr --tensorboard-dir ./exp_results_snr&
#PID8=$!;

#wait ${PID1} && wait ${PID2} && wait ${PID3} && wait ${PID4} && wait ${PID5} && wait ${PID6} && wait ${PID7} && wait ${PID8}

# --------D. CREMA-D数据集；多模态信息给单模态带来的增强----------
#CUDA_VISIBLE_DEVICES=0 python train_snn.py --model spikformer --dataset CREMAD --epoch 100 --batch-size 128 --num-classes 6 --step 4 --modality audio --load-avmodel --output ./exp_results_singleModality --tensorboard-dir ./exp_results_singleModality&
#PID1=$!;
#
#CUDA_VISIBLE_DEVICES=1 python train_snn.py --model spikformer --dataset CREMAD --epoch 100 --batch-size 128 --num-classes 6 --step 4 --modality visual --load-avmodel --output ./exp_results_singleModality --tensorboard-dir ./exp_results_singleModality&
#PID2=$!;
#
#CUDA_VISIBLE_DEVICES=2 python train_snn.py --model spikformer --embed_dims 64 --dataset UrbanSound8K --epoch 100 --batch-size 128 --num-classes 10 --step 4 --modality audio --load-avmodel --output ./exp_results_singleModality --tensorboard-dir ./exp_results_singleModality&
#PID3=$!;
#
#CUDA_VISIBLE_DEVICES=3 python train_snn.py --model spikformer --embed_dims 64 --dataset UrbanSound8K --epoch 100 --batch-size 128 --num-classes 10 --step 4 --modality visual --load-avmodel --output ./exp_results_singleModality --tensorboard-dir ./exp_results_singleModality&
#PID4=$!;
#
#wait ${PID1} && wait ${PID2} && wait ${PID3} && wait ${PID4}

# --------E. Avmnist-dvs数据集；我们的方法与其他sotas的比较----------
CUDA_VISIBLE_DEVICES=0 python train_snn.py --model AVspikformer --dataset AVmnistdvs --epoch 100 --batch-size 128 --num-classes 10 --step 64 --modality audio-visual --shallow-sps --embed_dims 32 --event-size 28 --cross-attn --attn-method SpatialTemporal --alpha 1.5 --contrastive &

CUDA_VISIBLE_DEVICES=1 python train_snn.py --model AVspikformer --dataset AVmnistdvs --epoch 100 --batch-size 128 --num-classes 10 --step 64 --modality audio-visual --shallow-sps --embed_dims 64 --event-size 28 --cross-attn --attn-method SpatialTemporal --alpha 1.5 --contrastive &

CUDA_VISIBLE_DEVICES=2 python train_snn.py --model AVspikformer --dataset AVmnistdvs --epoch 100 --batch-size 128 --num-classes 10 --step 64 --modality audio-visual --shallow-sps --embed_dims 128 --event-size 28 --cross-attn --attn-method SpatialTemporal --alpha 1.5 --contrastive &

CUDA_VISIBLE_DEVICES=3 python train_snn.py --model AVspikformer --dataset AVmnistdvs --epoch 100 --batch-size 128 --num-classes 10 --step 64 --modality audio-visual --shallow-sps --embed_dims 256 --event-size 28 --cross-attn --attn-method SpatialTemporal --alpha 1.5 --contrastive &

CUDA_VISIBLE_DEVICES=4 python train_snn.py --model AVspikformer --dataset AVmnistdvs --epoch 100 --batch-size 128 --num-classes 10 --step 64 --modality audio-visual --shallow-sps --embed_dims 512 --event-size 28 --cross-attn --attn-method SpatialTemporal --alpha 1.5 --contrastive &

# --------F. 超参数实验---------------
#CUDA_VISIBLE_DEVICES=0 python train_snn.py --model AVspikformer --embed_dims 64 --dataset UrbanSound8K --epoch 100 --batch-size 128 --num-classes 10 --step 4 --modality audio-visual --cross-attn --attn-method SpatialTemporal --alpha 1.5 --contrastive --temperature 0.03 &
#PID1=$!;
#
#CUDA_VISIBLE_DEVICES=1 python train_snn.py --model AVspikformer --embed_dims 64 --dataset UrbanSound8K --epoch 100 --batch-size 128 --num-classes 10 --step 4 --modality audio-visual --cross-attn --attn-method SpatialTemporal --alpha 1.5 --contrastive --temperature 0.05 &
#PID2=$!;
#
#CUDA_VISIBLE_DEVICES=2 python train_snn.py --model AVspikformer --embed_dims 64 --dataset UrbanSound8K --epoch 100 --batch-size 128 --num-classes 10 --step 4 --modality audio-visual --cross-attn --attn-method SpatialTemporal --alpha 1.5 --contrastive --temperature 0.08 &
#PID3=$!;
#
#CUDA_VISIBLE_DEVICES=3 python train_snn.py --model AVspikformer --embed_dims 64 --dataset UrbanSound8K --epoch 100 --batch-size 128 --num-classes 10 --step 4 --modality audio-visual --cross-attn --attn-method SpatialTemporal --alpha 1.5 --contrastive --temperature 0.1 &
#PID4=$!;
#
#CUDA_VISIBLE_DEVICES=4 python train_snn.py --model AVspikformer --dataset CREMAD --epoch 100 --batch-size 128 --num-classes 6 --step 4 --modality audio-visual --cross-attn --attn-method SpatialTemporal --alpha 1.5 --contrastive --temperature 0.03 &
#PID5=$!;
#
#CUDA_VISIBLE_DEVICES=5 python train_snn.py --model AVspikformer --dataset CREMAD --epoch 100 --batch-size 128 --num-classes 6 --step 4 --modality audio-visual --cross-attn --attn-method SpatialTemporal --alpha 1.5 --contrastive --temperature 0.05 &
#PID6=$!;
#
#CUDA_VISIBLE_DEVICES=6 python train_snn.py --model AVspikformer --dataset CREMAD --epoch 100 --batch-size 128 --num-classes 6 --step 4 --modality audio-visual --cross-attn --attn-method SpatialTemporal --alpha 1.5 --contrastive --temperature 0.08 &
#PID7=$!;
#
#CUDA_VISIBLE_DEVICES=7 python train_snn.py --model AVspikformer --dataset CREMAD --epoch 100 --batch-size 128 --num-classes 6 --step 4 --modality audio-visual --cross-attn --attn-method SpatialTemporal --alpha 1.5 --contrastive --temperature 0.1 &
#PID8=$!;
#
#wait ${PID1} && wait ${PID2} && wait ${PID3} && wait ${PID4} && wait ${PID5} && wait ${PID6} && wait ${PID7} && wait ${PID8}


# --------超参数箱线图----------
#CUDA_VISIBLE_DEVICES=0 python train_snn.py --model AVspikformer --dataset CREMAD --epoch 100 --batch-size 128 --num-classes 6 --step 4 --modality audio-visual --cross-attn --attn-method SpatialTemporal --alpha 0.0 --contrastive --temperature 0.07 --seed 47 --output ./exp_results_seed47 --tensorboard-dir ./exp_results_seed47&
#PID1=$!;
#
#CUDA_VISIBLE_DEVICES=1 python train_snn.py --model AVspikformer --dataset CREMAD --epoch 100 --batch-size 128 --num-classes 6 --step 4 --modality audio-visual --cross-attn --attn-method SpatialTemporal --alpha 0.5 --contrastive --temperature 0.07 --seed 47 --output ./exp_results_seed47 --tensorboard-dir ./exp_results_seed47&
#PID2=$!;
#
#CUDA_VISIBLE_DEVICES=2 python train_snn.py --model AVspikformer --dataset CREMAD --epoch 100 --batch-size 128 --num-classes 6 --step 4 --modality audio-visual --cross-attn --attn-method SpatialTemporal --alpha 1.0 --contrastive --temperature 0.07 --seed 47 --output ./exp_results_seed47 --tensorboard-dir ./exp_results_seed47&
#PID3=$!;
#
#CUDA_VISIBLE_DEVICES=3 python train_snn.py --model AVspikformer --dataset CREMAD --epoch 100 --batch-size 128 --num-classes 6 --step 4 --modality audio-visual --cross-attn --attn-method SpatialTemporal --alpha 1.5 --contrastive --temperature 0.07 --seed 47 --output ./exp_results_seed47 --tensorboard-dir ./exp_results_seed47&
#PID4=$!;
#
#CUDA_VISIBLE_DEVICES=4 python train_snn.py --model AVspikformer --dataset CREMAD --epoch 100 --batch-size 128 --num-classes 6 --step 4 --modality audio-visual --cross-attn --attn-method SpatialTemporal --alpha 2.0 --contrastive --temperature 0.07 --seed 47 --output ./exp_results_seed47 --tensorboard-dir ./exp_results_seed47&
#PID5=$!;
#
#wait ${PID1} && wait ${PID2} && wait ${PID3} && wait ${PID4} && wait ${PID5}
#
#
#
#CUDA_VISIBLE_DEVICES=0 python train_snn.py --model AVspikformer --dataset CREMAD --epoch 100 --batch-size 128 --num-classes 6 --step 4 --modality audio-visual --cross-attn --attn-method SpatialTemporal --alpha 0.0 --contrastive --temperature 0.07 --seed 1024 --output ./exp_results_seed47 --tensorboard-dir ./exp_results_seed1024&
#PID1=$!;
#
#CUDA_VISIBLE_DEVICES=1 python train_snn.py --model AVspikformer --dataset CREMAD --epoch 100 --batch-size 128 --num-classes 6 --step 4 --modality audio-visual --cross-attn --attn-method SpatialTemporal --alpha 0.5 --contrastive --temperature 0.07 --seed 1024 --output ./exp_results_seed47 --tensorboard-dir ./exp_results_seed1024&
#PID2=$!;
#
#CUDA_VISIBLE_DEVICES=2 python train_snn.py --model AVspikformer --dataset CREMAD --epoch 100 --batch-size 128 --num-classes 6 --step 4 --modality audio-visual --cross-attn --attn-method SpatialTemporal --alpha 1.0 --contrastive --temperature 0.07 --seed 1024 --output ./exp_results_seed47 --tensorboard-dir ./exp_results_seed1024&
#PID3=$!;
#
#CUDA_VISIBLE_DEVICES=3 python train_snn.py --model AVspikformer --dataset CREMAD --epoch 100 --batch-size 128 --num-classes 6 --step 4 --modality audio-visual --cross-attn --attn-method SpatialTemporal --alpha 1.5 --contrastive --temperature 0.07 --seed 1024 --output ./exp_results_seed47 --tensorboard-dir ./exp_results_seed1024&
#PID4=$!;
#
#CUDA_VISIBLE_DEVICES=4 python train_snn.py --model AVspikformer --dataset CREMAD --epoch 100 --batch-size 128 --num-classes 6 --step 4 --modality audio-visual --cross-attn --attn-method SpatialTemporal --alpha 2.0 --contrastive --temperature 0.07 --seed 1024 --output ./exp_results_seed47 --tensorboard-dir ./exp_results_seed1024&
#PID5=$!;
#
#wait ${PID1} && wait ${PID2} && wait ${PID3} && wait ${PID4} && wait ${PID5}
