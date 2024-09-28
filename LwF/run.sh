# VGGSound_100
CUDA_VISIBLE_DEVICES=0,1 python train_incremental_lwf.py --dataset VGGSound_100 --num_classes 100 --class_num_per_step 10 --modality audio --max_epoches 300 --num_workers 0 --lr 1e-3 --lr_decay False --milestones 100 --weight_decay 1e-4 --train_batch_size 256 --infer_batch_size 128&

PID1=$!;

CUDA_VISIBLE_DEVICES=2,3 python train_incremental_lwf.py --dataset VGGSound_100 --num_classes 100 --class_num_per_step 10 --modality visual --max_epoches 200 --num_workers 4 --lr 1e-3 --lr_decay False --milestones 100 --weight_decay 1e-4 --train_batch_size 256 --infer_batch_size 128&

PID2=$!;

CUDA_VISIBLE_DEVICES=4,5 python train_incremental_lwf.py --dataset VGGSound_100 --num_classes 100 --class_num_per_step 10 --modality audio-visual --max_epoches 200 --num_workers 4 --lr 1e-3 --lr_decay False --milestones 100 --weight_decay 1e-4 --train_batch_size 256 --infer_batch_size 128&

PID3=$!;
wait ${PID1} && wait ${PID2} && wait ${PID2}


# KSounds
CUDA_VISIBLE_DEVICES=0,1 python train_incremental_ssil.py --dataset ksounds --num_classes 30 --class_num_per_step 6 --modality audio --max_epoches 200 --num_workers 0 --lr 1e-3 --lr_decay False --milestones 100 --weight_decay 1e-4 --train_batch_size 256 --infer_batch_size 128&

PID1=$!;

CUDA_VISIBLE_DEVICES=2,3 python train_incremental_ssil.py --dataset ksounds --num_classes 30 --class_num_per_step 6 --modality visual --max_epoches 200 --num_workers 4 --lr 1e-3 --lr_decay False --milestones 100 --weight_decay 1e-4 --train_batch_size 256 --infer_batch_size 128&

PID2=$!;

CUDA_VISIBLE_DEVICES=4,5 python train_incremental_ssil.py --dataset ksounds --num_classes 30 --class_num_per_step 6 --modality audio-visual --max_epoches 100 --num_workers 4 --lr 1e-3 --lr_decay False --milestones 100 --weight_decay 1e-4 --train_batch_size 256 --infer_batch_size 128&

PID3=$!;
wait ${PID1} && wait ${PID2} && wait ${PID2}


# AVE
CUDA_VISIBLE_DEVICES=0,1 python train_incremental_ssil.py --dataset AVE --num_classes 28 --class_num_per_step 7 --modality audio --max_epoches 300 --num_workers 0 --lr 1e-3 --lr_decay False --milestones 100 --weight_decay 1e-4 --train_batch_size 256 --infer_batch_size 128&

PID1=$!;

CUDA_VISIBLE_DEVICES=2,3 python train_incremental_ssil.py --dataset AVE --num_classes 28 --class_num_per_step 7 --modality visual --max_epoches 200 --num_workers 2 --lr 1e-3 --lr_decay False --milestones 100 --weight_decay 1e-4 --train_batch_size 256 --infer_batch_size 128&

PID2=$!;

CUDA_VISIBLE_DEVICES=4,5 python train_incremental_ssil.py --dataset AVE --num_classes 28 --class_num_per_step 7 --modality audio-visual --max_epoches 200 --num_workers 2 --lr 1e-3 --lr_decay False --milestones 100 --weight_decay 1e-4 --train_batch_size 256 --infer_batch_size 128&

PID3=$!;
wait ${PID1} && wait ${PID2} && wait ${PID2}