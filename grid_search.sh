
for wd in 0.01 0.001;
do
    for lr in 0.0005 0.005 0.00005;
    do
        for sam_w in 0.1 1 10;
        do
            CUDA_VISIBLE_DEVICES=0 python tools/train.py \
                --config ./configs/SegSAMPLerEmbMLP_config.py \
                --cfg-options optimizer.lr=${lr} \
                optimizer.weight_decay=${wd} \
                train_batch_size_per_gpu=4 \
                test_batch_size_per_gpu=4 \
                logger.experiment_name=segformer_uv_beijing_${lr}_${wd}_${sam_w}\
                model_cfg.SAM_weights=${sam_w}\
                model_cfg.hyperparameters.optimizer.lr=${lr}\
                model_cfg.hyperparameters.optimizer.weight_decay=${wd}\
                callbacks.1.dirpath=./work_dir/beijing_${lr}_${wd}_${sam_w}/checkpoints\
                trainer_cfg.callbacks.1.dirpath=./work_dir/beijing_${lr}_${wd}_${sam_w}/checkpoints\
                trainer_cfg.default_root_dir=./work_dir/beijing_${lr}_${wd}_${sam_w}/logs\
                trainer_cfg.logger.save_dir=./work_dir/beijing_${lr}_${wd}_${sam_w}/logs
        done
    done
done
```





