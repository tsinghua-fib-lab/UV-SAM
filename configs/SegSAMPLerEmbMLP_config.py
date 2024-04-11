custom_imports = dict(imports=['mmseg.datasets', 'mmseg.models'], allow_failed_imports=False)

sub_model_train = [
    'seg_path_backbone',
    'seg_path_decode_head',
    'data_preprocessor',
    'project_head',
]

sub_model_optim = {
    'seg_path_decode_head': {'lr_mult': 1},
    'seg_path_backbone': {'lr_mult': 1},
    'project_head': {'lr_mult': 1},
}

max_epochs =100

optimizer = dict(
    type='AdamW',
    sub_model=sub_model_optim,
    lr=0.0005,
    betas=(0.9, 0.999),
    weight_decay=1e-3
)

param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=1e-4,
        by_epoch=True,
        begin=0,
        end=1,
        # update by iter
        convert_to_iter_based=True),
    # main learning rate scheduler
    dict(
        type='CosineAnnealingLR',
        T_max=max_epochs,
        by_epoch=True,
        begin=1,
        end=max_epochs,
    ),
]


param_scheduler_callback = dict(
    type='ParamSchedulerHook'
)



evaluator_ = dict(type='IoUPLMetric', iou_metrics=['mIoU'])

evaluator = dict(
    val_evaluator=evaluator_,
)


image_size = (1024, 1024)

data_preprocessor = dict(
    type='mmseg.SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=image_size

)

num_things_classes = 2
num_stuff_classes = 0
num_classes = num_things_classes + num_stuff_classes
prompt_shape = (90, 4)


model_cfg = dict(
    type='SegSAMPLerEmbMLP',
    hyperparameters=dict(
        optimizer=optimizer,
        param_scheduler=param_scheduler,
        evaluator=evaluator,
    ),
    need_train_names=sub_model_train,
    data_preprocessor=data_preprocessor,
    backbone=dict(
        #type='vit_h',
        #checkpoint ='pretrain/sam/sam_vit_h_4b8939.pth',#put the path of the pretrained model here
        # type='vit_b',
        # checkpoint='pretrain/sam/sam_vit_b_01ec64.pth',#put the path of the pretrained model here
        type='vit_l',
        checkpoint='/segment_anything/weights/sam_vit_l_0b3195.pth',#put the path of the pretrained model here
    ),


    seg_path_backbone=dict(
        type='mmseg.MixVisionTransformer',
        in_channels=3,
        embed_dims=32,
        num_stages=4,
        num_layers=[2, 2, 2, 2],
        num_heads=[1, 2, 5, 8],
        patch_sizes=[7, 3, 3, 3],
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3),
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1),
    seg_path_pretrained='/mmsegmentation-main/pretrain/mmseg_new.pth',##put the path of the SegFormer pretrained model here
    seg_path_decode_head=dict(
        type='mmseg.SegformerHead',
        in_channels=[32, 64, 160, 256],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=2,
        align_corners=False,
        loss_decode=dict(
            type='mmseg.CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    SAM_weights=0.1
)

exp_name = 'SegSAMPLerEmbMLP'
logger = dict(
    type='WandbLogger',
    project="UV-SAM",
    group='SAM',
    save_dir=f'results/{exp_name}/logs',
)



callbacks = [
    param_scheduler_callback,
    dict(
        type='ModelCheckpoint',
        dirpath=f'results/{exp_name}/checkpoints',
        save_last=True,
        mode='max',
        monitor='valmiou_0',
        save_top_k=2,
        filename='epoch_{epoch}-map_{valmiou_0:.4f}'
    ),
    dict(
        type='LearningRateMonitor',
        logging_interval='step'
    )
]


trainer_cfg = dict(
    compiled_model=False,
    devices=1,
    default_root_dir=f'results/{exp_name}',
    max_epochs=max_epochs,
    logger=logger,
    callbacks=callbacks,
    log_every_n_steps=20,
    check_val_every_n_epoch=1,
)


backend_args = None
train_pipeline = [
    dict(type='mmseg.LoadImageFromFile'),
    dict(type='mmseg.LoadAnnotations', reduce_zero_label=False),
    #dict(type='mmdet.Resize', scale=image_size),
    dict(
        type='mmseg.RandomResize',
        scale=(2048, 512),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),

    dict(type='mmseg.RandomCrop', crop_size=image_size, cat_max_ratio=0.75),
    dict(type='mmseg.Resize', scale=image_size, keep_ratio=True),
    dict(type='mmseg.RandomFlip', prob=0.5),
    dict(type='mmseg.PhotoMetricDistortion'),
    dict(type='mmseg.PackSegInputs')
]

test_pipeline = [
    dict(type='mmseg.LoadImageFromFile', backend_args=backend_args),
    dict(type='mmseg.Resize',scale=(1024, 1024), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='mmseg.LoadAnnotations',reduce_zero_label=False),
    dict(
        type='mmseg.PackSegInputs')
]

predict_pipeline = [
    dict(type='mmseg.Resize', scale=image_size),
    dict(
        type='mmseg.PackSegInputs',
        meta_keys=('ori_shape', 'img_shape', 'scale_factor'))
]
train_batch_size_per_gpu = 4
train_num_workers = 4
test_batch_size_per_gpu = 4
test_num_workers = 4
persistent_workers = True

data_parent = '../data/'


dataset_type = 'UVSegDataset'

val_loader = dict(
        batch_size=test_batch_size_per_gpu,
        num_workers=test_num_workers,
        persistent_workers=persistent_workers,            

        dataset=dict(
            type=dataset_type,
            data_root=data_parent,
            data_prefix=dict(img_path='image/val', seg_map_path= 'mask/val'),

            test_mode=True,
            pipeline=test_pipeline,
            backend_args=backend_args))

datamodule_cfg = dict(
    type='PLDataModule',
    train_loader=dict(
        batch_size=train_batch_size_per_gpu,
        num_workers=train_num_workers,
        persistent_workers=persistent_workers,
        dataset=dict(
            type=dataset_type,
            data_root=data_parent,
            data_prefix=dict(img_path='image/train' ,seg_map_path = 'mask/train'),
            pipeline=train_pipeline,
            backend_args=backend_args)
    ),
    val_loader=val_loader,
)