# dataset settings
dataset_type = "W300LP"
img_norm_cfg = dict(mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                    std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
                    to_rgb=True)
train_pipeline = [
    dict(type="LoadImageFromFileWLP"),
    dict(type="Resize", size=240),
    dict(type="RandomCrop", size=224),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="ImageToTensor", keys=["img"]),
    dict(type="ToTensor", keys=["gt_labels", "gt_angles"]),
    dict(type="Collect", keys=["img", "gt_labels", "gt_angles"])
]
test_pipeline = [
    dict(type="LoadImageFromFileWLP"),
    dict(type="Resize", size=240),
    dict(type="CenterCrop", size=224),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="ImageToTensor", keys=["img"]),
    dict(type="ToTensor", keys=["gt_labels", "gt_angles"]),
    dict(type="Collect", keys=["img", "gt_labels", "gt_angles"])
]
data = dict(samples_per_gpu=128,
            workers_per_gpu=1,
            train=dict(type=dataset_type,
                       data_prefix="./data/300W_LP",
                       pipeline=train_pipeline),
            val=dict(type=dataset_type,
                     data_prefix="./data/300W_LP",
                     pipeline=train_pipeline),
            test=dict(type=dataset_type,
                      data_prefix="./data/300W_LP",
                      pipeline=train_pipeline))

# model settings
model = dict(type="HopeNet",
             backbone=dict(type="ShuffleNetV2", widen_factor=1.0),
             neck=dict(type="GlobalAveragePooling"),
             head=dict(type="HopenetHead",
                       in_channels=1024,
                       num_bins=67,
                       alpha=0.001),
             norm_eval=True)

# optimizer
optimizer = dict(
    type="Adam",
    lr=0.001,
    weight_decay=0.0005,
    paramwise_cfg=dict(custom_keys={".head": dict(lr_multi=5)}))
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy="fixed")
total_epochs = 100

# checkpoint saving
checkpoint_config = dict(interval=10, create_symlink=False)
# yapf:disable
log_config = dict(
    interval=100,
    hooks=[
        dict(type="TextLoggerHook"),
        dict(type="TensorboardLoggerHook")
    ])
# yapf:enable
evaluation = dict(interval=10)  # do evaluation every 'interval' epoches
dist_params = dict(backend="nccl")
log_level = "INFO"
load_from = None
resume_from = None
workflow = [("train", 1)]
