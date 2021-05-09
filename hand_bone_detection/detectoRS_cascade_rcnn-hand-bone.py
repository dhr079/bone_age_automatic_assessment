_base_ = '../../configs/detectors/detectors_cascade_rcnn_r50_1x_coco.py'
# _base_ = '../../configs/detectors/detectors_cascade_rcnn_r50_2e_coco.py'


model = dict(
#     pretrained=None,

    rpn_head=dict(
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[4],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 12, 16, 20])))

img_norm_cfg = dict(
#     mean=[125.], std=[58.5], to_rgb=False)
#     mean=[153.467, 153.467, 153.467], std=[63.978, 63.978, 63.978], to_rgb=False)    ## pung
    mean=[125., 125., 125.], std=[58.5, 58.5, 58.5], to_rgb=False)    ## pneumonia

train_pipeline = [
    dict(type='LoadImageFromFile', ),
#     dict(type='LoadImageFromFile',  color_type='grayscale'),False
    dict(type='LoadAnnotations', with_bbox=True),
#     dict(type='Resize', img_scale=(1400, 1024), keep_ratio=True),
    dict(type='Resize', img_scale=(600, 600), keep_ratio=False),
#     dict(type='ToTensor',keys=['img']),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
#     dict(type='AddGaussianNoise', mean=0., std=0.005),     ##
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
#         img_scale=(1400, 1024),
        img_scale=(600, 600),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
#             dict(type='RandomFlip'),
            dict(type='RandomFlip', flip_ratio=0.0),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=1,
    train=dict(
        pipeline=train_pipeline),
    val=dict(
        pipeline=test_pipeline),
    test=dict(
        pipeline=test_pipeline))