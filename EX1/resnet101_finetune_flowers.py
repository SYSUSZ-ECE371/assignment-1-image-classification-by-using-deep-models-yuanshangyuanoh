_base_ = [
    '../_base_/models/resnet101.py',         # 模型架构
    '../_base_/datasets/imagenet_bs64.py',  # 数据加载
    '../_base_/schedules/imagenet_bs256.py', # 训练计划
    '../_base_/default_runtime.py'          # 默认运行时设置
]

model = dict(
    backbone=dict(
        frozen_stages=3,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmclassification/v0/resnet/resnet101_8xb32_in1k_20210831-539c63f8.pth',
            prefix='backbone',
        )),
    head=dict(
        num_classes=5,  # 修改为5个花卉类别
        topk=(1, ),     # 只评估top1准确率
        loss=dict(type='LabelSmoothLoss', label_smooth_val=0.1),  # 标签平滑
    )
)

data_preprocessor = dict(
    num_classes=5,
    # RGB格式的归一化参数(ImageNet统计值) 将像素值缩放到相近范围，加速模型收敛
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # 将图像从BGR转换为RGB
    to_rgb=True,
)

bgr_mean = data_preprocessor['mean'][::-1]
bgr_std = data_preprocessor['std'][::-1]

train_pipeline = [
    # 加载图像
    dict(type='LoadImageFromFile'),
    # 以一定概率水平翻转图像
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    # 随机调整大小并裁剪到224
    dict(type='RandomResizedCrop', scale=224),
    dict(
        type='AutoAugment',
        policies='imagenet',
        hparams=dict(pad_val=[round(x) for x in bgr_mean])),
    # 随机擦除
    dict(
        type='RandomErasing',
        erase_prob=0.2,
        mode='rand',
        # 遮挡面积比例范围
        min_area_ratio=0.02,
        max_area_ratio=1 / 3,
        # 使用ImageNet统计值生成随机填充色
        fill_color=bgr_mean,
        fill_std=bgr_std),
    dict(type='PackClsInputs'),
]
test_pipeline = [
    # 加载图像
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=224),
    dict(type='PackClsInputs'),
]

train_dataloader = dict(
    batch_size=64,
    num_workers=4,
    dataset=dict(
        type='ImageNet',
        data_root='data/flower_dataset',
        ann_file='train.txt',
        data_prefix='train',
        classes='data/flower_dataset/classes.txt',
        pipeline=train_pipeline,
    )
)

val_dataloader = dict(
    batch_size=64,
    num_workers=4,
    dataset=dict(
        type='ImageNet',
        data_root='data/flower_dataset',
        ann_file='val.txt',
        data_prefix='val',
        classes='data/flower_dataset/classes.txt',
        pipeline=test_pipeline,
    )
)
test_dataloader = val_dataloader

optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001),
    # 梯度裁剪 稳定训练过程
    clip_grad=dict(max_norm=35, norm_type=2),
    # # batch size较小使用梯度累计
    # accumulative_counts=4
)

param_scheduler = [
    # 预热
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=True,
        begin=0,
        end=5,
        convert_to_iter_based=True),
    # 主要学习率策略
    dict(
        type='CosineAnnealingLR',
        T_max=295,
        eta_min=1.0e-6,
        by_epoch=True,
        begin=5,
        end=100),
    # 配置动量调整策略 使损失函数收敛更快
    dict(
        type='CosineAnnealingParamScheduler',
        param_name='weight_decay',
        eta_min=0.00001,
        by_epoch=True,
        begin=0,
        end=100)
]

train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1)

# 评估配置
val_evaluator = dict(type='Accuracy', topk=(1,))
test_evaluator = val_evaluator

load_from = 'checkpoints/resnet101_8xb32_in1k.pth'