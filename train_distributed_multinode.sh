# Experiment names in their respective table are included as comments:

## FB augmented training runs (larger models):
# python train_with_gradient_descent.py name=fbaug_1_resnet152 hyp=fb1 model=resnet152 impl/setup=distributed impl.setup.rank=SLURM impl.setup.world_size=1 impl.setup.url=env:// impl.checkpoint.name=fbaug_1_resnet152
# python train_with_gradient_descent.py name=fbaug_2_resnet152 hyp=fb2 model=resnet152 impl/setup=distributed impl.setup.rank=SLURM impl.setup.world_size=1 impl.setup.url=env:// impl.checkpoint.name=fbaug_2_resnet152
# python train_with_gradient_descent.py name=fbaug_clip_resnet152 hyp=fbclip model=resnet152 impl/setup=distributed impl.setup.rank=SLURM impl.setup.world_size=1 impl.setup.url=env:// impl.checkpoint.name=fbaug_clip_resnet152
# python train_with_gradient_descent.py name=fbaug_gradreg_lr08_resnet152 hyp=gradreg model=resnet152 impl/setup=distributed impl.setup.rank=SLURM impl.setup.world_size=1 impl.setup.url=env:// impl.checkpoint.name=fbaug_gradreg_lr08_resnet152
# python train_with_gradient_descent.py name=fbaug_highreg_lr08_resnet152 hyp=gradreg data.batch_size=32 model=resnet152 impl/setup=distributed impl.setup.rank=SLURM impl.setup.world_size=1 impl.setup.url=env:// impl.checkpoint.name=fbaug_highreg_lr08_resnet152
# python train_with_gradient_descent.py name=fbaug_highreg_lr08_shuffle_resnet152 hyp=gradreg data.batch_size=32 hyp.shuffle=True model=resnet152 impl/setup=distributed impl.setup.rank=SLURM impl.setup.world_size=1 impl.setup.url=env:// impl.checkpoint.name=fbaug_highreg_lr08_shuffle_resnet152
#
# python train_with_gradient_descent.py name=fbaug_1_densenet121 hyp=fb1 model=densenet121 impl/setup=distributed impl.setup.rank=SLURM impl.setup.world_size=1 impl.setup.url=env:// impl.checkpoint.name=fbaug_1_densenet121
# python train_with_gradient_descent.py name=fbaug_2_densenet121 hyp=fb2 model=densenet121 impl/setup=distributed impl.setup.rank=SLURM impl.setup.world_size=1 impl.setup.url=env:// impl.checkpoint.name=fbaug_2_densenet121
# python train_with_gradient_descent.py name=fbaug_clip_densenet121 hyp=fbclip model=densenet121 impl/setup=distributed impl.setup.rank=SLURM impl.setup.world_size=1 impl.setup.url=env:// impl.checkpoint.name=fbaug_clip_densenet121
# python train_with_gradient_descent.py name=fbaug_gradreg_lr08_densenet121 hyp=gradreg model=densenet121 impl/setup=distributed impl.setup.rank=SLURM impl.setup.world_size=1 impl.setup.url=env:// impl.checkpoint.name=fbaug_gradreg_lr08_densenet121
# python train_with_gradient_descent.py name=fbaug_highreg_lr08_densenet121 hyp=gradreg data.batch_size=32 model=densenet121 impl/setup=distributed impl.setup.rank=SLURM impl.setup.world_size=1 impl.setup.url=env:// impl.checkpoint.name=fbaug_highreg_lr08_densenet121
# python train_with_gradient_descent.py name=fbaug_highreg_lr08_shuffle_densenet121 hyp=gradreg data.batch_size=32 hyp.shuffle=True model=densenet121 impl/setup=distributed impl.setup.rank=SLURM impl.setup.world_size=1 impl.setup.url=env:// impl.checkpoint.name=fbaug_highreg_lr08_shuffle_densenet121
#
# python train_with_gradient_descent.py name=fbaug_1_resnet50 hyp=fb1 model=resnet50 impl/setup=distributed impl.setup.rank=SLURM impl.setup.world_size=1 impl.setup.url=env:// impl.checkpoint.name=fbaug_1_resnet50
# python train_with_gradient_descent.py name=fbaug_2_resnet50 hyp=fb2 model=resnet50 impl/setup=distributed impl.setup.rank=SLURM impl.setup.world_size=1 impl.setup.url=env:// impl.checkpoint.name=fbaug_2_resnet50
# python train_with_gradient_descent.py name=fbaug_clip_resnet50 hyp=fbclip model=resnet50 impl/setup=distributed impl.setup.rank=SLURM impl.setup.world_size=1 impl.setup.url=env:// impl.checkpoint.name=fbaug_clip_resnet50
# python train_with_gradient_descent.py name=fbaug_gradreg_lr08_resnet50 hyp=gradreg model=resnet50 impl/setup=distributed impl.setup.rank=SLURM impl.setup.world_size=1 impl.setup.url=env:// impl.checkpoint.name=fbaug_gradreg_lr08_resnet50
# python train_with_gradient_descent.py name=fbaug_highreg_lr08_resnet50 hyp=gradreg data.batch_size=32 model=resnet50 impl/setup=distributed impl.setup.rank=SLURM impl.setup.world_size=1 impl.setup.url=env:// impl.checkpoint.name=fbaug_highreg_lr08_resnet50
# python train_with_gradient_descent.py name=fbaug_highreg_lr08_shuffle_resnet50 hyp=gradreg data.batch_size=32 hyp.shuffle=True model=resnet50 impl/setup=distributed impl.setup.rank=SLURM impl.setup.world_size=1 impl.setup.url=env:// impl.checkpoint.name=fbaug_highreg_lr08_shuffle_resnet50
#


## FB fixed dataset (long running jobs)

# 10x CIFAR:
# python train_with_gradient_descent.py name=SGD_10_CIFAR hyp=base_sgd data/db=LMDB data.augmentations_train= data.db.rounds=10 hyp.train_semi_stochastic=True # Baseline SGD

python train_with_gradient_descent.py name=fb_10_1 data/db=LMDB data.augmentations_train= data.db.rounds=10 hyp=fb1 impl/setup=distributed impl.setup.rank=SLURM impl.setup.world_size=1 impl.setup.url=env:// impl.checkpoint.name=fb_10_1
python train_with_gradient_descent.py name=fb_10_2 data/db=LMDB data.augmentations_train= data.db.rounds=10 hyp=fb2 impl/setup=distributed impl.setup.rank=SLURM impl.setup.world_size=1 impl.setup.url=env:// impl.checkpoint.name=fb_10_2
python train_with_gradient_descent.py name=fb_10_clip data/db=LMDB data.augmentations_train= data.db.rounds=10 hyp=fbclip impl/setup=distributed impl.setup.rank=SLURM impl.setup.world_size=1 impl.setup.url=env:// impl.checkpoint.name=fb_10_clip
python train_with_gradient_descent.py name=fb_10_gradreg_lr08 data/db=LMDB data.augmentations_train= data.db.rounds=10 hyp=gradreg  impl/setup=distributed impl.setup.rank=SLURM impl.setup.world_size=1 impl.setup.url=env:// impl.checkpoint.name=fb_10_gradreg_lr08
python train_with_gradient_descent.py name=fb_10_highreg_lr08 data/db=LMDB data.augmentations_train= data.db.rounds=10 hyp=gradreg data.batch_size=32   impl/setup=distributed impl.setup.rank=SLURM impl.setup.world_size=1 impl.setup.url=env:// impl.checkpoint.name=fb_10_highreg_lr08

#40x CIFAR:
# python train_with_gradient_descent.py name=SGD_10_CIFAR data/db=LMDB data.augmentations_train= data.db.rounds=40 hyp=base_sgd hyp.train_semi_stochastic=True
#
python train_with_gradient_descent.py name=fb_40_1 data/db=LMDB data.augmentations_train= data.db.rounds=40 hyp=fb1 impl/setup=distributed impl.setup.rank=SLURM impl.setup.world_size=1 impl.setup.url=env:// impl.checkpoint.name=fb_40_1
python train_with_gradient_descent.py name=fb_40_2 data/db=LMDB data.augmentations_train= data.db.rounds=40 hyp=fb2 impl/setup=distributed impl.setup.rank=SLURM impl.setup.world_size=1 impl.setup.url=env:// impl.checkpoint.name=fb_40_2
python train_with_gradient_descent.py name=fb_40_clip data/db=LMDB data.augmentations_train= data.db.rounds=40 hyp=fbclip impl/setup=distributed impl.setup.rank=SLURM impl.setup.world_size=1 impl.setup.url=env:// impl.checkpoint.name=fb_40_clip
python train_with_gradient_descent.py name=fb_40_gradreg_lr08 data/db=LMDB data.augmentations_train= data.db.rounds=40 hyp=gradreg impl/setup=distributed impl.setup.rank=SLURM impl.setup.world_size=1 impl.setup.url=env:// impl.checkpoint.name=fb_40_gradreg_lr08
python train_with_gradient_descent.py name=fb_40_highreg_lr08 data/db=LMDB data.augmentations_train= data.db.rounds=40 hyp=gradreg data.batch_size=32 impl/setup=distributed impl.setup.rank=SLURM impl.setup.world_size=1 impl.setup.url=env:// impl.checkpoint.name=fb_40_highreg_lr08


# Use checkpointing or multi-GPUs setups to finish the later settings in a reasonable time.
# Both are implemented and more info can be found in the config folder.
