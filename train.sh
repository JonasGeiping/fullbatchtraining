

## FB augmented training runs:
python train_with_gradient_descent.py name=baseline_sgd hyp=base_sgd

python train_with_gradient_descent.py name=fbaug_1 hyp=fb1
python train_with_gradient_descent.py name=fbaug_2 hyp=fb2
python train_with_gradient_descent.py name=fbaug_clip hyp=fbclip
python train_with_gradient_descent.py name=fbaug_gradreg_lr08 hyp=gradreg
python train_with_gradient_descent.py name=fbaug_highreg_lr08 hyp=gradreg data.batch_size=32

python train_with_gradient_descent.py name=fbaug_highreg_lr08_shuffle hyp=gradreg data.batch_size=32 hyp.shuffle=True


## FB fixed dataset:
# no augmentations:
python train_with_gradient_descent.py name=noaug_sgd data.augmentations_train= hyp=base_sgd

python train_with_gradient_descent.py name=fb_noaug_1 data.augmentations_train= hyp=fb1
python train_with_gradient_descent.py name=fb_noaug_2 data.augmentations_train= hyp=fb2
python train_with_gradient_descent.py name=fb_noaug_clip data.augmentations_train= hyp=fbclip
python train_with_gradient_descent.py name=fb_noaug_gradreg_lr08 data.augmentations_train= hyp=gradreg
python train_with_gradient_descent.py name=fb_noaug_highreg_lr08 data.augmentations_train= hyp=gradreg data.batch_size=32


# 10x CIFAR:
python train_with_gradient_descent.py name=SGD_10_CIFAR hyp=base_sgd data/db=LMDB data.augmentations_train= hyp.train_semi_stochastic=True

python train_with_gradient_descent.py name=fb_10_1 data/db=LMDB data.augmentations_train= hyp=fb1
python train_with_gradient_descent.py name=fb_10_2 data/db=LMDB data.augmentations_train=hyp=fb2
python train_with_gradient_descent.py name=fb_10_clip data/db=LMDB data.augmentations_train= hyp=fbclip
python train_with_gradient_descent.py name=fb_10_gradreg_lr08 data/db=LMDB data.augmentations_train= hyp=gradreg
python train_with_gradient_descent.py name=fb_10_highreg_lr08 data/db=LMDB data.augmentations_train= hyp=gradreg data.batch_size=32

#40x CIFAR:
python train_with_gradient_descent.py name=fb_40_gradreg_lr08 data/db=LMDB data.augmentations_train= data.db.rounds=40 hyp=gradreg


# Use checkpointing or multi-GPUs setups to finish the later settings in a reasonable time. Both are implemented and more info can be found in the config folder.
