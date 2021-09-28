

## FB augmented training runs:
python train_with_gradient_descent.py name=baseline_sgd hyp.train_stochastic=True hyp.steps=300 data.batch_size=128 impl.shuffle=True

python train_with_gradient_descent.py name=fbaug_1 hyp.train_stochastic=False hyp.steps=300 data.batch_size=128
python train_with_gradient_descent.py name=fbaug_2 hyp.train_stochastic=False hyp.steps=3000 data.batch_size=128 hyp.optim.lr=0.4 hyp.warmup=400 hyp.scheduler=cosine-4000
python train_with_gradient_descent.py name=fbaug_clip hyp.train_stochastic=False hyp.steps=3000 data.batch_size=128 hyp.optim.lr=0.4 hyp.warmup=400 hyp.scheduler=cosine-4000 hyp.grad_clip=0.25
python train_with_gradient_descent.py name=fbaug_gradreg_lr08 hyp.train_stochastic=False hyp.steps=3000 data.batch_size=128 hyp.optim.lr=0.8 hyp.warmup=400 hyp.scheduler=cosine-4000 hyp.grad_reg.block_strength=0.5 hyp.grad_clip=0.25
python train_with_gradient_descent.py name=fbaug_highreg_lr08 hyp.train_stochastic=False hyp.steps=3000 data.batch_size=32 hyp.optim.lr=0.8 hyp.warmup=400 hyp.scheduler=cosine-4000 hyp.grad_reg.block_strength=0.5 hyp.grad_clip=0.25

python train_with_gradient_descent.py name=fbaug_highreg_lr08_shuffle hyp.train_stochastic=False hyp.steps=3000 data.batch_size=32 hyp.optim.lr=0.8 hyp.warmup=400 hyp.scheduler=cosine-4000 hyp.grad_reg.block_strength=0.5 hyp.grad_clip=0.25 impl.shuffle=True


## FB fixed dataset:
# no augmentations:
python train_with_gradient_descent.py name=noaug_sgd data.augmentations_train= hyp.train_stochastic=True hyp.steps=300 data.batch_size=128 impl.shuffle=True

python train_with_gradient_descent.py name=fb_noaug_1 data.augmentations_train= hyp.train_stochastic=False hyp.steps=300 data.batch_size=128
python train_with_gradient_descent.py name=fb_noaug_2 data.augmentations_train= hyp.train_stochastic=False hyp.steps=3000 data.batch_size=128 hyp.optim.lr=0.4 hyp.warmup=400 hyp.scheduler=cosine-4000
python train_with_gradient_descent.py name=fb_noaug_clip data.augmentations_train= hyp.train_stochastic=False hyp.steps=3000 data.batch_size=128 hyp.optim.lr=0.4 hyp.warmup=400 hyp.scheduler=cosine-4000 hyp.grad_clip=0.25
python train_with_gradient_descent.py name=fb_noaug_gradreg_lr08 data.augmentations_train= hyp.train_stochastic=False hyp.steps=3000 data.batch_size=128 hyp.optim.lr=0.4 hyp.warmup=400 hyp.scheduler=cosine-4000 hyp.grad_reg.block_strength=0.5 hyp.grad_clip=0.25
python train_with_gradient_descent.py name=fb_noaug_highreg_lr08 data.augmentations_train= hyp.train_stochastic=False hyp.steps=3000 data.batch_size=32 hyp.optim.lr=0.4 hyp.warmup=400 hyp.scheduler=cosine-4000 hyp.grad_reg.block_strength=0.5 hyp.grad_clip=0.25


# 10x CIFAR:
python train_with_gradient_descent.py name=SGD_10_CIFAR data/db=LMDB data.augmentations_train= hyp.train_stochastic=True hyp.train_semi_stochastic=True hyp.steps=300 data.batch_size=128 impl.shuffle=True

python train_with_gradient_descent.py name=fb_10_1 data/db=LMDB data.augmentations_train= hyp.train_stochastic=False hyp.steps=300 data.batch_size=128
python train_with_gradient_descent.py name=fb_10_2 data/db=LMDB data.augmentations_train= hyp.train_stochastic=False hyp.steps=3000 data.batch_size=128 hyp.optim.lr=0.4 hyp.warmup=400 hyp.scheduler=cosine-4000
python train_with_gradient_descent.py name=fb_10_clip data/db=LMDB data.augmentations_train= hyp.train_stochastic=False hyp.steps=3000 data.batch_size=128 hyp.optim.lr=0.4 hyp.warmup=400 hyp.scheduler=cosine-4000 hyp.grad_clip=0.25
python train_with_gradient_descent.py name=fb_10_gradreg_lr08 data/db=LMDB data.augmentations_train= hyp.train_stochastic=False hyp.steps=3000 data.batch_size=128 hyp.optim.lr=0.8 hyp.warmup=400 hyp.scheduler=cosine-4000 hyp.grad_reg.block_strength=0.5 hyp.grad_clip=0.25
python train_with_gradient_descent.py name=fb_10_highreg_lr08 data/db=LMDB data.augmentations_train= hyp.train_stochastic=False hyp.steps=3000 data.batch_size=32 hyp.optim.lr=0.8 hyp.warmup=400 hyp.scheduler=cosine-4000 hyp.grad_reg.block_strength=0.5 hyp.grad_clip=0.25

#40x CIFAR:
python train_with_gradient_descent.py name=fb_40_gradreg_lr08 data/db=LMDB data.augmentations_train= data.db.rounds=40 hyp.train_stochastic=False hyp.steps=3000 data.batch_size=128 hyp.optim.lr=0.8 hyp.warmup=400 hyp.scheduler=cosine-4000 hyp.grad_reg.block_strength=0.5 hyp.grad_clip=0.25


# Use checkpointing or multi-GPUs setups to finish the later settings in a reasonable time. Both are implemented and more info can be found in the config folder.
