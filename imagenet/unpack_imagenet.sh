#!/bin/bash
mkdir -p $SLURM_TMPDIR/imagenet/train
cd $SLURM_TMPDIR/imagenet/train
tar  -xf /home/whitev4/scratch/imagenet/ILSVRC2012_img_train.tar --to-command='mkdir ${TAR_REALNAME%.tar}; tar -xC ${TAR_REALNAME%.tar}'

mkdir -p $SLURM_TMPDIR/imagenet/val
cd $SLURM_TMPDIR/imagenet/val
tar  -xf /home/whitev4/scratch/imagenet/ILSVRC2012_img_val.tar
cp /home/whitev4/v1-models/V1-models/imagenet/valprep.sh $SLURM_TMPDIR/imagenet/val/valprep.sh
./valprep.sh
