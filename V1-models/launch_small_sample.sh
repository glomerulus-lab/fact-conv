#!/bin/bash

sbatch setoff_small_sample.sh True True V1 post Resnet_V1_frozen_spatial_and_channel_post
sbatch setoff_small_sample.sh True False V1 post Resnet_V1_frozen_spatial_post
sbatch setoff_small_sample.sh False True V1 post Resnet_V1_frozen_channel_post
sbatch setoff_small_sample.sh False False V1 post Resnet_V1_post
sbatch setoff_small_sample.sh True True V1 pre Resnet_V1_frozen_spatial_and_channel_pre
sbatch setoff_small_sample.sh True False V1 pre Resnet_V1_frozen_spatial_pre
sbatch setoff_small_sample.sh False True V1 pre Resnet_V1_frozen_channel_pre
sbatch setoff_small_sample.sh False False V1 pre Resnet_V1_pre

sbatch setoff_small_sample.sh True True default post Resnet_default_frozen_spatial_and_channel_post
sbatch setoff_small_sample.sh True False default post Resnet_default_frozen_spatial_post
sbatch setoff_small_sample.sh False True default post Resnet_default_frozen_channel_post
sbatch setoff_small_sample.sh False False default post Resnet_default_post
sbatch setoff_small_sample.sh True True default pre Resnet_default_frozen_spatial_and_channel_pre
sbatch setoff_small_sample.sh True False default pre Resnet_default_frozen_spatial_pre
sbatch setoff_small_sample.sh False True default pre Resnet_default_frozen_channel_pre
sbatch setoff_small_sample.sh False False default pre Resnet_default_pre
