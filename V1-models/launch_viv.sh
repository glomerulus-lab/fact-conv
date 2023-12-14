#!/bin/bash

sbatch setoff_viv.sh True True V1 post V1_frozen_spatial_and_channel_post
sbatch setoff_viv.sh True False V1 post V1_frozen_spatial_post
sbatch setoff_viv.sh False True V1 post V1_frozen_channel_post
sbatch setoff_viv.sh False False V1 post V1_post
sbatch setoff_viv.sh True True V1 pre V1_frozen_spatial_and_channel_pre
sbatch setoff_viv.sh True False V1 pre V1_frozen_spatial_pre
sbatch setoff_viv.sh False True V1 pre V1_frozen_channel_pre
sbatch setoff_viv.sh False False V1 pre V1_pre

sbatch setoff_viv.sh True True default post default_frozen_spatial_and_channel_post
sbatch setoff_viv.sh True False default post default_frozen_spatial_post
sbatch setoff_viv.sh False True default post default_frozen_channel_post
sbatch setoff_viv.sh False False default post default_post
sbatch setoff_viv.sh True True default pre default_frozen_spatial_and_channel_pre
sbatch setoff_viv.sh True False default pre default_frozen_spatial_pre
sbatch setoff_viv.sh False True default pre default_frozen_channel_pre
sbatch setoff_viv.sh False False default pre default_pre
