#!/bin/bash

bash setoff_grad.sh False False V1 post Record_Resnet_V1_post | tee record_grad_LCLS_post.out
bash setoff_grad.sh False False V1 pre Record_Resnet_V1_pre | tee record_grad_LCLS_pre.out
#bash setoff_grad.sh False False V1 post Record_Resnet_V1_frozen_spatial_post | tee record_grad_LCLS_post.out
#bash setoff_grad.sh False False V1 pre Record_Resnet_V1_frozen_spatial_pre | tee record_grad_LCLS_pre.out
