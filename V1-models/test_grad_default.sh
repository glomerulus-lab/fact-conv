#!/bin/bash

echo "*** Experiment 1 ***"
bash setoff_viv_grad.sh True True default post V1_frozen_spatial_and_channel_post
echo "*** Experiment 2 ***"
bash setoff_viv_grad.sh True False default post V1_frozen_spatial_post
echo "*** Experiment 3 ***"
bash setoff_viv_grad.sh False True default post V1_frozen_channel_post
echo "*** Experiment 4 ***"
bash setoff_viv_grad.sh False False default post V1_post
echo "*** Experiment 5 ***"
bash setoff_viv_grad.sh True True default pre V1_frozen_spatial_and_channel_pre
echo "*** Experiment 6 ***"
bash setoff_viv_grad.sh True False default pre V1_frozen_spatial_pre
echo "*** Experiment 7 ***"
bash setoff_viv_grad.sh False True default pre V1_frozen_channel_pre
echo "*** Experiment 8 ***"
bash setoff_viv_grad.sh False False default pre V1_pre
