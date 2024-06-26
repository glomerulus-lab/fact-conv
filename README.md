# Learning and Aligning Structured Random Feature Networks
This is the repo for the Learning and Aligning Structured Random Feature Networks paper, accepted at the ICLR Re-Align 2024 workshop. The code for our factorized covariance and rainbow sampling methods can be found in the `FactConv` directory and the code for Randomized Scattering Networks can be found in `RSN_experiments`. 

## Factorized Random Feature Convolutions
Our Factorized Covariance module is located in `FactConv/conv_modules.py` and can be called similarly to a nn.Conv2d module, like so `m = FactConv2d(in_channels=3, out_channels=32, kernel_size=(3,3))`. 

V1 initalization based on the receptive V1 field of mice is located in `FactConv/V1_covariance.py` and can be called like so; `V1_init(m, size=2, spatial_freq=0.1, scale=1, center=center)` where `center=((m.kernel_size[0]-1)/2, (m.kernel_size[1]-1)/2)`. 

Run `python3 setup.py install` to install the Factored Covariance module

## Rainbow Sampling
Rainbow sampling can be done with our `RainbowSampler` class object as so:

```
from rainbow import RainbowSampler
R = RainbowSampler(net, trainloader)
rainbow_net = R.sample()
```

To use our factorized ResNet in our rainbow sampling procedure as outlined in "Learning and Aligning Structured Random Feature Networks" by White et al., specify `RainbowSampler(..., sampling='structured_alignment', wa=True, in_wa=True, aca=True)`. This can be specified for both FactConv2d and nn.Conv2d modules.

To do the rainbow sampling procedure of "A Rainbow in Deep Network Black Boxes" by Guth et al., specify `RainbowSampler(..., sampling='cc_specification', wa=False, aca=True)`. This is specified specifically for networks using nn.Conv2d modules. 

[This google drive link](https://drive.google.com/drive/folders/10nCfKAoX_ghrm4G-oxbZg2_UIyNyaL1W?usp=sharing) contains our pretrained ResNet18 networks, with both convolutional and factorized convolutional varations.

# Cite Us

If you found this repository or our paper helpful, please cite us as shown below:

```bibtex
@inproceedings{
white2024learning,
title={Learning and Aligning Structured Random Feature Networks},
author={Vivian White and Muawiz Sajjad Chaudhary and Guy Wolf and Guillaume Lajoie and Kameron Decker Harris},
booktitle={ICLR 2024 Workshop on Representational Alignment},
year={2024},
url={https://openreview.net/forum?id=vWhUQXQoFF}
}
```
