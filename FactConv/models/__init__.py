from .resnet import ResNet18
from .LC_models import LC_CIFAR10
from .function_utils import replace_layers_factconv2d,\
replace_layers_diagfactconv2d, replace_layers_diagchanfactconv2d,\
turn_off_covar_grad, replace_layers_scale, init_V1_layers,\
replace_layers_lowrank, replace_layers_lowrankplusdiag,\
replace_layers_lowrankK1, replace_layers_offdiag


def define_models(args):
    if 'resnet18' in args.net:
       model = ResNet18()
    if 'rsn' in args.net:
        model = LC_CIFAR10(hidden_dim=100, size=2, spatial_freq=0.1, scale=1,
                bias=True, freeze_spatial=False, freeze_channel=False,
                spatial_init='V1')
    if args.width != 1:
        replace_layers_scale(model, args.width)
    if 'fact' in args.net:
       replace_layers_factconv2d(model)
    if 'diag' in args.net:
        replace_layers_diagfactconv2d(model)
    if 'diagchan' in args.net:
        replace_layers_diagchanfactconv2d(model)
    if 'lowrank' in args.net:
        replace_layers_lowrank(model, args.spatial_k, args.channel_k)
    if 'lr-diag' in args.net:
        replace_layers_lowrankplusdiag(model)
    if 'lr-K1' in args.net:
        replace_layers_lowrankK1(model, args.channel_k)
    if 'offdiag' in args.net:
        replace_layers_offdiag(model)
    if 'v1' in args.net:
        init_V1_layers(model, bias=False)
    if 'us' in args.net:
        turn_off_covar_grad(model, 'spatial')
        print('US')
    if 'uc' in args.net:
        turn_off_covar_grad(model, 'channel')
        print('UC')

    return model
