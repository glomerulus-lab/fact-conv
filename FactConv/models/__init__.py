from .resnet import ResNet18, ResNet9
from .switched_resnet import SwitchedResNet18, SwitchedResNet9,\
SwitchedResNet34, SwitchedResNet50
from .aligned_resnet import AlignedResNet18, AlignedResNet9, AlignedResNet34,\
AlignedResNet50
from .pre_bn_resnet import PostBNResNet18, PostBNResNet9
from .function_utils import replace_layers_factconv2d,replace_layers_factprojconv2d,\
replace_layers_diagfactconv2d, replace_layers_diagchanfactconv2d,\
turn_off_covar_grad, replace_layers_scale, init_V1_layers,\
replace_layers_resample_align, replace_layers_offfactconv2d,\
replace_layers_bias, replace_layers_gmm, replace_layers_eighconv2d,\
replace_layers_rqfreeconv2d, replace_layers_rqalignedconv2d, \
replace_layers_rqfreeleftconv2d, replace_layers_rqfreeleftalignedconv2d, \
replace_layers_eighalignedconv2d, replace_layers_eighfixedconv2d,\
replace_layers_eighfixedalignedconv2d, replace_layers_lowrank,\
replace_layers_lrdiag


def define_models(args):
    if 'resnet18' in args.net:
       model = ResNet18()
    if 'resnet9' in args.net:
       model = ResNet9()
    if 'pre_bn_resnet18' in args.net:
        model = SwitchedResNet18()
    if 'pre_bn_resnet9' in args.net:
        model = SwitchedResNet9()
    if 'pre_bn_resnet34' in args.net:
        model = SwitchedResNet34()
    if 'pre_bn_resnet50' in args.net:
        model = SwitchedResNet50()
    if 'pre_bn_aligned_resnet18' in args.net:
        model = AlignedResNet18()
    if 'pre_bn_aligned_resnet9' in args.net:
        model = AlignedResNet9()
    if 'pre_bn_aligned_resnet34' in args.net:
        model = AlignedResNet34()
    if 'pre_bn_aligned_resnet50' in args.net:
        model = AlignedResNet50()
    if 'post_bn_aligned_resnet18' in args.net:
        model = PostBNResNet18()
    if 'post_bn_aligned_resnet9' in args.net:
        model = PostBNResNet9()

    if args.width != 1:
        replace_layers_scale(model, args.width)
    if args.bias == 1:
        replace_layers_bias(model)
    if args.gmm >= 1:
        replace_layers_gmm(model, args.gmm)

    if 'fact' in args.net:
       replace_layers_factconv2d(model)
    if 'lowrank' in args.net:        
        replace_layers_lowrank(model, args.channel_k)
    if 'lrdiag' in args.net:        
        replace_layers_lrdiag(model, args.channel_k)

    if 'eigh' in args.net:
       replace_layers_eighconv2d(model)

    if 'eigh_fixed' in args.net:
       replace_layers_eighfixedconv2d(model)

    if 'rqfree' in args.net:
       replace_layers_rqfreeconv2d(model)

    if 'eigh' in args.net and 'aligned' in args.net:
       replace_layers_eighalignedconv2d(model)

    if 'eigh_fixed' in args.net and 'aligned' in args.net:
       replace_layers_eighfixedalignedconv2d(model)
 
    if 'rqfree' in args.net and 'aligned' in args.net:
       replace_layers_rqalignedconv2d(model)

    if 'rqfree' in args.net and 'lefthand' in args.net:
       replace_layers_rqfreeleftconv2d(model)

    if 'rqfree' in args.net and 'lefthand' in args.net and 'aligned' in args.net:
       replace_layers_rqfreeleftalignedconv2d(model)

    if 'proj' in args.net:
       replace_layers_factprojconv2d(model)
    # if 'diag' in args.net:
    #     replace_layers_diagfactconv2d(model)
    #if 'off' in args.net:
    #    replace_layers_offfactconv2d(model)
    if 'diagchan' in args.net:
        replace_layers_diagchanfactconv2d(model)
    if "v1" in args.net:
        init_V1_layers(model, bias=False)
    if "us" in args.net:
        turn_off_covar_grad(model, "spatial")
    if "uc" in args.net:
        turn_off_covar_grad(model, "channel")
    #if "resample" in args.net:
    #    replace_layers_resample_align(model, args.rank)

    return model
