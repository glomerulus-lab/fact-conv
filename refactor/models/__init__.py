from .resnet import ResNet18
from .function_utils import replace_layers_factconv2d, turn_off_grad, replace_layers_scale, init_V1_layers


def define_models(args):
    if 'resnet18' in args.net:
       model = ResNet18()
    if 'fact' in args.net:
       replace_layers_factconv2d(model)
    if "v1" in args.net:
        init_V1_layers(model, bias=False)
    if "us" in args.net:
        turn_off_grad(model, "spatial")
    if "uc" in args.net:
        turn_off_grad(model, "channel")
    if args.width != 1:
        replace_layers_scale(model, args.width)
    return model
