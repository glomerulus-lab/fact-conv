from .resnet import ResNet18
from .function_utils import replace_layers_keep_weight, turn_off_grada, replace_layers_scale
def define_models(args):
    if 'resnet18' in args.net:
       model = ResNet18()
    if 'fact' in args.net:
       replace_layers_keep_weight(model)
    if "v1" in args.net:
        # TODO: import V1_init function from structured-random-features
        V1_init(model)
    if "us" in args.net: 
        # TODO: make turn_off_grad function
        turn_off_grad(model, "spatial")
    if "uc" in args.net:
        turn_off_grad(model, "channel")
    if args.width != 1:
        replace_layers_scale(model, args.width)
    return model
