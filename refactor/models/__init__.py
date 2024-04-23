from .resnet import ResNet18
from .function_utils import replace_layers_keep_weight, turn_off_grada, replace_layers_scale
def define_models(args):
    if 'resnet18' in args.net:
       model = ResNet18()
    if 'fact' in args.net:
       replace_layers_keep_weight(model)
    if args.spatial_init == 'V1':
        # TODO: import V1_init function from structured-random-features
        V1_init(model)
    if args.freeze_spatial == True: 
        # TODO: make turn_off_grad function
        turn_off_grad(model, "spatial")
    if args.freeze_channel == True:
        turn_off_grad(model, "channel")
    if args.width != 1:
        replace_layers_scale(model, args.width)
    return model
