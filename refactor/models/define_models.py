from .resnet import ResNet18

def define_models(args):
    if args.net == 'resnet18':
        model = ResNet18()
    return model
