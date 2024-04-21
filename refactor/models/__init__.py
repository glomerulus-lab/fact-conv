from .resnet import ResNet18
from .V1_models import V1_CIFAR10, V1_CIFAR100

def define_models(args):
    if args.net == 'resnet18':
        model = ResNet18()
    elif args.net == 'rsn_cifar10':
        model = V1_CIFAR10(hidden_dim=100, size=args.s,
                spatial_freq=args.f, scale=args.scale)
    elif args.net == 'rsn_cifar100':
        model = V1_CIFAR100()
    return model
