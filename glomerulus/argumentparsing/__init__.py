import argparse
import os

from argparse import ArgumentParser, BooleanOptionalAction
from .actions import *



def get_basic_parser(*args, **kwargs):
    parser = len(args) and isinstance(args[0], ArgumentParser) and args.pop(0) or ArgumentParser(*args, **kwargs)
    parser.add_argument('--verbose', '-v', action='count', default=0,
                        help='Increase verbosity (can be repeated)')
    parser.add_argument('--lr',  type=float, default=0.1,
                        help='Set base learning rate.')
    parser.add_argument('--net',
                        default='conv',
                        choices=['conv','fact'],
                        help="which convmodule to use")
    parser.add_argument('--freeze-spatial', action=BooleanOptionalAction, default=True,
                        help="freeze spatial filters for LearnableCov models")
    parser.add_argument('--freeze-channel', action=BooleanOptionalAction, default=True,
                        help="freeze channels for LearnableCov models")
    parser.add_argument('--spatial-init',
                        default='V1',
                        choices=['default', 'V1'],
                        help="initialization for spatial filters for LearnableCov models")
    parser.add_argument('--name', type=str, default='SingletonConv', 
                        help='filename for saved model')
    parser.add_argument('--load-model', type=str, default='conv_model_init.pt',
                        help='filename for loaded model')
    parser.add_argument('--seed',   type=int, default=0, help='training seed')
    tmpdir = os.environ.get('SLURM_TMPDIR',
             os.environ.get('TMPDIR', '/tmp'))
    parser.add_argument('--tmpdir', default=tmpdir, help='Temporary directory')
    return parser


def add_pytorch_group(parser, /):
    ptgroup = parser.add_argument_group('PyTorch', 'PyTorch-specific controls')
    ptgroup.add_argument('--benchmark',         action='store_true',                     help='Enable PyTorch benchmark mode')
    ptgroup.add_argument('--allow-cublas-tf32', action=BooleanOptionalAction,            help='Enable PyTorch TF32 mode (cuBLAS only)')
    ptgroup.add_argument('--allow-cudnn-tf32',  action=BooleanOptionalAction,            help='Enable PyTorch TF32 mode (cuDNN only)')
    ptgroup.add_argument('--allow-tf32',        action=MultiTargetBooleanOptionalAction, help='Enable PyTorch TF32 mode',
                        dest=('allow_cublas_tf32', 'allow_cudnn_tf32'))
    return ptgroup


def add_distributed_group(parser, /):
    distgroup = parser.add_argument_group('Distributed', 'PyTorch Distributed controls')
    distgroup.add_argument('--distributed', action='store_true', help='Enable PyTorch Distributed')
    distgroup.add_argument('--rank',        action=DistRankAction)
    distgroup.add_argument('--local-rank',  action=DistLocalRankAction)
    distgroup.add_argument('--world-size',  action=DistWorldSizeAction)
    distgroup.add_argument('--master-port', action=DistMasterPortAction)
    distgroup.add_argument('--master-addr', type=str, default=os.environ.get('MASTER_ADDR', '127.0.0.1'))
    return distgroup
