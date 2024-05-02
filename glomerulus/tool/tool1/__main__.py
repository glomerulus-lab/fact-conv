import sys

from ...argumentparsing import *

argp = get_basic_parser(description='Complex Tool 1')
add_pytorch_group(argp)
add_distributed_group(argp)
print(argp.parse_args(sys.argv[1:]))
