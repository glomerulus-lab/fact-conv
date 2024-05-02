import sys

from ..argumentparsing import *

argp = get_basic_parser(description='Simple Tool 2')
print(argp.parse_args(sys.argv[1:]))
