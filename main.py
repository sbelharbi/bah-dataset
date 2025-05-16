import sys
from os.path import dirname, abspath, join, basename, expanduser, normpath

root_dir = dirname((abspath(__file__)))
sys.path.append(root_dir)

from parseit import parse_input
from parseit import Dict2Obj
# from instantiator import get_optimizer_for_params
import dllogger as DLLogger
from tools import fmsg
from tools import plot_tracker
from tools import state_dict_to_cpu
from tools import state_dict_to_gpu
from tools import MyDataParallel
from reproducibility import set_seed
from experiment import Experiment


if __name__ == '__main__':
    args, mode, eval_config = parse_input()

    args = Dict2Obj(args)

    exp = Experiment(args)
    exp.prepare()
    exp.run()
