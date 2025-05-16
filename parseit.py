# Sel-contained-as-possible module handles parsing the input using argparse.
# handles seed, and initializes some modules for reproducibility.
import copy
import os
from os.path import dirname, abspath, join, basename, expanduser, normpath
import sys
import argparse
from copy import deepcopy
import warnings
import subprocess
import fnmatch
import glob
import shutil
import datetime as dt
from typing import Tuple
import socket
import getpass

import yaml
import json
import numpy as np
import torch

import constants

root_dir = dirname((abspath(__file__)))
sys.path.append(root_dir)

import reproducibility

from dllogger import ArbJSONStreamBackend
from dllogger import Verbosity
from dllogger import ArbStdOutBackend
from dllogger import ArbTextStreamBackend
import dllogger as DLLogger
from tools import fmsg

from default_config import get_config
from default_config import get_root_wsol_dataset


def mkdir(fd):
    if not os.path.isdir(fd):
        os.makedirs(fd, exist_ok=True)


def find_files_pattern(fd_in_, pattern_):
    assert os.path.exists(fd_in_), "Folder {} does not exist.".format(fd_in_)
    files = []
    for r, d, f in os.walk(fd_in_):
        for file in f:
            if fnmatch.fnmatch(file, pattern_):
                files.append(os.path.join(r, file))

    return files


def str2bool(v):
    if isinstance(v, bool):
        return v

    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def null_str(v):
    if v in [None, '', 'None']:
        return 'None'

    if isinstance(v, str):
        return v

    raise NotImplementedError(f"{v}, type: {type(v)}")


class Dict2Obj(object):
    def __init__(self, dictionary):
        for key in dictionary.keys():
            setattr(self, key, dictionary[key])

    def __repr__(self):
        attrs = str([x for x in self.__dict__])
        return "<Dict2Obj: %s" % attrs


def get_args(args: dict) -> dict:
    parser = argparse.ArgumentParser()

    # JMT parsing --------------------------------------------------------------
    # 1. Experiment Setting
    # 1.2. Paths
    parser.add_argument('--dataset_path',
                        default=None, type=str,
                        help='The root directory of the preprocessed dataset.'
                             'This has been automatically set.')
    parser.add_argument('--load_path',
                        default=None,
                        type=str,
                        help='The path to load the trained models, '
                             'such as the backbone.')
    parser.add_argument('--save_path',
                        default='/something/ABAW3/weights_saved',
                        type=str,
                        help='The path to save the trained models ')

    parser.add_argument('--visual_backbone_path',
                        default=None,
                        type=null_str,
                        help='The path to save the visual backbone.')

    # End JMT parsing ----------------------------------------------------------

    parser.add_argument("--dataset_name", type=str, help="dataset name.")

    parser.add_argument("--exp_id", type=str, default=None, help="Exp id.")

    parser.add_argument("--cudaid", type=str, default=None,
                        help="cuda id: '0,1,2,3'")
    parser.add_argument("--seed", type=int, default=None, help="Seed.")
    parser.add_argument("--mode", type=str, default=None,
                        help="Mode: 'Training', 'Eval'.")
    parser.add_argument("--verbose", type=str2bool, default=None,
                        help="Verbosity (bool).")

    parser.add_argument("--train_p", type=float, default=None,
                        help="Percentage of video data to be used from "
                             "trainset.")
    parser.add_argument("--valid_p", type=float, default=None,
                        help="Percentage of video data to be used from "
                             "validset.")
    parser.add_argument("--test_p", type=float, default=None,
                        help="Percentage of video data to be used from "
                             "testset.")


    parser.add_argument("--amp", type=str2bool, default=None,
                        help="Whether to use automatic mixed precision for "
                             "training.")

    parser.add_argument("--opt__name_optimizer", type=str, default=None,
                        help="Name of the optimizer 'sgd', 'adam'.")
    parser.add_argument("--opt__lr", type=float, default=None,
                        help="Learning rate (optimizer)")
    parser.add_argument("--opt__momentum", type=float, default=None,
                        help="Momentum (optimizer)")
    parser.add_argument("--opt__dampening", type=float, default=None,
                        help="Dampening for Momentum (optimizer)")
    parser.add_argument("--opt__nesterov", type=str2bool, default=None,
                        help="Nesterov or not for Momentum (optimizer)")
    parser.add_argument("--opt__weight_decay", type=float, default=None,
                        help="Weight decay (optimizer)")
    parser.add_argument("--opt__beta1", type=float, default=None,
                        help="Beta1 for adam (optimizer)")
    parser.add_argument("--opt__beta2", type=float, default=None,
                        help="Beta2 for adam (optimizer)")
    parser.add_argument("--opt__eps_adam", type=float, default=None,
                        help="eps for adam (optimizer)")
    parser.add_argument("--opt__amsgrad", type=str2bool, default=None,
                        help="amsgrad for adam (optimizer)")
    parser.add_argument("--opt__lr_scheduler", type=str2bool, default=None,
                        help="Whether to use or not a lr scheduler")
    parser.add_argument("--opt__name_lr_scheduler", type=str, default=None,
                        help="Name of the lr scheduler.")
    parser.add_argument("--opt__gamma", type=float, default=None,
                        help="Gamma of the lr scheduler. (mystep)")
    parser.add_argument("--opt__last_epoch", type=int, default=None,
                        help="Index last epoch to stop adjust LR(mystep)")
    parser.add_argument("--opt__min_lr", type=float, default=None,
                        help="Minimum allowed value for lr.")
    parser.add_argument("--opt__t_max", type=float, default=None,
                        help="T_max, maximum epochs to restart. (cosine)")
    parser.add_argument("--opt__mode", type=str, default=None,
                        help="'min', 'max'. how to reduce lr w.r.t the "
                             "monitored quantity: when it stops decreasing ("
                             "min) or increasing (max) "
                             "(reduceonplateau, MyWarmupScheduler).")
    parser.add_argument("--opt__factor", type=float, default=None,
                        help="Factor, factor by which learning rate is "
                             "reduced. (reduceonplateau, MyWarmupScheduler)")
    parser.add_argument("--opt__patience", type=int, default=None,
                        help="Patience, number of epoch with no improvement. ("
                             "reduceonplateau, MyWarmupScheduler)")
    parser.add_argument("--opt__step_size", type=int, default=None,
                        help="Step size for lr scheduler.")
    parser.add_argument('--opt__gradual_release', default=None, type=int,
                        help='Whether to gradually release some layers?. '
                             'Default: 1.')
    parser.add_argument('--opt__release_count', default=None, type=int,
                        help='How many layer groups to release?. Default: 3.')
    parser.add_argument('--opt__milestone', default=None, type=str,
                        help='The specific epochs to do something. '
                             'Default: "0". use "+" to separate milestones. '
                             'eg: "10+20+30".')
    parser.add_argument('--opt__load_best_at_each_epoch', default=None,
                        type=str2bool,
                        help='Whether to load the best models state at the end '
                             'of each epoch?. Default: True.')

    # From JMT =================================================================
    # 1.4. Load checkpoint or not?
    parser.add_argument('--resume', default=None, type=str2bool,
                        help='Resume from checkpoint? boolean.')
    # 1.6. What modality to use?
    # supported: video, vggish, bert, EXPR_continuous_label.
    # EXPR_continuous_label: is mandatory.
    # to use multiple modalities at once, separate them by '+'.
    # "video+vggish+bert+EXPR_continuous_label"
    parser.add_argument('--modality',
                        type=str,
                        default='video+vggish+VA_continuous_label',
                        help="Used modalities.")
    # Calculate mean and std for each modality?
    parser.add_argument('--calc_mean_std', default=None, type=str2bool,
                        help='Calculate the mean and std and save to a '
                             'pickle file')
    # --emotion: NOT USED.
    parser.add_argument('--emotion', default="valence", type=str,
                        help='The emotion dimension to focus when updating'
                             ' gradient: arousal, valence, both')

    # 2. Training settings.
    parser.add_argument('--num_heads', default=None, type=int,
                        help="Number of heads for LFAN model. Default: 2.")
    parser.add_argument('--modal_dim', default=None, type=int,
                        help="Modality dim for LFAN model. Default: 32.")
    parser.add_argument('--tcn_kernel_size', default=None, type=int,
                        help='The size of the 1D kernel for temporal '
                             'convolutional networks for LFAN model. '
                             'Default: 5.')

    # 2.1. Overall settings
    parser.add_argument('--model_name', type=str, default=None,
                        help='LFAN, CAN, JMT.')
    parser.add_argument('--num_folds', default=None, type=int,
                        help="Total number of folds.")
    parser.add_argument('--fold_to_run', default=None, type=int,
                        help='Which fold to run.')
    parser.add_argument('--use_pretrained_w', default=None, type=str2bool,
                        help='Use or not pretrained weights to initialize the '
                             'model.')
    parser.add_argument('--pretrained_subpath', default=None, type=str,
                        help='Subpath points to the folder containing the '
                             'weights of the model s weights to use.')


    # 2.2. Epochs and data
    parser.add_argument('--num_epochs', default=None, type=int,
                        help='The total of epochs to run during training. '
                             'Default: 100.')
    parser.add_argument('--min_num_epochs', default=None, type=int,
                        help='The minimum epoch to run at least. Default: 5.')
    parser.add_argument('--early_stopping', default=None, type=int,
                        help='If no improvement, the number of epoch to'
                             ' run before halting the training. Default: 50.')
    parser.add_argument('--window_length', default=None, type=int,
                        help='The length in point number to windowing the data.'
                             'Default: 300.')
    parser.add_argument('--hop_length', default=None, type=int,
                        help='The step size or stride to move the window. '
                             'Default: 200.')
    parser.add_argument('--train_batch_size', default=None, type=int,
                        help="Train batch size.")
    parser.add_argument('--eval_batch_size', default=None, type=int,
                        help="Evaluation batch size.")
    parser.add_argument('--num_workers', default=None, type=int,
                        help="Number of workers for dataloader.")

    # 2.1. Scheduler and Parameter Control: to include with opt__XXXXX

    # parser.add_argument('-scheduler', default='plateau', type=str,
    #                     help='plateau, cosine')  # delete. use exist
    # parser.add_argument('-learning_rate', default=1e-5, type=float,
    #                     help='The initial learning rate.')  # delete.
    # parser.add_argument('-min_learning_rate', default=1.e-8, type=float,
    #                     help='The minimum learning rate.')  # delete.

    # 2.2. Groundtruth settings
    parser.add_argument('--time_delay', default=None, type=float,
                        help='For time_delay=n, it means the n-th label points '
                             'will be taken as the 1st, and the following ones '
                             'will be shifted accordingly.'
                             'The rear point will be duplicated to meet the '
                             'original length.'
                             'This is used to compensate the human labeling '
                             'delay. Default: 0.')
    parser.add_argument('--metrics', default="rmse", type=str,
                        help='The evaluation metrics.')
    parser.add_argument('--save_plot', type=str2bool, default=None,
                        help='Whether to plot the session-wise output/target'
                             ' or not?')

    parser.add_argument('--use_other_class', type=str2bool, default=None,
                        help='Use or not the class "Other" for the case of '
                             'the dataset "C_EXPR_DB".')

    parser.add_argument('--train_supervision_type', type=str, default=None,
                        help='What type training supervision.')


    # End JMT ==================================================================

    input_parser = parser.parse_args()
    attributes = input_parser.__dict__.keys()

    for k in attributes:
        val_k = getattr(input_parser, k)
        if k in args.keys():
            if val_k is not None:
                args[k] = val_k

        else:
            raise ValueError(f"Key {k} was not found in args. ... [NOT OK]")

    os.environ['MYSEED'] = str(args["seed"])
    max_seed = (2 ** 32) - 1
    msg = f"seed must be: 0 <= {int(args['seed'])} <= {max_seed}"
    assert 0 <= int(args['seed']) <= max_seed, msg

    args['outd'] = outfd(Dict2Obj(copy.deepcopy(args)))

    cmdr = os.path.isfile(join(args['outd'], 'passed.txt'))
    if cmdr:
        warnings.warn('EXP {} has already been done. EXITING.'.format(
            args['outd']))
        sys.exit(0)

    torch.cuda.set_device(0)
    args['cudaid'] = 0

    # sanity check
    # we only fully support classification.
    assert args['task'] == constants.CLASSFICATION, args['task']
    assert 0. < args['train_p'] <= 100., args['train_p']
    assert isinstance(args['train_p'], float), type(args['train_p'])

    assert args['train_supervision_type'] in constants.TR_SUP, args[
        'train_supervision_type']

    modalities = args['modality'].split('+')
    assert constants.EXPR in modalities

    for mdl in modalities:
        assert mdl in [constants.VIDEO, constants.VGGISH, constants.BERT,
                       constants.EXPR], mdl

    if args['use_pretrained_w']:
        # currently, we support only LFAN.
        assert args['model_name'] == constants.LFAN, args['model_name']

        px = join(args['load_path'], args['pretrained_subpath'])
        assert os.path.isdir(px), px
        m_w = join(px, 'model.pt')
        m_conf = join(px, 'config.yml')
        assert os.path.isfile(m_w), m_w
        assert os.path.isfile(m_conf), m_conf
        with open(m_conf, 'r') as fx:
            _conf = yaml.safe_load(fx)
        assert _conf['model_name'] == constants.LFAN, _conf['model_name']
        assert _conf['modality'] == args['modality'], f"{_conf['modality']} |" \
                                                      f" {args['modality']}"
    # --

    reproducibility.set_to_deterministic(seed=int(args["seed"]), verbose=True)

    return args


def outfd(args):
    tag = [('id', args.exp_id)]

    tag = [(el[0], str(el[1])) for el in tag]
    tag = '-'.join(['_'.join(el) for el in tag])

    parent_lv = join("exps", args.dataset_name, f"fold-{args.fold_to_run}")
    subpath = join(parent_lv, tag)
    outd = join(root_dir, subpath)

    outd = expanduser(outd)
    os.makedirs(outd, exist_ok=True)

    return outd


def wrap_sys_argv_cmd(cmd: str, pre):
    splits = cmd.split(' ')
    el = splits[1:]
    pairs = ['{} {}'.format(i, j) for i, j in zip(el[::2], el[1::2])]
    pro = splits[0]
    sep = ' \\\n' + (len(pre) + len(pro) + 2) * ' '
    out = sep.join(pairs)
    return "{} {} {}".format(pre, pro, out)


def get_tag_device(args: dict) -> str:
    tag = ''

    if torch.cuda.is_available():
        txt = subprocess.run(
            ['nvidia-smi', '--list-gpus'],
            stdout=subprocess.PIPE).stdout.decode('utf-8').split('\n')
        try:
            cudaids = args['cudaid'].split(',')
            tag = 'CUDA devices: \n'
            for cid in cudaids:
                tag += 'ID: {} - {} \n'.format(cid, txt[int(cid)])
        except IndexError:
            tag = 'CUDA devices: lost.'

    return tag


def parse_input() -> Tuple[dict, str, dict]:

    # Mandatory
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, help="mode: TRAINING, EVALUATION")
    input_args, _ = parser.parse_known_args()

    mode = input_args.mode
    assert mode in constants.MODES, mode

    eval_config = {
        'eval_set': '',
        'fd_exp': ''
    }

    if mode == constants.TRAINING:

        _parser = argparse.ArgumentParser()

        _parser.add_argument("--dataset_name", type=str, help="dataset name.")
        _input_args, _ = _parser.parse_known_args()
        ds = _input_args.dataset_name

        args: dict = get_config(ds)

        args['t0'] = dt.datetime.now()

        args: dict = get_args(args)

        log_backends = [
            ArbJSONStreamBackend(Verbosity.VERBOSE,
                                 join(args['outd'], "log.json")),
            ArbTextStreamBackend(Verbosity.VERBOSE,
                                 join(args['outd'], "log.txt")),
        ]

        if args['verbose']:
            log_backends.append(ArbStdOutBackend(Verbosity.VERBOSE))

        DLLogger.init_arb(backends=log_backends, master_pid=os.getpid())

        DLLogger.log(fmsg("Start time: {}".format(args['t0'])))

        DLLogger.log(fmsg(f"Fold: {args['fold_to_run']}"))

        outd = args['outd']

        with open(join(outd, "config.yml"), 'w') as fyaml:
            _args = copy.deepcopy(args)
            # todo: do the necessary changes.
            yaml.dump(_args, fyaml)

        str_cmd = wrap_sys_argv_cmd(" ".join(sys.argv), "time python")
        with open(join(outd, "cmd.sh"), 'w') as frun:
            frun.write("#!/usr/bin/env bash \n")
            frun.write(str_cmd)

    elif mode == constants.EVALUATION:
        parser.add_argument("--target_ds_name", type=str,
                            help="The dataset name on which we evaluate.")
        parser.add_argument("--eval_set", type=str,
                            help="Evaluation set: test, valid, train")
        parser.add_argument("--case_best_model", type=str,
                            help="Tag for the best model to be selected.")
        parser.add_argument("--fd_exp", type=str,
                            help="Absolute path to the exp folder")
        input_args, _ = parser.parse_known_args()
        eval_set = input_args.eval_set
        fd_exp = input_args.fd_exp
        target_ds_name = input_args.target_ds_name
        case_best_model = input_args.case_best_model

        assert eval_set in [constants.TRAINSET, constants.VALIDSET,
                            constants.TESTSET], eval_set
        assert os.path.isdir(fd_exp), fd_exp

        store_results_pkl = join(fd_exp, f'{eval_set}-reevaluation.pkl')
        if os.path.isfile(store_results_pkl):
            print(f"This evaluation has already been done. Exiting."
                  f"Fd_exp: {fd_exp}."
                  f"Eval_set: {eval_set}")
            sys.exit(0)

        args_path = join(fd_exp, 'config.yml')  # todo: delete
        assert os.path.isfile(args_path), args_path
        with open(args_path, 'r') as fx:
            args: dict = yaml.safe_load(fx)

        # for now: we support training on MELD and test on C-EXPR-DB-CHALLENGE
        assert args['dataset_name'] == constants.MELD, args['dataset_name']
        assert target_ds_name == constants.C_EXPR_DB_CHALLENGE, target_ds_name
        assert case_best_model in [constants.FRM_VOTE,
                                   constants.FRM_AVG_PROBS,
                                   constants.FRM_AVG_LOGITS], case_best_model

        args['mode'] = constants.EVALUATION
        # switch datasets
        args['dataset_name'] = target_ds_name
        args['folds_dir'] = join(root_dir, 'folds', target_ds_name)
        args["dataset_path"] = join(get_root_wsol_dataset(), target_ds_name)
        args['test_p'] = 100.
        args['train_p'] = 100.
        args['valid_p'] = 100.
        args['fold_to_run'] = 0
        args['num_folds'] = 1
        args['num_workers'] = 0

        # ======================================================================

        log_backends = [
            ArbJSONStreamBackend(Verbosity.VERBOSE,
                                 join(fd_exp, f"log-eval-{eval_set}.json")),
            ArbTextStreamBackend(Verbosity.VERBOSE,
                                 join(fd_exp, f"log-eval-{eval_set}.txt")),
        ]

        if args['verbose']:
            log_backends.append(ArbStdOutBackend(Verbosity.VERBOSE))

        DLLogger.init_arb(backends=log_backends, master_pid=os.getpid())

        DLLogger.log(fmsg("Start time: {}".format(args['t0'])))

        DLLogger.log(fmsg(f"Fold: {args['dataset_name']} -  EVAL: {eval_set}"))

        eval_config = {
            'eval_set': eval_set,
            'fd_exp': fd_exp,
            'target_ds_name': target_ds_name,
            'case_best_model': case_best_model
        }

    else:
        raise NotImplementedError(f"Mode: {mode}.")
    DLLogger.flush()
    return args, mode, eval_config
