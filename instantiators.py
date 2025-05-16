import warnings
import sys
import os
from os.path import dirname, abspath, join, basename
from copy import deepcopy

import torch
import torch.nn as nn
import yaml
from torch.optim import SGD
from torch.optim import Adam
import torch.optim.lr_scheduler as lr_scheduler



from abaw5_pre_processing.dlib.utils.tools import Dict2Obj
from abaw5_pre_processing.dlib.utils.shared import format_dict_2_str
import dllogger as DLLogger
import constants

from base.scheduler import GradualWarmupScheduler
from base.scheduler import MyWarmupScheduler
from base.scheduler import GradualWarmupScheduler
from base.scheduler import MyWarmupScheduler
from base.scheduler import MyStepLR
from base.scheduler import MyCosineLR


__all__ = ['get_optimizer_scheduler']


def standardize_otpmizers_params(optm_dict):
    """
    Standardize the keys of a dict for the optimizer.
    all the keys starts with 'optn[?]__key' where we keep only the key and
    delete the initial.
    the dict should not have a key that has a dict as value. we do not deal
    with this case. an error will be raise.

    :param optm_dict: dict with specific keys.
    :return: a copy of optm_dict with standardized keys.
    """

    assert isinstance(optm_dict, dict), type(optm_dict)
    new_optm_dict = deepcopy(optm_dict)
    loldkeys = list(new_optm_dict.keys())

    for k in loldkeys:
        if k.startswith('opt'):
            msg = "'{}' is a dict. it must not be the case." \
                  "otherwise, we have to do a recursive thing....".format(k)
            assert not isinstance(new_optm_dict[k], dict), msg

            new_k = k.split('__')[1]
            new_optm_dict[new_k] = new_optm_dict.pop(k)

    return new_optm_dict


def get_optimizer_scheduler(args, params, epoch: int, best: float):
    """
    Instantiate an optimizer and its lr scheduler.
    """
    hparams = deepcopy(args)
    hparams = standardize_otpmizers_params(hparams)
    hparams = Dict2Obj(hparams)

    op_col = {}
    _params = [{'params': params, 'lr': hparams.lr}]

    op_name = hparams.name_optimizer
    assert op_name in constants.OPTIMIZERS, f"{op_name}, {constants.OPTIMIZERS}"

    if op_name == constants.SGD:
        optimizer = SGD(params=params,
                        momentum=hparams.momentum,
                        dampening=hparams.dampening,
                        weight_decay=hparams.weight_decay,
                        nesterov=hparams.nesterov)
        op_col['optim_name'] = hparams.name_optimizer
        op_col['lr'] = hparams.lr
        op_col['momentum'] = hparams.momentum
        op_col['dampening'] = hparams.dampening
        op_col['weight_decay'] = hparams.weight_decay
        op_col['nesterov'] = hparams.nesterov

    elif op_name == constants.ADAM:
        optimizer = Adam(params=params,
                         betas=(hparams.beta1, hparams.beta2),
                         eps=hparams.eps_adam,
                         weight_decay=hparams.weight_decay,
                         amsgrad=hparams.amsgrad)
        op_col['optim_name'] = hparams.name_optimizer
        op_col['lr'] = hparams.lr
        op_col['beta1'] = hparams.beta1
        op_col['beta2'] = hparams.beta2
        op_col['weight_decay'] = hparams.weight_decay
        op_col['amsgrad'] = hparams.amsgrad
    else:
        raise ValueError("Unsupported optimizer `{}` .... "
                         "".format(args.optimizer["name"]))

    if hparams.lr_scheduler:
        if hparams.name_lr_scheduler == constants.MYWARMUP:
            lrate_scheduler = MyWarmupScheduler(
                optimizer=optimizer,
                lr=hparams.lr,
                min_lr=hparams.min_lr,
                best=best,
                mode=hparams.mode,
                patience=hparams.patience,
                factor=hparams.factor,
                num_warmup_epoch=args.min_num_epochs,
                init_epoch=epoch
            )
            op_col['name_lr_scheduler'] = hparams.name_lr_scheduler

        elif hparams.name_lr_scheduler == constants.STEP:
            lrate_scheduler = lr_scheduler.StepLR(optimizer,
                                                  step_size=hparams.step_size,
                                                  gamma=hparams.gamma,
                                                  last_epoch=hparams.last_epoch)
            op_col['name_lr_scheduler'] = hparams.name_lr_scheduler
            op_col['step_size'] = hparams.step_size
            op_col['gamma'] = hparams.gamma
            op_col['last_epoch'] = hparams.last_epoch

        elif hparams.name_lr_scheduler == constants.COSINE:
            lrate_scheduler = lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=hparams.t_max,
                eta_min=hparams.min_lr,
                last_epoch=hparams.last_epoch)
            op_col['name_lr_scheduler'] = hparams.name_lr_scheduler
            op_col['T_max'] = hparams.T_max
            op_col['eta_min'] = hparams.eta_min
            op_col['last_epoch'] = hparams.last_epoch

        elif hparams.name_lr_scheduler == constants.MYSTEP:
            lrate_scheduler = MyStepLR(optimizer,
                                       step_size=hparams.step_size,
                                       gamma=hparams.gamma,
                                       last_epoch=hparams.last_epoch,
                                       min_lr=hparams.min_lr
                                       )
            op_col['name_lr_scheduler'] = hparams.name_lr_scheduler
            op_col['step_size'] = hparams.step_size
            op_col['gamma'] = hparams.gamma
            op_col['min_lr'] = hparams.min_lr
            op_col['last_epoch'] = hparams.last_epoch

        elif hparams.name_lr_scheduler == constants.MYCOSINE:
            lrate_scheduler = MyCosineLR(optimizer,
                                         coef=hparams.coef,
                                         max_epochs=hparams.max_epochs,
                                         min_lr=hparams.min_lr,
                                         last_epoch=hparams.last_epoch
                                         )
            op_col['name_lr_scheduler'] = hparams.name_lr_scheduler
            op_col['coef'] = hparams.coef
            op_col['max_epochs'] = hparams.max_epochs
            op_col['min_lr'] = hparams.min_lr
            op_col['last_epoch'] = hparams.last_epoch

        elif hparams.name_lr_scheduler == constants.MULTISTEP:
            lrate_scheduler = lr_scheduler.MultiStepLR(
                optimizer,
                milestones=hparams.milestones,
                gamma=hparams.gamma,
                last_epoch=hparams.last_epoch)
            op_col['name_lr_scheduler'] = hparams.name_lr_scheduler
            op_col['milestones'] = hparams.milestones
            op_col['gamma'] = hparams.gamma
            op_col['last_epoch'] = hparams.last_epoch

        else:
            raise ValueError("Unsupported learning rate scheduler `{}` .... "
                             "[NOT OK]".format(
                                hparams.name_lr_scheduler))
    else:
        lrate_scheduler = None

    DLLogger.log("Optimizer:\n{}".format(format_dict_2_str(op_col)))

    return optimizer, lrate_scheduler
