import os
from os.path import dirname, abspath, join, basename, expanduser, normpath
import sys

import constants

root_dir = dirname((abspath(__file__)))
sys.path.append(root_dir)


__all__ = ["get_config", "get_root_wsol_dataset"]


def get_root_wsol_dataset():
    baseurl = None
    if "HOST_XXX" in os.environ.keys():
        if os.environ['HOST_XXX'] == 'XXXXX':

            baseurl = f"~/datasets"

        else:
            raise NotImplementedError(os.environ['HOST_XXX'])

    else:
        raise NotImplementedError

    msg_unknown_host = "Sorry, it seems we are unable to recognize the " \
                       "host. You seem to be new to this code. " \
                       "We recommend you to add your baseurl on your own."
    if baseurl is None:
        raise ValueError(msg_unknown_host)

    return baseurl


def get_config(ds: str) -> dict:

    config = {
        "dataset_name": ds,  # name of the dataset
        "num_classes": constants.NUM_CLASSES[ds],  # number of
        # classes.
        "task": constants.DS_TASK[ds],  # task.
        "train_p": 100.,  # percentage of total number of trainset videos to be
        # used for train. can be useful for fast debug.
        "valid_p": 100.,  # percentage of total number of validset videos to be
        # used for validation. can be useful for fast debug.
        "test_p": 100.,  # percentage of total number of testset videos to be
        # used for test. can be useful for fast debug.

        "outd": '',  # absolute path of the experiment where to store
        # results. will be set automatically.
        "exp_id": '123456',  # unique id code of the experiment.

        "t0": "STARTING_TIME",  # start time
        "tend": "FINISHING_TIME",  # end time

        "seed": 0,  # seed for the experiment.
        "cudaid": "0",  # cuda id. it can be a list: "0,1,2,3".
        "verbose": True,  # if verbose, we print in stdout.
        "mode": constants.TRAINING,  # mode: training, eval.
        "resume": False,  # resume training or start fresh.
        "modality": "video+vggish+bert+EXPR_continuous_label",  # modalities
        # used; separated by '+'.
        "calc_mean_std": True,  # compute training stats.
        "emotion": "???",

        "model_name": constants.LFAN,  # name of the model
        "num_folds": 1,  # total number of folds.
        "fold_to_run": 0,  # which fold to run.
        "folds_dir": join(root_dir, 'folds', ds),  # absolute path to the folds
        # directory.
        "use_pretrained_w": False,  # if true, we use initialize the model by
        # pretrained weights.
        "pretrained_subpath": '',  # sub_path to the folder containing the
        # pretrained model to use. "load_path" is the absolute parent path.

        'amp': False,  # if true, use automatic mixed-precision for training

        "num_heads": 2,  # LFAN model
        "modal_dim": 32,  # LFAN model
        "tcn_kernel_size": 5,  # LFAN model

        "num_epochs": 100,  # max number of training epochs.
        "min_num_epochs": 5,  # minimum number of epochs.
        "early_stopping": 50,  # 50. max epochs to try when no improvement is
        # found.
        "window_length": 300,  # The length in point number to windowing the
        # data.
        "hop_length": 200,  # The step size or stride to move the window.

        "train_batch_size": 16,  # training batch size.
        "eval_batch_size": 1,  # evaluation batch size.
        "num_workers": 6,  # number of workers for dataloader.

        "opt__weight_decay": 0.0001,
        "opt__name_optimizer": constants.SGD,
        "opt__lr": 0.001,
        "opt__momentum": 0.9,
        "opt__dampening": 0.0,
        "opt__nesterov": True,
        "opt__beta1": 0.9,
        "opt__beta2": 0.999,
        "opt__eps_adam": 1e-8,
        "opt__amsgrad": False,

        # LR scheduler
        "opt__lr_scheduler": True,
        "opt__name_lr_scheduler": constants.MYSTEP,
        "opt__gamma": 0.1,
        "opt__step_size": 40,
        "opt__last_epoch": -1,
        "opt__min_lr": 1e-7,
        "opt__t_max": 100,
        "opt__mode": constants.MIN_MODE,
        "opt__factor": 0.5,
        "opt__patience": 10,
        "opt__gradual_release": 1,
        "opt__release_count": 3,
        "opt__milestone": "0",
        "opt__load_best_at_each_epoch": False,

        # Other
        "time_delay": 0,
        "metrics": "nrmse",
        "save_plot": False,
        "dataset_path": join(get_root_wsol_dataset(), ds),  # absolute path
        # where the concerned dataset is located.
        "load_path": join(root_dir, 'pretrained_models'),  # folder where the
        # pretrained weights
        "save_path": "",

        # other case:
        "use_other_class": False,  # can be true only for the dataset
        # 'C_EXPR_DB'. otherwise, it must be false. when true, the class
        # 'Other' is considered as well.
        # train supervision:
        "train_supervision_type": constants.TR_SUP_VIDEO_FR,
        "best_epoch": 0,  # best epoch for model selection.
        "visual_backbone_path": None,  # path to the visual backbone. when
        # specified with a valid path, it is used. when not specified,
        # we fall back to the original backbone res50_ir_0.887.

    }
    dataset_path = config['dataset_path']
    assert os.path.isdir(dataset_path), dataset_path

    folds_dir = config['folds_dir']
    assert os.path.isdir(folds_dir), folds_dir

    load_path = config['load_path']
    assert os.path.isdir(load_path), load_path

    return config
