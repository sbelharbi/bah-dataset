import copy
import os
from os.path import dirname, abspath, join, basename, expanduser, normpath
import sys
import random


import numpy as np
import torch
from torch.utils.data import DataLoader

import constants

root_dir = dirname(dirname((abspath(__file__))))
sys.path.append(root_dir)

from base.utils import load_pickle, save_to_pickle

import dllogger as DLLogger
from tools import fmsg


class GenericImageExperiment(object):
    def __init__(self, args):
        # Basic experiment settings.
        self.args = copy.deepcopy(args)
        self.exp_id = args.exp_id
        self.dataset_path = args.dataset_path
        self.load_path = args.load_path

        self.save_path = args.save_path

        self.outd = args.outd
        self.seed = args.seed
        self.resume = args.resume
        self.calc_mean_std = args.calc_mean_std

        self.device = self.init_device()

    def load_config(self):
        raise NotImplementedError

    def init_random_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def init_device(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        return device

    def init_model(self, **kwargs):
        raise NotImplementedError

    def init_dataloader(self, **kwargs):
        raise NotImplementedError

    def experiment(self):
        raise NotImplementedError


class GenericExperiment(GenericImageExperiment):
    def __init__(self, args):

        # Basic experiment settings.
        super().__init__(args)

        self.model_name = args.model_name
        self.num_folds = args.num_folds
        self.fold_to_run = args.fold_to_run
        self.folds_dir = args.folds_dir
        self.dataset_name = args.dataset_name

        # optimizer
        self.learning_rate = args.opt__lr
        self.weight_decay = args.opt__weight_decay
        self.name_optimizer = args.opt__name_optimizer
        self.momentum = args.opt__momentum
        self.dampening = args.opt__dampening
        self.nesterov = args.opt__nesterov
        self.beta1 = args.opt__beta1
        self.beta2 = args.opt__beta2
        self.eps_adam = args.opt__eps_adam
        self.amsgrad = args.opt__amsgrad

        # lr scheduler
        self.use_lr_scheduler: bool = args.opt__lr_scheduler
        self.name_lr_scheduler = args.opt__name_lr_scheduler
        self.min_learning_rate = args.opt__min_lr
        self.factor = args.opt__factor
        self.patience = args.opt__patience
        self.gamma = args.opt__gamma
        self.step_size = args.opt__step_size
        self.opt__last_epoch = args.opt__last_epoch
        self.t_max = args.opt__t_max

        self.mode = args.opt__mode
        self.gradual_release = args.opt__gradual_release
        self.release_count = args.opt__release_count
        self.milestone = args.opt__milestone
        self.opt__load_best_at_each_epoch = args.opt__load_best_at_each_epoch

        self.modality = args.modality.split('+')
        # in this work, https://arxiv.org/abs/2203.13031, video is expected
        # to be the modality used to combine its features with the final
        # features. it is expected to be the first modality. see the model
        # LFAN. but, the code works whatever the order.
        # assert self.modality[0] == constants.VIDEO, self.modality[0]

        assert len(self.modality) > 0, len(self.modality)
        for mdl in self.modality:
            assert mdl in constants.MODALITIES, mdl

        self.calc_mean_std = args.calc_mean_std

        self.window_length = args.window_length
        self.hop_length = args.hop_length
        self.train_batch_size = args.train_batch_size
        self.eval_batch_size = args.eval_batch_size

        self.time_delay = None
        self.dataset_info = None
        self.mean_std_dict = None
        self.data_arranger = None

        self.feature_dimension = None
        self.multiplier = None
        self.continuous_label_dim = None

        self.config = None

        if hasattr(args, 'emotion'):
            self.emotion = args.emotion

    def prepare(self):

        self.config = self.get_config()
        self.feature_dimension = self.get_feature_dimension(self.config)
        self.multiplier = self.get_multiplier(self.config)
        self.time_delay = self.get_time_delay(self.config)

        self.continuous_label_dim = self.get_selected_continuous_label_dim()

        self.dataset_info = load_pickle(
            join(self.dataset_path, "dataset_info.pkl"))

        self.init_randomness()

        self.data_arranger = self.init_data_arranger()
        if self.calc_mean_std:
            self.calc_mean_std_fn()

        self.mean_std_dict = load_pickle(self.get_mean_std_dict_path())

    def get_config(self):
        raise NotImplementedError

    def get_selected_continuous_label_dim(self):
        dim = [0 if self.emotion == "arousal" else 1]
        return dim

    def run(self):
        raise NotImplementedError

    def init_dataloader(self, fold):
        self.init_randomness()

        data_list = self.data_arranger.generate_partitioned_trial_list(
            window_length=self.window_length,
            hop_length=self.hop_length,
            windowing=True,
            window_eval=False  # for eval, we load the entire video. if the
            # video is longer than the model can handle such as in LFAN,
            # we window the video right before the forward. then, we gather
            # the results to be the same as the video length.
        )

        datasets, dataloaders = {}, {}
        for split, data in data_list.items():
            if len(data):

                # Cross-platform deterministic shuffling for the training set.
                if split == constants.TRAINSET:
                    for _ in range(100):
                        random.shuffle(data_list[split])

                datasets[split] = self.init_dataset(data_list[split],
                                                    self.continuous_label_dim,
                                                    split,
                                                    fold=None
                                                    )
                if split == constants.TRAINSET:
                    bsize = self.train_batch_size
                    shuffle = True
                elif split in [constants.TESTSET, constants.VALIDSET]:
                    bsize = self.eval_batch_size
                    shuffle = False
                else:
                    raise NotImplementedError(split)

                dataloaders[split] = DataLoader(
                    dataset=datasets[split],
                    batch_size=bsize,
                    shuffle=shuffle,
                    num_workers=self.args.num_workers,
                    pin_memory=True
                )

        return dataloaders

    def init_dataset(self, data, continuous_label_dim, mode, fold):
        raise NotImplementedError

    def init_model(self, **kwargs):
        raise NotImplementedError

    @staticmethod
    def get_time_delay(config):
        time_delay = config['time_delay']
        return time_delay

    @staticmethod
    def get_feature_dimension(config):
        feature_dimension = config['feature_dimension']
        return feature_dimension

    @staticmethod
    def get_multiplier(config):
        multiplier = config['multiplier']
        return multiplier

    def get_modality(self):
        raise NotImplementedError

    def init_data_arranger(self):
        raise NotImplementedError

    def get_mean_std_dict_path(self):
        j = self.fold_to_run
        path = os.path.join(self.dataset_path, f"mean_std_info_fold-{j}.pkl")
        return path

    def calc_mean_std_fn(self):
        # compute the mean and std of the CURRENT split. the old version
        # computes the stats of all folds at once (separately).

        path = self.get_mean_std_dict_path()

        if os.path.isfile(path):
            DLLogger.log(f"File exists. No need to recompute again: {path}")
            return 0

        DLLogger.log(f"Computing mean/std of "
                     f"(DS: {self.dataset_name}, fold: {self.fold_to_run}): "
                     f"{path}")

        data_list = self.data_arranger.generate_partitioned_trial_list(
            window_length=self.window_length,
            hop_length=self.hop_length,
            windowing=False,
            window_eval=False
        )
        mean_std = self.data_arranger.calculate_mean_std(data_list)

        save_to_pickle(path, mean_std, replace=True)

    def old_calc_mean_std_fn(self):
        path = self.get_mean_std_dict_path()

        mean_std_dict = {}
        for fold in range(self.num_folds):

            data_list = self.data_arranger.generate_partitioned_trial_list(
                window_length=self.window_length,
                hop_length=self.hop_length,
                fold=fold,
                windowing=False
            )
            mean_std_dict[fold] = self.data_arranger.calculate_mean_std(data_list)

        save_to_pickle(path, mean_std_dict, replace=True)

    def init_randomness(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def init_device(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        return device

    def init_config(self):
        raise NotImplementedError