import copy
import os
import sys
from os.path import dirname, abspath, join, basename, expanduser, normpath
from operator import itemgetter
from collections import OrderedDict
import yaml

import numpy as np
import tqdm
from torch.utils.data import Dataset
from torchvision.transforms import transforms

root_dir = dirname(dirname((abspath(__file__))))
sys.path.append(root_dir)

import constants
from base.transforms3D import *
from base.utils import load_npy
import dllogger as DLLogger
from tools import fmsg
from reproducibility import set_seed


class GenericDataArranger(object):
    def __init__(self, args, dataset_info: dict, dataset_path: str,
                 fold_to_run: int, folds_dir: str):

        self.args = args
        assert os.path.isdir(folds_dir), folds_dir
        self.fold_to_run = fold_to_run
        self.folds_dir = folds_dir

        self.dataset_info = dataset_info
        self.trial_list = self.generate_raw_trial_list(dataset_path)

        # not useful. will be replaced with self.data_per_split
        self.partition_range = self.partition_range_fn()
        self.fold_to_partition = self.assign_fold_to_partition()


        fold = self.args.fold_to_run
        folds_path = join(root_dir, self.args.folds_dir,
                          f"split-{fold}")
        path_class_id = join(folds_path, 'class_id.yaml')
        with open(path_class_id, 'r') as fcl:
            cl_int = yaml.safe_load(fcl)

        self.cl_to_int: dict = cl_int
        self.int_to_cl: dict = self.switch_key_val_dict(cl_int)

        self.data_per_split = self.create_splits()

    @staticmethod
    def switch_key_val_dict(d: dict) -> dict:
        out = dict()
        for k in d:
            assert d[k] not in out, 'more than 1 key with same value. wrong.'
            out[d[k]] = k

        return out

    def load_fold_txt(self, path_fold: str) -> dict:
        out = dict()
        with open(path_fold, 'r') as f:
            for line in f.readlines():
                line = line.strip('\n')
                v_id, cl_int = line.split(',')[0:2]
                txt = line.replace(f"{v_id},{cl_int},", '')
                assert v_id not in out, v_id
                cl_int = int(cl_int)
                out[v_id] = {'cl': cl_int, 'txt': txt}

        return out

    def create_splits(self):
        # create: train, valid, test splits data.
        j = self.fold_to_run
        data_per_split = dict()
        for split in self.dataset_info:
            path_fold = join(self.folds_dir, f"split-{j}", f"{split}.txt")
            cnd = (self.args.dataset_name == constants.C_EXPR_DB)
            cnd &= (not self.args.use_other_class)
            fold: dict = self.load_fold_txt(path_fold)

            _fold: dict = copy.deepcopy(fold)
            if cnd:  # remove class 'Other' from train if not needed.
                for itemx in fold:
                    _other_int = self.cl_to_int[constants.OTHER]
                    assert _other_int == 7, _other_int
                    if int(fold[itemx]['cl']) == _other_int:
                        _fold.pop(itemx)
                fold = _fold

            data_per_split[split] = []
            trial_label = []
            for trial in fold:
                for i, item in enumerate(self.trial_list):
                    _, _trial, _ = item
                    if trial == _trial:
                        data_per_split[split].append(self.trial_list[i])

                        trial_label.append([trial, fold[trial]['cl']])
                        break

            # trim train set if needed
            if split == constants.TRAINSET:
                p = self.args.train_p
            elif split == constants.VALIDSET:
                p = self.args.valid_p
            elif split == constants.TESTSET:
                p = self.args.test_p
            else:
                raise NotImplementedError(split)

            mm = len(data_per_split[split])
            if p < 100.:
                _nn = int(mm * p / 100.)
                _nn = max(1, _nn)

                # deal with unbalance. take p% per-class.
                new_data = self.keep_p_from_split(data_per_split[split],
                                                  trial_label, p / 100.)
                # shuffle
                if split == constants.TRAINSET:
                    for i in range(1000):
                        random.shuffle(new_data)

                set_seed(self.args.seed, verbose=False)
                data_per_split[split] = new_data
                nn = len(new_data)
                DLLogger.log(
                    fmsg(f"split: {split} goes from {mm} "
                         f"videos to {nn} (should be close to {_nn}) "
                         f"({p}%)."))
            else:
                DLLogger.log(
                    fmsg(f"split: {split} was maintained in full {mm} videos"
                         f"({p}%)."))

        return data_per_split

    def keep_p_from_split(self,
                          data: list,
                          data_with_label: list,
                          p: float) -> list:

        assert 0 < p <= 1., p
        assert isinstance(p, float), type(p)

        seed = self.args.seed
        set_seed(seed, verbose=False)

        assert len(data) == len(data_with_label), f"{len(data)} | " \
                                                  f"{len(data_with_label)}"

        cls = [item[1] for item in data_with_label]
        unique = np.unique(np.asarray(cls)).tolist()

        out_data = []
        for cl in unique:
            l = []
            l_cl = []
            for i, x in enumerate(cls):
                # random draw from Bernoulli with p
                if (x == cl) and (np.random.binomial(n=1, p=p) == 1):
                    l.append(data[i])

                # for the case where we never enter the above test we pick one.
                if x == cl:
                    l_cl.append(data[i])

            if l == []:
                j = np.random.randint(low=0, high=len(l_cl), size=(1,))[0]
                l = [l_cl[j]]

            assert l != [], f"{len(l)} | {cl}"
            out_data.extend(l)

        set_seed(seed, verbose=False)

        return out_data

    def generate_iterator(self):
        iterator = self.dataset_info['partition']  # list of str split name.
        return iterator

    def generate_partitioned_trial_list(self,
                                        window_length,
                                        hop_length,
                                        windowing=True,
                                        window_eval=True
                                        ):
        """
        Split the data into windows if requested. else, all data is taken.
        This is done per split (train, valid, test).

        :param window_length:
        :param hop_length:
        :param windowing: if true, the data of each split is windowed by a
        hop. else, all data is taken.
        :param window_eval: if true, the 'windowing' is applied over
        evaluation sets well (valid, test), else it is not.
        :return:
        """

        # old version.
        # train_validate_range = self.partition_range['train'
        #                        ] + self.partition_range['validate']
        # assert len(train_validate_range) == self.fold_to_partition['train'
        # ] + self.fold_to_partition['validate']
        #
        # partition_range = list(np.roll(train_validate_range, fold))
        # partition_range += self.partition_range['test'
        #                    ] + self.partition_range['extra']

        partitioned_trial = {}

        for split in self.data_per_split:
            data_split = self.data_per_split[split]
            partitioned_trial[split] = []

            for item in data_split:
                path, trial, length = item

                if windowing:
                    if split not in [constants.TESTSET, constants.VALIDSET]:
                        _window_length = window_length
                    else:
                        if window_eval:
                            _window_length = window_length
                        else:
                            _window_length = length
                else:
                    _window_length = length

                windowed_indices = self.windowing(np.arange(length),
                                                  window_length=_window_length,
                                                  hop_length=hop_length
                                                  )

                for index in windowed_indices:
                    partitioned_trial[split].append(
                        [path, trial, length, index])

        # old version
        # for partition, num_fold in self.fold_to_partition.items():
        #     partitioned_trial[partition] = []
        #
        #     for i in range(num_fold):
        #         index = partition_range.pop(0)
        #         trial_of_this_fold = list(itemgetter(*index)(self.trial_list))
        #
        #         if len(index) == 1:
        #             trial_of_this_fold = [trial_of_this_fold]
        #
        #         for path, trial, length in trial_of_this_fold:
        #             if not windowing:
        #                 window_length = length
        #
        #             windowed_indices = self.windowing(
        #                 np.arange(length),
        #                 window_length=window_length,
        #                 hop_length=hop_length)
        #
        #             for index in windowed_indices:
        #                 partitioned_trial[partition].append(
        #                     [path, trial, length, index])

        return partitioned_trial

    def calculate_mean_std(self, partitioned_trial: dict) -> dict:

        # we compute the mean and std of data used for training ONLY (train +
        # valid).
        # in the previous version, they compute these stats per split (train,
        # valid, test), separately.
        # the second change is that we compute the stats of each vector
        # component separately. The previous version merges all vector
        # components to compute the stats.

        feature_list = self.get_feature_list()

        mean_std_dict = {
            feature: {'mean': None, 'std': None} for feature in feature_list
        }
        # merge train and valid.
        data = partitioned_trial[constants.TRAINSET] + partitioned_trial[
            constants.VALIDSET]

        # mean.
        for feature in feature_list:
            assert feature in [constants.VGGISH, constants.BERT], feature
            lengths = 0
            sums = 0

            for path, _, _, _ in tqdm.tqdm(data, ncols=80, total=len(data)):
                samples = load_npy(path, feature)
                assert samples.ndim == 2, samples.ndim  # nframes, dim
                n, d = samples.shape

                lengths = lengths + n
                sums = sums + samples.sum(axis=0)

            mean_std_dict[feature]['mean'] = sums / (lengths + 1e-10)

        # STD
        for feature in feature_list:
            assert feature in [constants.VGGISH, constants.BERT], feature

            lengths = 0
            x_minus_mean_square = 0
            avg = mean_std_dict[feature]['mean']
            for path, _, _, _ in tqdm.tqdm(data, ncols=80, total=len(data)):
                samples = load_npy(path, feature)
                assert samples.ndim == 2, samples.ndim  # nframes, dim
                n, d = samples.shape
                diff = (samples - avg) ** 2
                diff = diff.sum(axis=0)
                lengths = lengths + n
                x_minus_mean_square = x_minus_mean_square + diff

            std = x_minus_mean_square / (lengths - 1)
            mean_std_dict[feature]['std'] = np.sqrt(std)

        return mean_std_dict

    def old_calculate_mean_std(self, partitioned_trial):
        feature_list = self.get_feature_list()

        mean_std_dict = {
            partition: {feature:
                            {'mean': None,
                             'std': None} for feature in feature_list
                        } for partition in partitioned_trial.keys()
        }

        # Calculate the mean
        for feature in feature_list:
            for partition, trial_of_a_partition in partitioned_trial.items():
                lengths = 0
                sums = 0
                for path, _, _, _ in trial_of_a_partition:
                    data = load_npy(path, feature)
                    data = data.flatten()
                    lengths += len(data)
                    sums += data.sum()
                mean_std_dict[partition][feature]['mean'] = sums / (lengths + 1e-10)

        # Then calculate the standard deviation.
        for feature in feature_list:
            for partition, trial_of_a_partition in partitioned_trial.items():
                lengths = 0
                x_minus_mean_square = 0
                mean = mean_std_dict[partition][feature]['mean']
                for path, _, _, _ in trial_of_a_partition:
                    data = load_npy(path, feature)
                    data = data.flatten()
                    lengths += len(data)
                    x_minus_mean_square += np.sum((data - mean) ** 2)
                x_minus_mean_square_divide_N_minus_1 = x_minus_mean_square / (lengths - 1)
                mean_std_dict[partition][feature]['std'] = np.sqrt(x_minus_mean_square_divide_N_minus_1)

        return mean_std_dict

    @staticmethod
    def partition_range_fn():
        raise NotImplementedError

    @staticmethod
    def assign_fold_to_partition():
        raise NotImplementedError

    @staticmethod
    def get_feature_list():
        feature_list = ['landmark', 'action_unit', 'mfcc', 'egemaps', 'vggish', 'bert']
        return feature_list

    def generate_raw_trial_list(self, dataset_path: str) -> list:


        # trial_dict = OrderedDict({'train': [],
        #                           'validate': [],
        #                           'extra': [],
        #                           'test': []}
        #                          )
        trial_dict = dict()
        for partition in self.dataset_info:
            trial_path = join(dataset_path, 'features',
                              self.dataset_info[partition]['data_folder']
                              )
            data_part = self.dataset_info[partition]
            trials = data_part['trial']

            trial_dict[partition] = []

            for idx in range(len(trials)):
                trial = trials[idx]
                path = join(trial_path, trial)
                length = data_part['length'][idx]
                # correct length for datasets with issues.
                if self.args.dataset_name in [constants.C_EXPR_DB_CHALLENGE,
                                              constants.C_EXPR_DB]:
                    video_p = join(path, 'video.npy')
                    assert os.path.isfile(video_p), video_p

                    vid = np.load(video_p)
                    length = vid.shape[0]

                trial_dict[partition].append([path, trial, length])

        # old version.

        # for idx, partition in enumerate(self.generate_iterator()):
        #
        #     if partition == "unused":
        #         continue
        #
        #     trial = self.dataset_info['trial'][idx]
        #     path = os.path.join(trial_path, trial)
        #     length = self.dataset_info['length'][idx]
        #
        #     trial_dict[partition].append([path, trial, length])

        # merge all data across all splits. we will re-divide them later
        # based on the current fold.

        trial_list = []
        for partition, trials in trial_dict.items():
            trial_list.extend(trials)

        return trial_list

    @staticmethod
    def windowing(x, window_length, hop_length):
        _length = len(x)

        if _length > window_length:
            steps = (_length - window_length) // hop_length + 1

            sampled_x = []
            for i in range(steps):
                start = i * hop_length
                end = start + window_length
                sampled_x.append(x[start:end])

            if sampled_x[-1][-1] < _length - 1:
                sampled_x.append(x[-window_length:])

        else:
            sampled_x = [x]

        return sampled_x


class GenericDataset(Dataset):
    def __init__(self,
                 data_list,
                 continuous_label_dim,
                 modality,
                 multiplier,
                 feature_dimension,
                 window_length,
                 mode,
                 mean_std=None,
                 time_delay=0,
                 feature_extraction=0
                 ):

        self.data_list = data_list
        self.continuous_label_dim = continuous_label_dim
        self.mean_std = mean_std
        self.mean_std_info = 0
        self.time_delay = time_delay
        self.modality = modality
        self.multiplier = multiplier
        self.feature_dimension = feature_dimension
        self.feature_extraction = feature_extraction
        self.window_length = window_length
        self.mode = mode
        self.transform_dict = {}
        self.get_3D_transforms()

    def get_index_given_emotion(self):
        raise NotImplementedError

    def get_3D_transforms(self, visual_backbone_path: str = None):

        normalize = GroupNormalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

        path = visual_backbone_path

        if path not in [None, '', 'None']:
            assert os.path.isdir(path), path
            # new models: apvit, resnet18,34,50,101,152.
            resize = 124
            crop_sz = 112

        else:
            # old original visual model
            resize = 48
            crop_sz = 40

        if constants.VIDEO in self.modality:
            if self.mode == constants.TRAINSET:
                self.transform_dict[constants.VIDEO] = transforms.Compose([
                    GroupNumpyToPILImage(use_inverse=False),
                    GroupScale(resize),  # todo: change if video encoder has
                    # changed.
                    GroupRandomCrop(resize, crop_sz),
                    GroupRandomHorizontalFlip(),
                    Stack(),
                    ToTorchFormatTensor(),
                    normalize
                ])
            else:
                self.transform_dict[constants.VIDEO] = transforms.Compose([
                    GroupNumpyToPILImage(use_inverse=False),
                    GroupScale(resize),
                    GroupCenterCrop(crop_sz),
                    Stack(),
                    ToTorchFormatTensor(),
                    normalize
                ])

        for feature in self.modality:
            if "continuous_label" not in feature and "video" not in feature:
                self.transform_dict[feature] = self.get_feature_transform(feature)

    def get_feature_transform(self, feature):
        if "logmel" in feature:
            transform = transforms.Compose([
                transforms.ToTensor()
            ])
        else:
            # to avoid this warning, we need to reshape and avoid [mean], [std]
            # torchvision/transforms/_functional_tensor.py:918:
            # UserWarning: Creating a tensor from a list of numpy.ndarrays
            # is extremely slow. Please consider converting the list to a
            # single numpy.ndarray with numpy.array() before converting to a
            # tensor. (Triggered internally at ../torch/csrc/utils/tensor_new.
            # cpp:275.)
            avg = self.mean_std[feature]['mean'].reshape(1, -1)
            std = self.mean_std[feature]['std'].reshape(1, -1)

            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=avg,
                                     std=std)
            ])


        return transform

    def __getitem__(self, index):

        path, trial, length, index = self.data_list[index]

        examples = {}

        there_is_continuous = False
        for feature in self.modality:
            examples[feature] = self.get_example(path, length, index, feature)
            # labels:
            # sometimes dataloader fails to convert to long.
            # we use float, then we convert later to long.
            if 'continuous_label' in feature:
                there_is_continuous = True
                examples[feature] = examples[feature].astype(np.float32)

        # breaks the code.
        # load global label: load the full frame label, get the video label,
        # then multiply it by the already extracted (window) label
        assert there_is_continuous
        filename = os.path.join(path, "EXPR_continuous_label.npy")
        _frame_labels = np.load(filename, mmap_mode='c')
        assert "EXPR_continuous_label" in examples, examples.keys()
        z = examples['EXPR_continuous_label']
        global_sup = z.max() * (z * 0. + 1.)
        examples['global_sup'] = global_sup

        if len(index) < self.window_length:
            index = np.arange(self.window_length)

        return examples, trial, length, index

    def __len__(self):
        return len(self.data_list)

    def get_example(self, path, length, index, feature):

        x = random.randint(0, self.multiplier[feature] - 1)
        random_index = index * self.multiplier[feature] + x

        # Probably, a trial may be shorter than the window, so the zero
        # padding is employed.
        if length < self.window_length:
            shape = (self.window_length,) + self.feature_dimension[feature]
            dtype = np.float32
            if feature == "video":
                dtype = np.int8
            example = np.zeros(shape=shape, dtype=dtype)

            example[index] = self.load_data(path, random_index, feature)
            # 0-padding labels issue.
            last_s = example[index[-1]]
            # duplicate the rest by the last element
            for i in range(index.size, self.window_length):
                example[i] = last_s

        else:
            example = self.load_data(path, random_index, feature)

        assert self.time_delay == 0, self.time_delay  # didnt revise this.
        if "continuous_label" in feature and self.time_delay != 0:
            example = np.concatenate(
                (example[self.time_delay:, :],
                 np.repeat(example[-1, :][np.newaxis],
                           repeats=self.time_delay, axis=0)), axis=0)

        if "continuous_label" not in feature:
            example = self.transform_dict[feature](np.asarray(example,
                                                              dtype=np.float32))
        return example

    def load_data(self, path, indices, feature):
        filename = os.path.join(path, feature + ".npy")

        # For the test set, labels of zeros are generated as dummies.
        data = np.zeros(((len(indices),) + self.feature_dimension[feature]),
                        dtype=np.float32)

        if os.path.isfile(filename):
            if self.feature_extraction:
                data = np.load(filename, mmap_mode='c')
            else:
                data = np.load(filename, mmap_mode='c')[indices]

            if "continuous_label" in feature:
                data = self.processing_label(data)

        return data

    def processing_label(self, label: np.ndarray):

        if self.args.task == constants.CLASSFICATION:
            assert label.ndim == 1, label.ndim
            return label[:, None]  # shape (n, 1)

        label = label[:, self.continuous_label_dim]
        if label.ndim == 1:
            label = label[:, None]
        return label



