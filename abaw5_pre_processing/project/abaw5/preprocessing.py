import os
import glob
import sys
import argparse
import os
from os.path import dirname, abspath, join, basename, expanduser, normpath
import more_itertools as mit

import pandas as pd
import numpy as np
import cv2
import tqdm

root_dir = dirname(dirname(dirname((abspath(__file__)))))
sys.path.append(root_dir)

from abaw5_pre_processing.base.preprocessing import GenericVideoPreprocessing

from abaw5_pre_processing.base.utils import ensure_dir
from abaw5_pre_processing.base.utils import save_to_pickle
from abaw5_pre_processing.base.utils import load_pickle
# from abaw5_pre_processing.project.abaw5 import _constants

import constants


class PreprocessingABAW5(GenericVideoPreprocessing):
    def __init__(self, split: str, part: int, nparts: int,
                 config, cnn_path=None):
        super().__init__(part, config, cnn_path=cnn_path)

        ds = config['dataset_name']
        assert ds in [constants.MELD, constants.C_EXPR_DB,
                      constants.C_EXPR_DB_CHALLENGE, constants.AH_DB,
                      constants.BAH_DB], ds
        assert split in [constants.TRAINSET, constants.VALIDSET,
                         constants.TESTSET], split

        self.split = split
        self.nparts = nparts
        self.part = part
        self.ds = ds

        self.trial_list = {}
        self.task = config['task']
        # self.task_list = ["VA"]
        # self.multitask = config['multi_task']
        if ds in [constants.MELD, constants.C_EXPR_DB,
                  constants.C_EXPR_DB_CHALLENGE, constants.AH_DB,
                  constants.BAH_DB]:
            self.partition_list = [split]
            # self.partition_list = [constants.TRAINSET, constants.VALIDSET,
            #                        constants.TESTSET]

        else:
            raise NotImplementedError(ds)
            # self.partition_list = ["train", "validate", "extra", "test"]

        self.generate_task_trial_list()

    def read_fold_content(self, p: str) -> list:
        ds = self.config['dataset_name']
        if ds in [constants.MELD, constants.C_EXPR_DB,
                  constants.C_EXPR_DB_CHALLENGE, constants.AH_DB,
                  constants.BAH_DB]:
            with open(p, 'r') as f:
                content = f.readlines()
                lines = []
                for l in content:
                    lines.append(l.strip())

            return lines

        else:
            raise NotImplementedError(ds)

    def generate_task_trial_list(self):

        # Load all the trials as per the task and partition
        # Overall, we have 3 tasks by 3 partitions, plus an extra universal
        # list, totally 10 lists.
        self.trial_list = {self.task: {partition: [] for partition in
                                       self.partition_list}}
        universal_list = self.get_all_file(
            join(self.config['root_directory'],
                 self.config['cropped_aligned_folder']), filename_only=True)

        # get folds.

        for partition in self.partition_list:
            if partition in [constants.TRAINSET, constants.VALIDSET,
                             constants.TESTSET]:

                # self.trial_list[self.task][partition] = [
                #     file.split(".txt")[0] for file in self.get_all_file(
                #         join(self.config['root_directory'],
                #              self.config['annotation_folder'],
                #              self.task, partition),
                #         filename_only=True)]

                # todo: for c_expr_db: copy manually the txt splits to
                #  root_directory. done in a rush.
                fold_file = join(self.config['root_directory'],
                                 f"{partition}.txt")
                assert os.path.isfile(fold_file), fold_file

                lines = self.read_fold_content(fold_file)
                self.trial_list[self.task][partition] = lines

            elif partition == "extra":
                self.trial_list[self.task][partition] = list(
                    set(universal_list).difference(
                        self.trial_list[self.task]['train']
                    ).difference(self.trial_list[self.task]['validate']))

        self.dataset_info['trial_list'] = self.trial_list

    def generate_iterator(self):
        path = join(self.config['root_directory'],
                    self.config['cropped_aligned_folder'])
        iterator = sorted([f.path for f in os.scandir(path) if f.is_dir()])
        return iterator

    def split_trials(self, full_trial_info):
        # divide all the data (list of videos of a specif split) into nparts.
        # then select part number 'self.part'.
        part = self.part
        nparts = self.nparts

        split_data = [list(c) for c in mit.divide(nparts, full_trial_info)]


        # if self.part == -1:
        #     return full_trial_info
        #
        # if self.part >= 8:
        #     raise ValueError("Part should be -1 or 0 - 7")  # Return an empty
        #     # list if the index is out of range
        #
        #
        # num_sublists = 8
        # total_length = len(full_trial_info)
        # sublist_length = total_length // num_sublists
        # remainder = total_length % num_sublists
        #
        # start_index = self.part * sublist_length + min(self.part, remainder)
        # end_index = start_index + sublist_length + (
        #     1 if self.part < remainder else 0)
        #
        # return full_trial_info[start_index:end_index]

        return split_data[part]

    def generate_per_trial_info_dict(self):

        split = self.split
        part = self.part
        nparts = self.nparts
        ds = self.ds

        per_trial_info_path = join(self.config['output_root_directory'],
                                   f"processing_records_"
                                   f"{ds}_{split}_{nparts}_{part}.pkl")

        if os.path.isfile(per_trial_info_path) and False:  # always
            per_trial_info = load_pickle(per_trial_info_path)

        else:
            ds = self.config['dataset_name']

            assert ds in [constants.MELD, constants.C_EXPR_DB,
                          constants.C_EXPR_DB_CHALLENGE, constants.AH_DB,
                          constants.BAH_DB], ds

            per_trial_info = []
            # iterator = self.generate_iterator()
            # get iterator over all videos folders.
            iterator = []
            entries = []
            per_partition_entries = {}
            id_to_partition = {}
            # merge train, valid, test.: not really since we use only one
            # partition (split) at a time.

            for partition in self.trial_list[self.task]:
                entry_part = self.trial_list[self.task][partition]
                entries = entries + entry_part

                per_partition_entries[partition] = {}
                # take only video paths.
                for e in entry_part:
                    _id = e.split(',')[0]
                    _label = e.split(',')[1]
                    _label: int = int(_label)
                    _p = f"{_id},{_label},"

                    _text = e.replace(_p, '')
                    assert _id not in per_partition_entries[partition], _id

                    per_partition_entries[partition][_id] = {
                        'label': _label,
                        'text': _text
                    }

                    iterator.append(_id)  # video id. e.g.: test/dia279_utt10
                    assert _id not in id_to_partition, _id
                    id_to_partition[_id] = partition


            subject_id = 0
            # looping through videos:
            print('Generating dict info of videos...')
            for idx, file in tqdm.tqdm(enumerate(iterator), total=len(
                    iterator), ncols=80):
                # file: video name without extension: video id.
                _cropped_file = join(self.config['cropped_aligned_folder'],
                                     file)
                this_trial = {}

                this_trial['cropped_aligned_path'] = _cropped_file
                # old:
                # this_trial['trial'] = file.split(os.sep)[-1]
                this_trial['trial'] = file
                this_trial['trial_no'] = 1
                this_trial['subject_no'] = subject_id
                video_path = self.get_video_path(file)
                this_trial['video_path'] = this_trial['audio_path'] = video_path

                video = cv2.VideoCapture(this_trial['video_path'])
                fps = video.get(cv2.CAP_PROP_FPS)
                this_trial['fps'] = fps
                this_trial['target_fps'] = fps

                video_partition = id_to_partition[file]
                video_label: int = per_partition_entries[video_partition][
                    file]['label']
                video_transcript = per_partition_entries[video_partition][
                    file]['text']

                this_trial['video_transcript'] = video_transcript
                this_trial['video_class'] = video_label

                if ds == constants.MELD:
                    this_trial, wild_trial = self.get_meld_partition(
                        this_trial, video_partition, video_label)

                elif ds == constants.AH_DB:
                    this_trial, wild_trial = self.get_ah_db_partition(
                        this_trial, video_partition, video_label)

                elif ds == constants.BAH_DB:
                    this_trial, wild_trial = self.get_bah_db_partition(
                        this_trial, video_partition, video_label)

                elif ds == constants.C_EXPR_DB:
                    this_trial, wild_trial = self.get_c_expr_db_partition(
                        this_trial, video_partition, video_label)

                elif ds == constants.C_EXPR_DB_CHALLENGE:
                    this_trial, wild_trial = self.get_c_expr_db_challenge_partition(
                        this_trial, video_partition, video_label)
                else:
                    raise NotImplementedError(ds)
                    # this_trial, wild_trial = self.get_partition(this_trial)

                assert wild_trial == 0, wild_trial

                if wild_trial:
                    continue

                # If it is from the test set, then the trial length is taken as
                # the video length, otherwise the label length (without the rows
                # containing -1 or -5).
                annotated_index = np.arange(
                    int(video.get(cv2.CAP_PROP_FRAME_COUNT)))

                n_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

                load_continuous_label_kwargs = {}
                load_continuous_label_kwargs['cols'] = self.config[
                    self.task + '_label_cols']
                load_continuous_label_kwargs['task'] = self.task

                if this_trial['has_' + self.task + '_label']:

                    assert ds in [constants.MELD, constants.C_EXPR_DB,
                                  constants.C_EXPR_DB_CHALLENGE,
                                  constants.AH_DB, constants.BAH_DB], ds

                    # old version for affwild.
                    # continuous_label = self.load_continuous_label(
                    #     this_trial[self.task + '_label_path'],
                    #     **load_continuous_label_kwargs)
                    #
                    # _, annotated_index = self.process_continuous_label(
                    #     continuous_label, **load_continuous_label_kwargs)

                    # classification: all frames are annotated.
                    # we dont update annotated_index. as it is, it points to
                    # all frames.
                    pass

                this_trial['length'] = len(annotated_index)

                get_annotated_index_kwargs = {}
                get_annotated_index_kwargs['source_frequency'] = this_trial[
                    'fps']

                get_annotated_index_kwargs['feature'] = "video"
                this_trial['video_annotated_index'] = self.get_annotated_index(
                    annotated_index, **get_annotated_index_kwargs)

                # cnn = self.config['cnn_extractor_config']['model_name']
                # this_trial['cnn_' + cnn + '_annotated_index'] = this_trial[
                # 'video_annotated_index']

                get_annotated_index_kwargs['feature'] = "mfcc"
                this_trial['mfcc_annotated_index'] = self.get_annotated_index(
                    annotated_index, **get_annotated_index_kwargs)

                get_annotated_index_kwargs['feature'] = "egemaps"
                this_trial['egemaps_annotated_index'] = \
                    self.get_annotated_index(annotated_index,
                                             **get_annotated_index_kwargs)

                get_annotated_index_kwargs['feature'] = "logmel"
                this_trial['logmel_annotated_index'] = self.get_annotated_index(
                    annotated_index, **get_annotated_index_kwargs)

                get_annotated_index_kwargs['feature'] = "vggish"
                this_trial['vggish_annotated_index'] = self.get_annotated_index(
                    annotated_index, **get_annotated_index_kwargs)

                this_trial['speech_annotated_index'] = annotated_index
                per_trial_info.append(this_trial)

                subject_id += 1

        ensure_dir(per_trial_info_path)
        save_to_pickle(per_trial_info_path, per_trial_info, replace=True)
        # todo: if slow, and need to parallelize, we need to adjust the splits.
        selected_info = self.split_trials(per_trial_info)
        self.per_trial_info = selected_info

    def generate_dataset_info(self):

        self.dataset_info['pseudo_partition'] = []

        for idx, record in enumerate(self.per_trial_info):

            if 'partition' in record[self.task]:
                # print(record['trial'])  # video id.
                partition = record[self.task]['partition']
                self.dataset_info['trial'].append(record['processing_record'][
                                                      'trial'])
                self.dataset_info['trial_no'].append(record['trial_no'])
                self.dataset_info['subject_no'].append(record['subject_no'])
                self.dataset_info['length'].append(record['length'])
                self.dataset_info['partition'].append(partition)

                # For verifying semi-supervision
                self.dataset_info['pseudo_partition'].append(partition)

                # if partition == "validate":
                #     self.dataset_info['pseudo_partition'].append(partition)
                # elif partition == "train":
                #     n, p = 1, .7  # number of toss, probability of each toss
                #     s = np.random.binomial(n, p, 1)
                #     if s == 1:
                #         self.dataset_info['pseudo_partition'].append('train')
                #     else:
                #         self.dataset_info['pseudo_partition'].append('extra')
                # else:
                #     self.dataset_info['pseudo_partition'].append('unused')

        self.dataset_info['data_folder'] = self.config['npy_folder']

        split = self.split
        part = self.part
        nparts = self.nparts
        ds = self.ds

        # path = join(self.config['output_root_directory'],
        #             f'dataset_info_{self.part}.pkl')
        path = join(self.config['output_root_directory'],
                    f'dataset_info_'
                    f'{ds}_{split}_{nparts}_{part}.pkl')

        save_to_pickle(path, self.dataset_info, replace=True)

    def compact_facial_image(self, path, annotated_index, extension="jpg"):
        from PIL import Image
        trial_length = len(annotated_index)

        frame_matrix = np.zeros((
            trial_length, self.config['video_size'], self.config['video_size'],
            3), dtype=np.uint8)

        for j, frame in enumerate(range(trial_length)):
            current_frame_path = os.path.join(path, str(j + 1).zfill(5) +
                                              ".jpg")
            if os.path.isfile(current_frame_path):
                current_frame = Image.open(current_frame_path)
                frame_matrix[j] = current_frame.resize(
                    (self.config['video_size'], self.config['video_size']))
        return frame_matrix

    def extract_continuous_label_fn(self, idx, npy_folder):

            condition = self.per_trial_info[idx]['has_' + self.task + '_label']

            if condition:

                load_continuous_label_kwargs = {}

                if self.per_trial_info[idx]['has_' + self.task + '_label']:
                    load_continuous_label_kwargs['cols'] = self.config[
                        self.task + '_label_cols']
                    load_continuous_label_kwargs['task'] = self.task

                    video_label = self.per_trial_info[idx][
                        self.task + '_label_path']
                    n_frames = self.per_trial_info[idx]['length']
                    continuous_label = np.ones(
                        shape=(n_frames,),  dtype=np.int64) * int(video_label)

                    # old version for affwild.

                    # continuous_label = self.load_continuous_label(
                    #     self.per_trial_info[idx][self.task + '_label_path'],
                    #     **load_continuous_label_kwargs)
                    #
                    # continuous_label, annotated_index = \
                    #     self.process_continuous_label(
                    #         continuous_label, **load_continuous_label_kwargs)

                    if self.config['save_npy']:
                        filename = join(npy_folder,
                                        self.task + "_continuous_label.npy")
                        if not os.path.isfile(filename):
                            ensure_dir(filename)
                            np.save(filename, continuous_label)

    def load_continuous_label(self, path, **kwargs):

        cols = kwargs['cols']

        continuous_label = pd.read_csv(path, sep=",",
                                       skipinitialspace=True, usecols=cols,
                                       index_col=False).values.squeeze()

        return continuous_label

    def get_annotated_index(self, annotated_index, **kwargs):

        feature = kwargs['feature']
        source_frequency = kwargs['source_frequency']  # fps of video.
        target_frequency = self.config['frequency'][feature]

        if feature in ["video", "vggish", "mfcc", "egemaps", "logmel"]:
            target_frequency = source_frequency

        sampled_index = np.asarray(
            np.round(target_frequency / source_frequency * annotated_index),
            dtype=np.int64)

        return sampled_index

    def process_continuous_label(self, continuous_label, **kwargs):
        task = kwargs['task']
        not_labeled = 0

        if continuous_label.ndim == 1:
            continuous_label = continuous_label[:, np.newaxis]

        if task == "AU":
            not_labeled = -12
        elif task == "EXPR":
            not_labeled = -1
        elif task == "VA":
            not_labeled = -10
        else:
            ValueError("Unknown task!")

        row_wise_sum = np.sum(continuous_label, axis=1)
        annotated_index = np.asarray(
            np.where(row_wise_sum != not_labeled)[0], dtype=np.int64)

        return continuous_label[annotated_index], annotated_index

    @staticmethod
    def read_txt(txt_file):
        lines = pd.read_csv(txt_file, header=None)[0].tolist()
        return lines

    def get_meld_partition(self, this_trial, partition, label):
        """
        Same as 'self.get_partition' but for MELD dataset.
        """
        ds = self.config['dataset_name']
        assert ds == constants.MELD, ds

        this_trial[self.task] = {}
        this_trial[self.task]['partition'] = partition
        this_trial[self.task + '_label_path']: int = label  # changed from
        # path to label.

        this_trial['has_' + self.task + '_label'] = 1

        wild_trial = 0

        return this_trial, wild_trial

    def get_ah_db_partition(self, this_trial, partition, label):
        """
        Same as 'self.get_partition' but for AH_DB dataset.
        """
        ds = self.config['dataset_name']
        assert ds == constants.AH_DB, ds

        this_trial[self.task] = {}
        this_trial[self.task]['partition'] = partition
        this_trial[self.task + '_label_path']: int = label  # changed from
        # path to label.

        this_trial['has_' + self.task + '_label'] = 1

        wild_trial = 0

        return this_trial, wild_trial

    def get_bah_db_partition(self, this_trial, partition, label):
        """
        Same as 'self.get_partition' but for BAH_DB dataset.
        """
        ds = self.config['dataset_name']
        assert ds == constants.BAH_DB, ds

        this_trial[self.task] = {}
        this_trial[self.task]['partition'] = partition
        this_trial[self.task + '_label_path']: int = label  # changed from
        # path to label.

        this_trial['has_' + self.task + '_label'] = 1

        wild_trial = 0

        return this_trial, wild_trial

    def get_c_expr_db_partition(self, this_trial, partition, label):
        """
        Same as 'self.get_partition' but for C_EXPR_DB dataset.
        """
        ds = self.config['dataset_name']
        assert ds == constants.C_EXPR_DB, ds

        this_trial[self.task] = {}
        this_trial[self.task]['partition'] = partition
        this_trial[self.task + '_label_path']: int = label  # changed from
        # path to label.

        this_trial['has_' + self.task + '_label'] = 1

        wild_trial = 0

        return this_trial, wild_trial

    def get_c_expr_db_challenge_partition(self, this_trial, partition, label):
        """
        Same as 'self.get_partition' but for C_EXPR_DB_CHALLENGE dataset.
        """
        ds = self.config['dataset_name']
        assert ds == constants.C_EXPR_DB_CHALLENGE, ds

        this_trial[self.task] = {}
        this_trial[self.task]['partition'] = partition
        this_trial[self.task + '_label_path']: int = label  # changed from
        # path to label.

        this_trial['has_' + self.task + '_label'] = 1

        wild_trial = 0

        return this_trial, wild_trial

    def get_partition(self, this_trial):

        # Some trials may not be in any task and any partitions. We have to
        # exclude it.
        wild_trial = 1

        this_trial[self.task] = {}
        this_trial['has_' + self.task + '_label'] = 0

        trial_name = this_trial['trial']

        for partition in self.partition_list:
            if trial_name in self.trial_list[self.task][partition]:
                this_trial[self.task]['partition'] = partition
                this_trial[self.task + '_label_path'] = None
                label_path = join(self.config['root_directory'],
                                  self.config['annotation_folder'], self.task,
                                  partition, trial_name + ".txt")
                this_trial[self.task + '_label_path'] = label_path

                if os.path.isfile(label_path):
                    this_trial['has_' + self.task + '_label'] = 1

            wild_trial = 0

        return this_trial, wild_trial

    @staticmethod
    def get_output_filename(**kwargs):
        trial_name = kwargs['trial_name']
        return trial_name

    def get_video_path(self, video_name):

        ds = self.config['dataset_name']

        if ds == constants.MELD:
            dataset_root_path = self.config['dataset_root_path']
            video_path = join(dataset_root_path, 'original',
                              f"{video_name}.mp4")
            assert os.path.isfile(video_path), video_path

            return video_path

        elif ds == constants.AH_DB:
            dataset_root_path = self.config['dataset_root_path']
            video_path = join(dataset_root_path, 'videos',
                              f"{video_name}")
            assert os.path.isfile(video_path), video_path

            return video_path

        elif ds == constants.BAH_DB:
            dataset_root_path = self.config['dataset_root_path']
            video_path = join(dataset_root_path, f"{video_name}")
            assert os.path.isfile(video_path), video_path

            return video_path

        elif ds == constants.C_EXPR_DB:
            dataset_root_path = self.config['dataset_root_path']
            video_path = join(dataset_root_path, 'trimmed_videos',
                              f"{video_name}.mp4")
            assert os.path.isfile(video_path), video_path

            return video_path

        elif ds == constants.C_EXPR_DB_CHALLENGE:
            dataset_root_path = self.config['dataset_root_path']
            video_path = join(dataset_root_path,
                              f"{video_name}.mp4")
            assert os.path.isfile(video_path), video_path

            return video_path

        else:
            raise NotImplementedError(ds)

        # old: affwild.
        # else:
        #     if video_name.endswith("_right"):
        #         video_name = video_name[:-6]
        #     elif video_name.endswith("_left"):
        #         video_name = video_name[:-5]
        #
        #     video_name = video_name.replace("cropped_aligned", "raw_videos")
        #     video_name = [video_name + ".mp4" if os.path.isfile(
        #         video_name + ".mp4") else video_name + ".avi"][0]
        #     return video_name

    @staticmethod
    def get_all_file(path, filename_only=False):

        all_files = glob.glob(os.path.join(path, "*"))
        if filename_only:
            all_files = [file.split(os.sep)[-1] for file in all_files]
        return all_files


if __name__ == '__main__':
    import argparse
    import sys

    parser = argparse.ArgumentParser(description='Hello, world.')

    parser.add_argument('-python_package_path',
                        default='/home/zhangsu/phd4',
                        type=str,
                        help='The path to the entire repository.')
    args = parser.parse_args()
    sys.path.insert(0, args.python_package_path)

    # from project.abaw5.configs import config
    from abaw5_pre_processing.project.abaw5.configs import config
    part = -1
    pre = PreprocessingABAW5(part, config)
    pre.generate_per_trial_info_dict()
    pre.prepare_data()
