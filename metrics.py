import copy
import sys
from os.path import dirname, abspath, join, basename, expanduser, normpath
from typing import Tuple
import random
from collections import Counter
from pprint import pprint
from pprint import pformat


import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

import constants

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
from tools import print_confusion_mtx
from tools import print_vector


__all__ = ['compute_f1_score',
           'compute_class_acc',
           'compute_confusion_matrix',
           'format_trg_pred_frames',
           'format_trg_pred_video',
           'PerfTracker'
           ]


def softmax(x: np.ndarray) -> np.ndarray:
    assert x.ndim == 2, x.ndim

    _exp = np.exp(x)

    return _exp / np.sum(_exp, axis=1).reshape((-1, 1))


def format_trg_pred_frames(data: dict,
                           ignore_class: int) -> Tuple[list, list]:
    """
    dict: key: video id. assumes no windowing. :
    {
     'labels': numpy.ndarray,
     'logits': numpy.ndarray
    }
    """
    limited = False
    if isinstance(ignore_class, int):
        assert ignore_class == 7, ignore_class  # C_EXPR_DB, 'Other' class.
        # last class.
        limited = True

    preds = []
    trgs = []
    for _id in data:
        _labels = data[_id]['labels'].tolist()
        _logits = data[_id]['logits']
        assert _logits.ndim == 2, _logits.ndim
        if limited:
            _logits = _logits[:, :-1]  # drop last class.

        _preds: list = np.argmax(_logits, axis=1).flatten().tolist()

        assert len(_preds) == len(_labels), f"{len(_preds)} | {len(_labels)}"

        for i, l in enumerate(_labels):
            if limited and l == ignore_class:
                continue

            trgs.append(l)
            preds.append(_preds[i])

    return preds, trgs


def format_trg_pred_video(data: dict,
                          ignore_class: int) -> Tuple[list, list]:
    """
    Assumes there is one label per video.
    dict: key: video id. assumes no windowing. :
    {
     'labels': numpy.ndarray,
     'logits': numpy.ndarray
    }
    """
    limited = False
    if isinstance(ignore_class, int):
        assert ignore_class == 7, ignore_class  # C_EXPR_DB, 'Other' class.
        # last class.
        limited = True

    preds = []
    trgs = []
    for _id in data:
        _labels = data[_id]['g_labels']
        # _labels = _labels.max() * (_labels * 0. + 1.)  # for AH dataset: we
        # # have the real per_frame labels. the max is the real global video.
        _unique: list = np.unique(_labels).tolist()
        assert len(_unique) == 1, len(_unique)
        _label = _unique[0]

        if limited and _label == ignore_class:
            continue

        _labels = _labels.tolist()
        _logits: np.ndarray = data[_id]['logits']
        assert _logits.ndim == 2, _logits.ndim
        if limited:
            _logits = _logits[:, :-1]  # drop last class.

        # decide based on majority voting
        _preds: list = np.argmax(_logits, axis=1).flatten().tolist()
        _x = Counter(_preds)
        _m_vote_pred = _x.most_common(1)[0][0]

        # decide based on avg logits
        _avg_logits = _logits.mean(axis=0, keepdims=True)
        _avg_logits_pred = np.argmax(_avg_logits, axis=1).tolist()[0]

        # decide based on avg probs
        _probs = softmax(_logits)
        _avg_probs = _probs.mean(axis=0, keepdims=True)
        _avg_probs_pred = np.argmax(_avg_probs, axis=1).tolist()[0]

        trgs.append(_label)
        preds.append(
            {
                constants.FRM_VOTE: _m_vote_pred,
                constants.FRM_AVG_LOGITS: _avg_logits_pred,
                constants.FRM_AVG_PROBS: _avg_probs_pred
            }
        )

    return preds, trgs


def compute_f1_score(trgs: list,
                     preds: list,
                     f1_type: str) -> Tuple[np.ndarray, float]:
    """
    Compute F1 score.
    """
    assert f1_type in [constants.W_F1, constants.MACRO_F1], f1_type
    if f1_type == constants.MACRO_F1:

        f1_s = f1_score(trgs, preds, average=None)
        macro_f1 = np.mean(f1_s).item()
        f1 = macro_f1

    elif f1_type == constants.W_F1:
        # per-class f1.
        f1_s = f1_score(trgs, preds, average=None)
        f1 = f1_score(trgs, preds, average='weighted')

    else:
        raise NotImplementedError(f1_type)

    return f1_s, f1


def compute_class_acc(trgs: list, preds: list) -> float:
    """
    Compute classification accuracy.
    """

    _trgs = np.array(trgs, dtype=np.float32)
    _preds = np.array(preds, dtype=np.float32)
    classification_acc = (((_preds == _trgs) * 1.).mean() * 100.).item()

    return classification_acc


def compute_confusion_matrix(trgs: list, preds: list):
    _trgs = np.array(trgs, dtype=np.float32)
    _preds = np.array(preds, dtype=np.float32)

    conf_mtx = confusion_matrix(y_true=_trgs,
                                y_pred=_preds,
                                sample_weight=None,
                                normalize='true'
                                )

    return conf_mtx


class PerfTracker(object):
    def __init__(self,
                 master_ignore_class=None,
                 master_metric=constants.MACRO_F1,
                 master_level=constants.FRAME_LEVEL,
                 master_video_pred=constants.FRM_VOTE
                 ):
        super(PerfTracker, self).__init__()

        self.first = True
        self.holder_list = []

        self.master_ignore_class = master_ignore_class
        self.master_metric = master_metric
        self.master_level = master_level
        self.master_video_pred = master_video_pred
        self.best_value = None
        self.best_value_idx = 0

        self.cnt = 0
        self.is_last_best = False
        self.current_status_str = 'None'
        self.best_status_str = 'None'

    def is_master(self, ignore_class, metric, level, video_pred) -> bool:
        cnd = (ignore_class == self.master_ignore_class)
        cnd &= (metric == self.master_metric)
        cnd &= (level == self.master_level)

        if level == constants.VIDEO_LEVEL:
            cnd &= (video_pred == self.master_video_pred)

        return cnd

    def init_tracker(self, data: dict):
        _data = copy.deepcopy(data)
        self.holder_list = [data]
        self.cnt = 0
        self.is_last_best = True

        for ignore_class in _data:
            for metric in _data[ignore_class]:
                for level in _data[ignore_class][metric]:
                    if level == constants.FRAME_LEVEL:
                        for k, value in _data[
                            ignore_class][metric][level].items():

                            _data[ignore_class][metric][
                                level][k] = [value]

                            if k == 'master':
                                if self.is_master(ignore_class, metric,
                                                  level, None):
                                    self.best_value = value
                                    self.best_value_idx = 0

                                    msg = f"MASTER: {ignore_class}, " \
                                          f"{metric}, {level}: {value:.6f}"

                                    self.current_status_str = msg
                                    self.best_status_str = msg

                    if level == constants.VIDEO_LEVEL:
                        for video_pred in _data[
                            ignore_class][metric][level]:
                            for k, value in _data[
                                ignore_class][metric][level][
                                video_pred].items():

                                _data[ignore_class][metric][
                                    level][video_pred][k] = [value]

                                if k == 'master':
                                    if self.is_master(ignore_class, metric,
                                                      level, video_pred):
                                        self.best_value = value
                                        self.best_value_idx = 0

                                        msg = f"MASTER: {ignore_class}, " \
                                              f"{metric}, {level}, " \
                                              f"{video_pred}: {value:.6f}"

                                        self.current_status_str = msg
                                        self.best_status_str = msg

    def report(self, data: dict, int_to_cl: dict) -> str:
        _data = data
        msg = ''

        for ignore_class in _data:
            for metric in _data[ignore_class]:
                for level in _data[ignore_class][metric]:
                    if level == constants.FRAME_LEVEL:
                        for k, value in _data[
                            ignore_class][metric][level].items():

                            c_msg = ''

                            if k == 'master':
                                if metric in [constants.CL_ACC,
                                              constants.MACRO_F1,
                                              constants.W_F1]:
                                    c_msg = f"{ignore_class}, " \
                                            f"{metric}, {level}: {value:.8f}"

                                    if metric == constants.CL_ACC:
                                        c_msg += '%'

                                elif metric == constants.CFUSE_MARIX:
                                    mtrx = print_confusion_mtx(value, int_to_cl)
                                    c_msg = f"{ignore_class}, " \
                                            f"{metric}, {level}:\n {mtrx}"
                                else:
                                    raise NotImplementedError(metric)

                                if self.is_master(ignore_class, metric,
                                                  level, None):
                                    c_msg = f"Master: {c_msg}"

                            elif k == 'per_cl':
                                if metric in [constants.MACRO_F1]:
                                    vec = print_vector(value, int_to_cl)
                                    c_msg = f"{ignore_class}, " \
                                            f"{metric}, {level}:\n {vec}"

                            else:
                                raise NotImplementedError(k)

                            msg = f"{msg}\n{c_msg}\n"

                    if level == constants.VIDEO_LEVEL:
                        for video_pred in _data[
                            ignore_class][metric][level]:
                            for k, value in _data[
                                ignore_class][metric][level][
                                video_pred].items():

                                c_msg = ''

                                if k == 'master':
                                    if metric in [constants.CL_ACC,
                                                  constants.MACRO_F1,
                                                  constants.W_F1]:
                                        c_msg = f"{ignore_class}, " \
                                                f"{metric}, {level}, " \
                                                f"{video_pred}: " \
                                                f"{value:.8f}"

                                        if metric == constants.CL_ACC:
                                            c_msg += '%'

                                    elif metric == constants.CFUSE_MARIX:
                                        mtrx = print_confusion_mtx(value,
                                                                   int_to_cl)
                                        c_msg = f"{ignore_class}, " \
                                                f"{metric}, {level}, " \
                                                f"{video_pred}" \
                                                f":\n {mtrx}"
                                    else:
                                        raise NotImplementedError(metric)

                                    if self.is_master(ignore_class, metric,
                                                      level, video_pred):
                                        c_msg = f"Master: {c_msg}"

                                elif k == 'per_cl':
                                    if metric in [constants.MACRO_F1]:
                                        vec = print_vector(value, int_to_cl)
                                        c_msg = f"{ignore_class}, " \
                                                f"{metric}, {level}," \
                                                f" {video_pred}" \
                                                f":\n {vec}"

                                else:
                                    raise NotImplementedError(k)

                                msg = f"{msg}\n{c_msg}\n"

        return msg

    def append(self, data: dict):

        if self.first:
            self.first = False

            self.init_tracker(data)
            return 0

        _data = copy.deepcopy(data)

        self.cnt += 1
        self.holder_list.append(data)
        is_best = False

        for ignore_class in _data:
            for metric in _data[ignore_class]:
                for level in _data[ignore_class][metric]:
                    if level == constants.FRAME_LEVEL:
                        for k, value in _data[
                            ignore_class][metric][level].items():

                            _data[ignore_class][metric][
                                level][k] = [value]

                            if k == 'master':
                                if self.is_master(ignore_class, metric,
                                                  level, None):

                                    c_msg = f"Current MASTER: " \
                                            f"{ignore_class}, " \
                                            f"{metric}, {level}:" \
                                            f" {value:.6f} " \
                                            f"(EP. {self.cnt - 1})"

                                    if value >= self.best_value:

                                        self.best_value = value
                                        self.best_value_idx = self.cnt

                                        is_best = True

                                    b_msg = f"BEST MASTER: {ignore_class}, " \
                                            f"{metric}, {level}: " \
                                            f"{self.best_value:.6f} " \
                                            f"(EP. {self.best_value_idx - 1})"

                                    self.current_status_str = c_msg
                                    self.best_status_str = b_msg

                    if level == constants.VIDEO_LEVEL:
                        for video_pred in _data[
                            ignore_class][metric][level]:
                            for k, value in _data[
                                ignore_class][metric][level][
                                video_pred].items():

                                _data[ignore_class][metric][
                                    level][video_pred][k] = [value]

                                if k == 'master':
                                    if self.is_master(ignore_class, metric,
                                                      level, video_pred):

                                        c_msg = f"Current MASTER:" \
                                                f" {ignore_class}, " \
                                                f"{metric}, {level}, " \
                                                f"{video_pred}: " \
                                                f"{value:.6f} " \
                                                f"(EP. {self.cnt - 1})"

                                        if value >= self.best_value:
                                            self.best_value = value
                                            self.best_value_idx = self.cnt
                                            is_best = True

                                        b_msg = f"BEST MASTER:" \
                                                f" {ignore_class}, " \
                                                f"{metric}, {level}, " \
                                                f"{video_pred}: " \
                                                f"{self.best_value:.6f} " \
                                                f"(EP. " \
                                                f"{self.best_value_idx - 1})"

                                        self.current_status_str = c_msg
                                        self.best_status_str = b_msg

        self.is_last_best = is_best


if __name__ == '__main__':
    seed = 0
    np.random.seed(seed)
    random.seed(seed)
    n = 1000

    data = dict()
    ignore_class = None

    assert ignore_class in [None, 7], ignore_class
    ncls = 8

    for i in range(100):
        data[i] = {
            'labels': np.random.randint(
                low=0, high=ncls, size=(1,)) * np.ones(n),
            'logits': np.random.rand(n, ncls)
        }

    # formatting frame level
    preds, trgs = format_trg_pred_frames(data, ignore_class=ignore_class)
    f1, avg_f1 = compute_f1_score(trgs, preds, constants.W_F1)
    acc = compute_class_acc(trgs, preds)
    cnf_mtx = compute_confusion_matrix(trgs, preds)
    print('Frame decision: -----')
    print(f"Ncls: {ncls}. Ignore_class: {ignore_class}")
    print(f"per-class F1 scores: {f1}")
    print(f"AVG F1 score: {avg_f1}")
    print(f"CL acc: {acc}")
    print(f"COnfusion matrix: {cnf_mtx}")
    print(10 * '-')

    # formatting video level
    preds, trgs = format_trg_pred_video(data, ignore_class=ignore_class)

    for k in preds[0]:
        print(f'Video decision {k}: -----')
        _preds_k = [item[k] for item in preds]
        f1, avg_f1 = compute_f1_score(trgs, _preds_k, constants.W_F1)
        acc = compute_class_acc(trgs, _preds_k)
        cnf_mtx = compute_confusion_matrix(trgs, _preds_k)

        print(f"Ncls: {ncls}. Ignore_class: {ignore_class}")
        print(f"per-class F1 scores: {f1}")
        print(f"AVG F1 score: {avg_f1}")
        print(f"CL acc: {acc}")
        print(f"COnfusion matrix: {cnf_mtx}")
        print(10 * '-')

