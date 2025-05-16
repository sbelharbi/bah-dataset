import copy
import sys
from os.path import dirname, abspath, join, basename, expanduser, normpath
from typing import Tuple


import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import auc

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

import constants

# Metrics: macro F1 score
MACRO_F1 = 'MACRO_F1'  # Macro F1: unweighted average.
W_F1 = 'W_F1'  # F1 score: weighted.
CL_ACC = 'CL_ACC'  # classification acc.
CFUSE_MARIX = 'CONFUSION_MATRIX'  # confusion matrix

# A/H metrics (additional) ------
F1_POS = 'F1_POS'  # F1 positive class
F1_NEG = 'F1_NEG'  # F1 negative class

AP_POS = 'Average_precision_POS'  # AP for positive class.
# --------------------------------


__all__ = ['bah_perfs']


def softmax(x, h: float = 1.):
    e_x = np.exp(x * h)
    return e_x / np.sum(e_x, axis=1, keepdims=True)


def average_precision(probs_cl_1: np.ndarray, gt: np.ndarray) -> float:

    conf_step: float = 0.001
    confidences = np.arange(0, 1, conf_step).tolist()

    n_pos = gt.sum().item()
    assert n_pos > 0, f"{n_pos}"

    l_prec = []
    l_rec = []

    for th in confidences:
        preds = (probs_cl_1 >= th).astype(float)

        tp = np.sum(np.logical_and(preds == 1, gt == 1))
        fp = np.sum(np.logical_and(preds == 1, gt == 0))
        fn = np.sum(np.logical_and(preds == 0, gt == 1))

        assert (tp + fn) == n_pos, f"{tp} | {fn} | {n_pos}"

        prec = 0.

        if (tp + fp) > 0:
            prec = (tp / (tp + fp)).item()

        rec = (tp / (tp + fn)).item()

        l_prec.append(prec)
        l_rec.append(rec)

    # ap
    _rec = np.array(l_rec)
    _prec = np.array(l_prec)

    ap = float(auc(_rec, _prec).item())

    return ap


def compute_cnf_mtx(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    conf_mtx = confusion_matrix(y_true=target,
                                y_pred=pred,
                                sample_weight=None,
                                normalize='true'
                                )

    # row: true. columns: predictions.

    return conf_mtx



def bah_perfs(gt: np.ndarray,
              hard_preds: np.ndarray,
              logits: np.ndarray = None) -> dict:
    """
    Note: class 1 is presence of A/H. Class 0 is absence of A/H.

    Compute Ambivalence/hesitancy recognition performances:
    - classification accuracy
    - Confusion matrix
    - F1 score for each class
    - WF1 score
    - Average (macro) F1
    - Average precision of class 1.

    :param gt: np.ndarray: binary ground true of shape (n,).
    :param hard_preds: np.ndarray: binary prediction of shape (n,).
    :param logits: np.ndarray: prediction logits of shape (n, 2). if logits
    dont exist, set it to None.
    :return: dict with the following performance:
    """
    assert isinstance(gt, np.ndarray), type(gt)
    assert isinstance(hard_preds, np.ndarray), type(hard_preds)
    assert gt.shape == hard_preds.shape, f"{gt.shape} | {hard_preds.shape}"
    n = gt.shape[0]
    if logits is not None:
        assert isinstance(logits, np.ndarray), type(logits)
        assert logits.ndim == 2, logits.ndim  # nsamples, ncls

        assert logits.shape[0] == n, f"{logits.shape[0]} | {n}"

    gt = gt.astype(float)
    hard_preds = hard_preds.astype(float)

    # Classification accuracy
    cl_acc = float((gt == hard_preds).mean().item())
    # Confusion matrix
    conf_mtx = compute_cnf_mtx(pred=hard_preds, target=gt)

    # F1
    _trg = gt.tolist()
    _prd = hard_preds.tolist()

    f1_s = f1_score(_trg, _prd, average=None)
    assert f1_s.shape == (2,), f1_s.shape
    f1_cl_0 = float(f1_s[0].item())
    f1_cl_1 = float(f1_s[1].item())

    macro_f1 = float(np.mean(f1_s).item())
    wf1 = f1_score(_trg, _prd, average='weighted')
    wf1 = float(wf1)

    # Average precision of class 1: presence of A/H.
    ap_cl_1 = 0.0
    if logits is not None:
        probs = softmax(logits, h=1.)
        ap_cl_1 = average_precision(probs_cl_1=probs[:, 1], gt=gt)

    perfs = {
        constants.CL_ACC: cl_acc,
        constants.CFUSE_MARIX: conf_mtx,
        constants.F1_POS: f1_cl_1,
        constants.F1_NEG: f1_cl_0,
        constants.W_F1: wf1,
        constants.MACRO_F1: macro_f1,
        constants.AP_POS: ap_cl_1
    }

    return perfs


def run():
    import torch
    from abaw5_pre_processing.dlib.utils.reproducibility import set_seed
    set_seed(0)
    n = 1024
    ncls = 2
    pred_logits = torch.rand(n, ncls)
    pred = torch.argmax(pred_logits, dim=1, keepdim=False)
    gt = np.random.randint(low=0, high=2, size=(n,))

    perfs = bah_perfs(gt, pred.numpy(), pred_logits.numpy())
    for k in perfs:
        print(f"{k}: {perfs[k]}")

    # If logits are not available.
    perfs = bah_perfs(gt, pred.numpy(), logits=None)


if __name__ == "__main__":
    run()