import sys
import argparse
import os
from os.path import dirname, abspath, join, basename, expanduser, normpath

root_dir = dirname(dirname(dirname(dirname((abspath(__file__))))))
sys.path.append(root_dir)

from abaw5_pre_processing.project.abaw5.preprocessing import PreprocessingABAW5
from abaw5_pre_processing.project.abaw5.configs import get_config
# from abaw5_pre_processing.project.abaw5 import _constants
import constants
from abaw5_pre_processing.dlib.utils.shared import announce_msg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # the behavior of part has been changed here.
    # now, it allows parallel data processing.
    # data of a single split (train, valid, test) is divided into nparts of
    # equal size. then, we process the part designated by 'part' variable.

    # the "video" modality is done separately. first, faces are cropped,
    # aligned, then stored as images using dlib/dataset_name.py
    # this current code allow to do that then compact the images but we didnt
    # use it. we compact the face using a separate code:
    # dlib/compact_face_images.py

    parser.add_argument('--ds', default=constants.BAH_DB, type=str,
                        help='Dataset name to be processed.'
                        )
    parser.add_argument('--split', default=constants.TRAINSET, type=str,
                        help='Name of the split to process: train/alid/test.'
                        )
    parser.add_argument('--part', default=0, type=int,
                        help='Which part of the data to preprocess? '
                        )
    parser.add_argument('--nparts', default=20, type=int,
                        help='How many parts to divide the data of a split.'
                        )
    args = parser.parse_args()
    ds = args.ds
    split = args.split
    part = args.part
    nparts = args.nparts

    assert ds in [constants.MELD,
                  constants.C_EXPR_DB,
                  constants.C_EXPR_DB_CHALLENGE,
                  constants.AH_DB,
                  constants.BAH_DB], ds
    assert split in [constants.TRAINSET, constants.VALIDSET,
                     constants.TESTSET], split
    assert part < nparts, f"{part} | {nparts}"
    assert nparts > 0, nparts
    assert isinstance(nparts, int), type(nparts)
    assert isinstance(part, int), type(part)

    if ds == constants.C_EXPR_DB:
        assert split in [constants.TRAINSET, constants.VALIDSET], split

    elif ds == constants.MELD:
        assert split in [constants.TRAINSET, constants.TESTSET,
                         constants.VALIDSET], split

    elif ds == constants.AH_DB:
        assert split in [constants.TRAINSET, constants.TESTSET,
                         constants.VALIDSET], split

    elif ds == constants.BAH_DB:
        assert split in [constants.TRAINSET, constants.TESTSET,
                         constants.VALIDSET], split

    elif ds == constants.C_EXPR_DB_CHALLENGE:
        assert split == constants.TRAINSET, split
    else:
        raise NotImplementedError(ds)

    msg = f"Processing {ds} {split} block: {part}/{nparts} ..."
    announce_msg(msg)
    log_dir = join(root_dir, 'logs-features-extract', ds, split)
    os.makedirs(log_dir, exist_ok=True)

    log_file_path = join(log_dir, f"{ds}-{split}-{nparts}-{part}.txt")
    logfile = open(log_file_path, 'w')
    logfile.write(f"{msg}\n")

    pre = PreprocessingABAW5(split, part, nparts, get_config(ds))
    pre.generate_per_trial_info_dict()
    pre.prepare_data(logfile=logfile)

    msg = f"Processing {ds} {split} block: {part}/{nparts} ... [DONE]"
    announce_msg(msg)
    logfile.write(f"{msg}\n")
    logfile.close()


