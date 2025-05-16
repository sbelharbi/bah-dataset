import math
import random
import copy
import argparse
import csv
import sys
import os
from os.path import join, dirname, abspath, basename
from typing import List

import more_itertools as mit

import numpy as np
import tqdm
import yaml
from PIL import Image
import cv2

root_dir = dirname(dirname(abspath(__file__)))
sys.path.append(root_dir)
master_root = dirname(root_dir)
sys.path.append(master_root)

from abaw5_pre_processing.project.abaw5.configs import get_config
import constants
from abaw5_pre_processing.dlib.utils.shared import announce_msg
from abaw5_pre_processing.dlib.utils.shared import find_files_pattern


def read_fold_content(config: dict, p: str) -> list:
    ds = config['dataset_name']
    if ds in [constants.MELD, constants.C_EXPR_DB,
              constants.C_EXPR_DB_CHALLENGE, constants.AH_DB,
              constants.BAH_DB]:
        with open(p, 'r') as f:
            content = f.readlines()
            lines = []
            for l in content:
                l = l.strip()
                _id = l.split(',')[0]
                _label = l.split(',')[1]
                _label: int = int(_label)
                _p = f"{_id},{_label},"

                _text = l.replace(_p, '')

                lines.append([_id, _label, _text])

        return lines

    else:
        raise NotImplementedError(ds)


def load_csv_split(config: dict, ds: str, split: str) -> list:

    if ds == constants.BAH_DB:
        fold_file = join(master_root, 'folds', ds, "fold-0", f"{split}.txt")

    else:
        fold_file = join(master_root, 'folds', ds, "split-0", f"{split}.txt")

    assert os.path.isfile(fold_file), fold_file

    lines = read_fold_content(config, fold_file)

    return lines


def get_part(lines: list, part: int, nparts: int) -> list:

    split_data = [list(c) for c in mit.divide(nparts, lines)]

    return split_data[part]


def get_root_wsol_dataset():
    baseurl = None
    if "HOST_XXX" in os.environ.keys():
        if os.environ['HOST_XXX'] == 'tay':
            # debug.
            baseurl = f"{os.environ['SBHOME']}/workspace/datasets/neurips25"

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


def load_top_face_all_frames(folder: str, extension: str) -> list:
    assert os.path.isdir(frames_source), frames_source

    list_files = find_files_pattern(folder, f"*.{extension}")
    holder = dict()
    for f in list_files:
        d = basename(dirname(f))
        assert d.startswith('frame-'), d
        i = d.split('-')[1]
        i = int(i)
        if i not in holder:
            holder[i] = [f]
        else:
            holder[i].append(f)

    # sort faces
    for i in holder:
        faces = holder[i]
        # faces = [basename(j) for j in faces]
        faces_n = [basename(name).split('-')[-1].split('.')[0] for name in faces]  #
        # v-dia990_utt4-f-56-face-0.jpg
        mixed = list(zip(faces, faces_n))

        mixed = sorted(mixed, key=lambda x: x[1], reverse=False)
        faces = [item[0] for item in mixed]
        assert len(faces) > 0, len(faces)
        top_face = faces[0]

        holder[i] = top_face

    results = [[holder[i], i] for i in holder]
    results = sorted(results, key=lambda x: x[1], reverse=False)

    all_top_faces = [item[0] for item in results]
    # sanity check
    for f in all_top_faces:
        assert os.path.isfile(f), f
        assert f.endswith(f"-0.{extension}"), f

    return all_top_faces


def load_top_face_all_frames_AH_DB(folder: str, extension: str) -> list:
    assert os.path.isdir(frames_source), frames_source

    list_files = find_files_pattern(folder, f"*.{extension}")
    holder = dict()
    for f in list_files:
        d = basename(f)
        assert d.startswith('frame-'), d
        i = d.split('.')[0].split('-')[1]

        i = int(i)
        if i not in holder:
            holder[i] = [f]
        else:
            holder[i].append(f)

    # sanity check
    for i in holder:
        assert len(holder[i]) == 1, f"{holder[i]} | {len(holder[i])}"


    results = [[holder[i][0], i] for i in holder]
    results = sorted(results, key=lambda x: x[1], reverse=False)

    all_top_faces = [item[0] for item in results]
    # sanity check
    for f in all_top_faces:
        assert os.path.isfile(f), f
        assert f.endswith(f".{extension}"), f

    return all_top_faces


def load_top_face_all_frames_BAH_DB(folder: str, extension: str) -> list:
    assert os.path.isdir(frames_source), frames_source

    list_files = find_files_pattern(folder, f"*.{extension}")
    holder = dict()
    for f in list_files:
        d = basename(f)
        assert d.startswith('frame-'), d
        i = d.split('.')[0].split('-')[1]

        i = int(i)
        if i not in holder:
            holder[i] = [f]
        else:
            holder[i].append(f)

    # sanity check
    for i in holder:
        assert len(holder[i]) == 1, f"{holder[i]} | {len(holder[i])}"


    results = [[holder[i][0], i] for i in holder]
    results = sorted(results, key=lambda x: x[1], reverse=False)

    all_top_faces = [item[0] for item in results]
    # sanity check
    for f in all_top_faces:
        assert os.path.isfile(f), f
        assert f.endswith(f".{extension}"), f

    return all_top_faces


if __name__ == '__main__':
    parser = argparse.ArgumentParser()


    # the "video" modality is done separately. first, faces are cropped,
    # aligned, then stored as images using dlib/dataset_name.py
    # we compact the face using this separate code:
    # dlib/compact_face_images.py instead of the ones used in
    # project/abaw5/main.py

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

    assert ds in [constants.MELD, constants.C_EXPR_DB,
                  constants.C_EXPR_DB_CHALLENGE, constants.AH_DB,
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
    log_dir = join(master_root, 'logs-features-extract-compact_faces', ds,
                   split)
    os.makedirs(log_dir, exist_ok=True)

    log_file_path = join(log_dir, f"{ds}-{split}-{nparts}-"
                                  f"{part}.txt")
    logfile = open(log_file_path, 'w')
    logfile.write(f"{msg}\n")

    config = get_config(ds)

    lines = load_csv_split(config, ds, split)
    samples = get_part(lines, part, nparts)

    data_root = get_root_wsol_dataset()
    frames_source = join(data_root, ds, 'cropped_aligned')
    dest_dir = join(data_root, ds, 'features/compacted_48')

    for l in tqdm.tqdm(samples, ncols=80, total=len(samples)):
        _id, label, txt = l

        if ds == constants.MELD:
            sr_frames = join(frames_source, _id)

        elif ds == constants.AH_DB:
            sr_frames = join(frames_source, _id)

        elif ds == constants.BAH_DB:
            sr_frames = join(frames_source, _id)

        elif ds == constants.C_EXPR_DB:
            z = _id.split(os.sep)[1]  # Other/53_Other_0 --> 53_Other_0
            sr_frames = join(frames_source, z)

        elif ds == constants.C_EXPR_DB_CHALLENGE:
            z = _id.split(os.sep)[1]  # videos/01
            sr_frames = join(frames_source, z)

        else:
            raise NotImplementedError(ds)

        assert os.path.isdir(sr_frames), sr_frames

        # dest
        dest = join(dest_dir, _id)
        os.makedirs(dest, exist_ok=True)

        # load top face of each frame to compact them at once:
        ext = 'jpg'
        if ds == constants.AH_DB:
            all_top_faces = load_top_face_all_frames_AH_DB(sr_frames, ext)

        elif ds == constants.BAH_DB:
            all_top_faces = load_top_face_all_frames_BAH_DB(sr_frames, ext)

        else:
            all_top_faces = load_top_face_all_frames(sr_frames, ext)

        n = len(all_top_faces)
        # sanity check
        # bert_path = join(dest, 'bert.npy')
        # assert os.path.isfile(bert_path), bert_path
        # bert_data = np.load(bert_path, mmap_mode='c')
        # assert bert_data.shape[0] == n, f"{bert_data.shape[0]} | {n} | {_id}"

        sizes = []
        modalities = ['bert', 'vggish', 'EXPR_continuous_label']
        for zzz in modalities:
            mod_p = join(dest, f'{zzz}.npy')
            assert os.path.isfile(mod_p), mod_p
            _data = np.load(mod_p, mmap_mode='c')
            sizes.append([zzz, _data.shape[0]])

        issue = False
        for modal, sz in sizes:
            if ds == constants.MELD:
                assert sz == n, f"{modal} | {sz} | {n} | {dest}"

            elif ds == constants.AH_DB:
                assert sz == n, f"{modal} | {sz} | {n} | {dest}"

            elif ds == constants.BAH_DB:
                assert sz == n, f"{modal} | {sz} | {n} | {dest}"

            else:
                # first pass: Comment the next line.
                # second pass. Uncomment this next line.
                assert sz == n, f"{modal} | {sz} | {n} | {dest}"

                if sz != n:
                    # for C_EXPR_DB_CHALLENGE, the extracted number of frames
                    # are always lower than the features for 5 videos.
                    # in this case, we lower the dim of the features to the
                    # number of frames.
                    # this issue occurs also with C_EXPR_DB.

                    video_id = basename(dest)
                    if ds == constants.C_EXPR_DB_CHALLENGE:
                        assert video_id in ['09', '10', '22', '24',
                                            '45'], video_id
                    assert n < sz, f"{n} | {sz}"

                    mod_p = join(dest, f'{modal}.npy')
                    assert os.path.isfile(mod_p), mod_p
                    _data = np.load(mod_p)
                    _new_data = _data[:n]
                    np.save(mod_p, _new_data)


            if sz != n:
                print(f"Issue: {modal} | {sz} | {n} | {dest}")
                issues = True

        frame_matrix = np.zeros((
            n, config['video_size'], config['video_size'], 3), dtype=np.uint8)

        for j, frame in enumerate(all_top_faces):
            current_frame = Image.open(frame)

            frame_matrix[j] = current_frame.resize((config['video_size'],
                                                    config['video_size'])
                                                   )

        cmp_faces_path = join(dest, f'{constants.VIDEO}.npy')
        np.save(cmp_faces_path, frame_matrix)
        msg = f"{_id},OK"
        announce_msg(msg)
        logfile.write(f"{msg}\n")

    msg = f"Processing {ds} {split} block: {part}/{nparts} ... [DONE]"
    announce_msg(msg)
    logfile.write(f"{msg}\n")

    logfile.close()
