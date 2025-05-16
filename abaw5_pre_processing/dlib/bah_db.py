"""
Crop, align faces for C-EXPR-DB. Creates splits.
"""
import math
import random
import copy
import argparse
import csv
import sys
import os
from os.path import join, dirname, abspath, basename
from typing import List
import datetime
from random import shuffle
from pprint import pprint


import numpy as np
import tqdm
import yaml
from PIL import Image
import cv2

root_dir = dirname(dirname(abspath(__file__)))
sys.path.append(root_dir)
master_root = dirname(root_dir)
sys.path.append(master_root)

from abaw5_pre_processing.dlib.face_landmarks.retinaface_align import RetinaFaceAlign
from abaw5_pre_processing.dlib.utils.tools import chunks_into_n
from abaw5_pre_processing.dlib.utils.tools import better_chunks_into_n


TIME_FORMAT = "%H:%M:%S.%f"
DEFAULT_HEADER = ['Begin Time - hh:mm:ss.ms',
                  'End Time - hh:mm:ss.ms',
                  'Fearfully Surprised',
                  'Happily Surprised',
                  'Sadly Surprised',
                  'Disgustedly Surprised',
                  'Angrily Surprised',
                  'Sadly Fearful',
                  'Sadly Angry',
                  'Other'
                  ]


if "HOST_XXX" in os.environ.keys():
    if os.environ['HOST_XXX'] == 'gsys':
        # CUDA_VISIBLE_DEVICES="" python dlib/face_landmarks/face_align.py
        import tensorflow
        path = join(os.environ['CUSTOM_CUDNN'],
                    'cudnn-10.1-linux-x64-v7.6.0.64/lib64/libcudnn.so')
        tensorflow.load_library(path)
        tensorflow.config.set_visible_devices([], 'GPU')
        print(path, 'Tensorflow has been loaded early, '
                    'and gpu-usage has been disabled')


# from abaw5_pre_processing.dlib.face_landmarks import FaceAlign
from abaw5_pre_processing.dlib.face_landmarks.faceevolve_align import FaceEvolveAlign
from abaw5_pre_processing.dlib.utils.tools import Dict2Obj
from abaw5_pre_processing.dlib.utils.tools import get_root_wsol_dataset
from abaw5_pre_processing.dlib.utils.shared import find_files_pattern
from abaw5_pre_processing.dlib.utils.tools import check_box_convention

from abaw5_pre_processing.dlib.utils.shared import announce_msg

# from abaw5_pre_processing.project.abaw5 import _constants
import constants

from abaw5_pre_processing.dlib.utils.reproducibility import set_seed
from abaw5_pre_processing.dlib.datasets.default_labels_order import ORDERED_EMOTIONS

from abaw5_pre_processing.dlib.datasets.ds_shared import CROP_SIZE
from abaw5_pre_processing.dlib.datasets.ds_shared import SEED
from abaw5_pre_processing.dlib.datasets.ds_shared import ALIGN_DEEP_FACE
from abaw5_pre_processing.dlib.datasets.ds_shared import ALIGN_FACE_EVOLVE

from abaw5_pre_processing.dlib.datasets.ds_shared import class_balance_stat
from abaw5_pre_processing.dlib.datasets.ds_shared import dump_set
from abaw5_pre_processing.dlib.datasets.ds_shared import print_args_info
from abaw5_pre_processing.dlib.datasets.ds_shared import per_cl_stats


def log_msg(msg: str, logfile):
    print(msg)
    if logfile is not None:
        logfile.write(f"{msg}\n")
        logfile.flush()


def get_all_videos(p: str) -> List[dict]:
    l = find_files_pattern(p, '*.mp4')
    out_f = dict()
    out_dia_utt = dict()

    for f in l:
        if not basename(f).startswith('dia'):
            continue

        b = basename(f).split('.')[0]
        dia_id = b.split('_')[0].split('dia')[1]
        utt_id = b.split('_')[1].split('utt')[1]

        try:
            i = int(dia_id)
        except:
            print(f"Found issues with {f} @ dialogue id.")

        try:
            i = int(utt_id)
        except:
            print(f"Found issues with {f} @ utterance id.")

        assert f not in out_f, f
        out_f[f] = [dia_id, utt_id]
        k = f"{dia_id}_{utt_id}"
        assert k not in out_dia_utt, k
        out_dia_utt[k] = f

    results = [out_f, out_dia_utt]
    return results


def is_ascii(s):
    return all(ord(c) < 128 for c in s)


def is_video_corrupted(p_vid: str, log_file) -> bool:
    corrupt_video = False

    try:
        cap = cv2.VideoCapture(p_vid)
        if not cap.isOpened():
            corrupt_video = True
    except cv2.error as e:
        corrupt_video = True
    except Exception as e:
        corrupt_video = True

    if corrupt_video:
        msg = f"Corrupted: {p_vid}"
        print(msg)
        if log_file is not None:
            log_file.write(f"{msg}\n")


    return corrupt_video


def simplify_csv():
    baseurl = get_root_wsol_dataset()
    ds = constants.MELD
    _dir_meld = join(baseurl, ds)

    _map = {
        constants.VALIDSET: 'dev_sent_emo.csv',
        constants.TRAINSET: 'train_sent_emo.csv',
        constants.TESTSET: 'test_sent_emo.csv'
    }

    keep = {
        'Utterance': 1,
        'Emotion': 3,
        'Dialogue_ID': 5,
        'Utterance_ID': 6
    }

    _remap_labels = {
        'surprise': constants.SURPRISE,
        'fear': constants.FEAR,
        'disgust': constants.DISGUST,
        'joy': constants.HAPPINESS,
        'sadness': constants.SADNESS,
        'anger': constants.ANGER,
        'neutral': constants.NEUTRAL
    }

    ordered_labels = ORDERED_EMOTIONS[ds]

    folds_dir = join(master_root, 'folds', ds, f"split-0")
    os.makedirs(folds_dir, exist_ok=True)
    log_file = open(join(_dir_meld, 'log.txt'), 'w')

    log_file.write(f"Dataset: {ds}\n")

    with open(join(folds_dir, "class_id.yaml"), 'w') as f:
        yaml.dump(ordered_labels, f)

    with open(join(_dir_meld, "class_id.yaml"), 'w') as f:
        yaml.dump(ordered_labels, f)


    for k in _map:
        f_b_name = _map[k]
        f_name = join(_dir_meld, f_b_name)
        out_file = join(_dir_meld, f"{k}.txt")
        fout = open(out_file, 'w', encoding='utf-8')

        msg = f"split: {k}"
        log_file.write(f"{msg}\n")
        print(msg)

        n_vid_missing = 0

        out_f, out_dia_utt = get_all_videos(join(_dir_meld, 'original', k))
        nvids = 0
        n_corpt = 0  # number of corrpted videos.

        with open(f_name, 'r', encoding="UTF-8") as f:

            csv_reader = csv.reader(f, delimiter=',')
            line_count = 0

            for row in csv_reader:

                # non-ascii.
                row[1] = row[1].replace("\x92", "'")
                row[1] = row[1].replace("\x85", " ")
                row[1] = row[1].replace("\x97", " ")
                row[1] = row[1].replace("\x91", " ")
                row[1] = row[1].replace("\x93", " ")
                row[1] = row[1].replace("\x94", " ")
                row[1] = row[1].replace("\x96", " ")
                row[1] = row[1].replace("\xa0", " ")
                row[1] = row[1].replace("’", "'")
                row[1] = row[1].replace("…", "...")

                if line_count == 0:
                    line_count += 1
                    continue

                ut = row[keep['Utterance']]
                emo = row[keep['Emotion']]
                d_id = row[keep['Dialogue_ID']]
                u_id = row[keep['Utterance_ID']]

                key1 = f"{d_id}_{u_id}"
                if not (key1 in out_dia_utt):
                    n_vid_missing += 1
                    continue  # some files are missing such as dial 110 utt 7
                    # at valid.
                file_name: str = out_dia_utt[key1]
                # check if file is not corrupted.
                if is_video_corrupted(file_name, log_file):
                    n_corpt += 1

                    continue

                file_id = file_name.replace(join(_dir_meld, 'original'), '')

                if file_id.startswith(os.sep):
                    file_id = file_id[1:]

                file_id = file_id.split('.')[0]
                label_str: str = _remap_labels[emo]
                label_int: int = ordered_labels[label_str]
                new_row = f"{file_id},{label_int},{ut}\n"
                fout.write(new_row)
                line_count += 1

                nvids += 1

        fout.close()
        cmd = f"cp {out_file} {folds_dir}"
        os.system(cmd)

        msg = f"split: {k}: missing videos: {n_vid_missing}."
        log_file.write(f"{msg}\n")
        print(msg)

        msg = f"split: {k}: corrupted videos: {n_corpt}."
        log_file.write(f"{msg}\n")
        print(msg)

        msg = f"split: {k}: total number of OK videos: {nvids}."
        log_file.write(f"{msg}\n")
        print(msg)

    log_file.close()


def store_faces(faces: List[np.ndarray], out_dir: str, bname: str, f_cnt: int,
                success: bool, frame_log_file, store_top_n_faces: int):

    os.makedirs(out_dir, exist_ok=True)
    assert store_top_n_faces > 0, store_top_n_faces
    assert isinstance(store_top_n_faces, int), type(store_top_n_faces)

    if store_top_n_faces > 1:
        _out_frame_dir = join(out_dir, f"frame-{f_cnt}")
        os.makedirs(_out_frame_dir, exist_ok=True)

        baseurl = get_root_wsol_dataset()
        _dir_out = join(baseurl, ds, 'cropped_aligned')

        for i, f in enumerate(faces):
            f = Image.fromarray(f, 'RGB')
            fname = join(_out_frame_dir, f"v-{bname}-f-{f_cnt}-face-{i}.jpg")
            f.save(fname, format='JPEG')

            # log file name
            _fname_short = fname.replace(_dir_out, '')
            if _fname_short.startswith(os.sep):
                _fname_short = _fname_short[1:]

            frame_log_file.write(f"{_fname_short},{int(success)}\n")

    elif store_top_n_faces == 1:

        assert len(faces) == 1, len(faces)

        face = faces[0]
        f = Image.fromarray(face, 'RGB')
        fname = join(out_dir, f"frame-{f_cnt}.jpg")
        f.save(fname, format='JPEG')

        # log file name
        _fname_short = fname

        frame_log_file.write(f"{_fname_short},{int(success)}\n")

    else:
        raise NotImplementedError(store_top_n_faces)



def process_one_video(p_video, out_dir: str, store_top_n_faces: int,
                      log_file, nblocks: int, process_block: int, ds: str,
                      split: str, frame_log_file, tag: str):
    assert os.path.isfile(p_video), p_video

    assert isinstance(store_top_n_faces, int), type(store_top_n_faces)
    assert store_top_n_faces > 0, store_top_n_faces
    assert store_top_n_faces == 1, store_top_n_faces

    os.makedirs(out_dir, exist_ok=True)

    issue_log_dir = join(master_root, 'face-crop-issues')
    os.makedirs(issue_log_dir, exist_ok=True)
    issue_log_path = join(issue_log_dir, f"{ds}-{split}-"
                                         f"{nblocks}-{process_block}.txt")

    f_cnt = 0
    corrupt_video = False
    try:
        cap = cv2.VideoCapture(p_video)
        if not cap.isOpened():
            corrupt_video = True
    except cv2.error as e:
        corrupt_video = True
    except Exception as e:
        corrupt_video = True

    if corrupt_video:
        log_file.write(f"video: {p_video}. "
                       f"N-frames: {0}. "
                       f"N-faces: {0}  XXXX CORRUPTED\n")
        log_file.flush()

        with open(issue_log_path, 'a') as fissue:
            fissue.write(f"video {p_video} is corrupted.")

        return -1

    success, frame = cap.read()  # bool, np.ndarray h, w, 3 uint8.
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    bname = basename(p_video).split('.')[0]
    if store_top_n_faces == 1:
        return_all_faces = False
    else:
        return_all_faces = True

    face_extractor = RetinaFaceAlign(out_size=constants.SZ256,
                                     verbose=False,
                                     no_warnings=False,
                                     return_all_faces=return_all_faces,
                                     confidence_threshold=0.9
                                     )
    faces = face_extractor.align(img_path=None, img=frame)
    faces = faces[:store_top_n_faces]

    previous_faces = faces
    if not face_extractor.success:
        with open(issue_log_path, 'a') as fissue:
            fissue.write(f"video {p_video} has no faces at the first frame.\n")

        faces_cnt = 0
    else:
        faces_cnt = len(faces)

    store_faces(faces, join(out_dir, tag), bname, f_cnt,
                face_extractor.success, frame_log_file, store_top_n_faces)
    f_cnt = 1
    status_previous = face_extractor.success

    while success:
        success, frame = cap.read()

        if success:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = face_extractor.align(img_path=None, img=frame)
            faces = faces[:store_top_n_faces]

            # take the previous faces when there are no faces detected at the
            # current frame.
            if not face_extractor.success:
                faces = previous_faces
                status = status_previous
            else:
                previous_faces = faces
                faces_cnt += len(faces)
                status_previous = face_extractor.success
                status = face_extractor.success

            store_faces(faces, join(out_dir, tag), bname, f_cnt,
                        status, frame_log_file, store_top_n_faces)

            f_cnt += 1

    log_file.write(f"video: {p_video}. "
                   f"N-frames: {f_cnt}. "
                   f"N-faces: {faces_cnt}\n")
    log_file.flush()
    frame_log_file.flush()


def crop_faces_align(ds: str, split: str, nblocks: int, process_block: int):
    assert ds == constants.BAH_DB, ds

    baseurl = get_root_wsol_dataset()
    _dir_ds = join(baseurl, ds)
    _dir_in = join(baseurl, ds)
    _dir_out = join(baseurl, ds, 'cropped_aligned')
    os.makedirs(_dir_out, exist_ok=True)

    assert split == constants.TRAINSET, split

    assert nblocks > 0, nblocks
    assert process_block <= nblocks, f"{process_block}, {nblocks}"

    lines = []  # do all folds at once.

    for _split in [constants.TRAINSET, constants.VALIDSET, constants.TESTSET]:
        path_fold = join(_dir_ds, 'folds', f"fold-0", f"{_split}.txt")
        assert os.path.isfile(path_fold), path_fold

        with open(path_fold, 'r') as f:
            content = f.readlines()
            for l in content:
                lines.append(l.strip())

    assert nblocks <= len(lines), f"{nblocks}, {lines}"
    blocks = list(better_chunks_into_n(lines, nblocks))
    block = blocks[process_block]
    store_top_n_faces = 1

    logs_dir = join(_dir_ds, 'logs-crop-face', split)
    os.makedirs(logs_dir, exist_ok=True)
    _dir_log_faces = join(logs_dir, 'log_face_crop_align')
    os.makedirs(_dir_log_faces, exist_ok=True)
    log_file_path = join(_dir_log_faces, f'log-nblocks-{nblocks}-'
                                         f'process-block-{process_block}.txt')
    log_file = open(log_file_path, 'w')

    _dir_frame_log_faces = join(_dir_ds, 'frame-logs-crop-face', split,
                                'frame-log_face_crop_align')
    os.makedirs(_dir_frame_log_faces, exist_ok=True)
    frame_log_file_path = join(_dir_frame_log_faces,
                               f'frame-log-nblocks-{nblocks}-'
                               f'process-block-{process_block}.txt')
    frame_log_file = open(frame_log_file_path, 'w')

    for l in block:
        _id = l.split(',')[0]
        path_video = join(_dir_in, f"{_id}")
        assert os.path.isfile(path_video), path_video

        if _id.startswith(os.sep):
            _id = _id[1:]

        process_one_video(path_video, _dir_out, store_top_n_faces, log_file,
                          nblocks, process_block, ds, split, frame_log_file,
                          _id)

    log_file.close()
    frame_log_file.close()

    print(f"Done cropping faces: {ds}, {split}, nblocks: {nblocks},"
          f" for block {process_block}.")


def is_timestamp_valid(t_stamp) -> bool:

    try:
        datetime.datetime.strptime(t_stamp, TIME_FORMAT)
    except:
        return False

    return True


def get_time(t_stamp: str):
    return datetime.datetime.strptime(t_stamp, TIME_FORMAT)


def read_annotation(path_annot: str, video_path) -> list:
    results = []

    with open(path_annot, 'r', encoding="UTF-8") as f:

        csv_reader = csv.reader(f, delimiter=',')
        line_count = 0

        for row in csv_reader:
            if line_count == 0:
                assert row == DEFAULT_HEADER, row
                line_count += 1
                continue

            _start = row[0]
            _end = row[1]
            assert is_timestamp_valid(_start), f"{_start} of {path_annot}"
            assert is_timestamp_valid(_end), f"{_end} of {path_annot}"
            _delta = get_time(_end) - get_time(_start)
            _delta = _delta.total_seconds()
            assert _delta > 0, _delta

            # label
            bin_labels_str = row[2:]
            bin_labels_int = [0 for _ in range(len(bin_labels_str))]

            for i in range(len(bin_labels_str)):
                x = bin_labels_str[i]
                if x == '':
                    bin_labels_int[i] = 0
                else:
                    x = int(x)
                    assert x == 1, f"{x}, {path_annot}"
                    bin_labels_int[i] = 1

            assert sum(bin_labels_int) == 1, sum(bin_labels_int)  # only one
            # label per timestamp
            j = bin_labels_int.index(1)
            assert 0 <= j <= 7, f"{j} | {path_annot}"  # 'Other' class is
            # included.
            label_str = DEFAULT_HEADER[2:][j]
            msg = f"{label_str} | {path_annot}"
            assert label_str in constants.EXPRESSIONS, msg

            results.append([video_path, _start, _end, _delta, label_str])

            line_count += 1

    return results


def build_video_name(des_dir: str, bname: str, _label_fmt: str) -> str:
    i = 0
    while True:
        dest_video = join(des_dir, f"{bname}_{_label_fmt}_{i}.mp4")
        if not os.path.isfile(dest_video):
            return dest_video
        i += 1


def truncate_one_video(video_path: str, annotation_path: str,
                       out_dir: str, log_file):
    c_status = is_video_corrupted(video_path, log_file)

    timestamps = read_annotation(annotation_path, video_path)
    bname = basename(video_path).split('.')[0]

    msg = f"Trimming {video_path} via {annotation_path}: START ..."
    log_msg(msg, log_file)

    for seg in timestamps:
        _, _start, _end, _delta, _label = seg
        _label_fmt = _label.replace(' ', '-')
        des_dir = join(out_dir, _label_fmt)
        os.makedirs(des_dir, exist_ok=True)

        dest_video = build_video_name(des_dir, bname, _label_fmt)

        cmd = f"ffmpeg -y -ss {_start} -to {_end} " \
              f"-i {video_path} -c:v copy -c:a copy {dest_video}"
        print(f"Running: {cmd}")
        os.system(cmd)

        c_status = is_video_corrupted(dest_video, log_file)
        msg = f"clip: {_start} to {_end} {_delta} (s) with label {_label}"
        log_msg(msg, log_file)

    msg = f"There are {len(timestamps)} clips in {video_path}"
    log_msg(msg, log_file)
    msg = 80 * "="
    log_msg(msg, log_file)


def dump_timestamps(timestamps: list, outfile: str):
    f = open(outfile, 'w')

    for seg in timestamps:
        video_path, _start, _end, _delta, _label = seg
        msg = f"{video_path},{_start},{_end},{_delta},{_label}"
        f.write(f"{msg}\n")

    f.close()


def gather_stats_video(annotation_path: str, video_path: str) -> list:
    timestamps = read_annotation(annotation_path, video_path)
    return timestamps


def pull_stats(t_stamps: list, log_file):

    stats = {}

    for seg in t_stamps:
        _, _start, _end, _delta, _label = seg

        if _label not in stats:
            stats[_label] = [_delta]
        else:
            stats[_label].append(_delta)

    duration = 0.0
    all_clips = 0
    log_msg(f"{80 * '='}\nStats of clips:", log_file)
    for _label in stats:
        secs = stats[_label]
        n = len(stats[_label])
        total_s = sum(secs)
        total_m = total_s / 60.

        duration = duration + total_s
        all_clips = all_clips + n

        msg = f"Class: {_label} has {n} clips. " \
              f"Total duration: {total_s: .3f} (s), " \
              f"or {total_m: .3f} (mins). "
        log_msg(msg, log_file)

    msg = f"TOTAL:  {all_clips} clips. " \
          f"Total duration: {duration:.3f} (s), or " \
          f"{duration / 60.: .3f} (mins). "
    log_msg(msg, log_file)


def truncate_all_videos():
    ds = constants.C_EXPR_DB
    baseurl = get_root_wsol_dataset()
    videos_folders = join(baseurl, ds, 'videos')
    annot_dir = join(baseurl, ds, 'annotation')
    out_dir = join(baseurl, ds, 'trimmed_videos')
    if os.path.isdir(out_dir):
        cmd = f"rm -r {out_dir}"
        os.system(cmd)

    os.makedirs(out_dir, exist_ok=True)

    l_videos = find_files_pattern(videos_folders, '*.mp4')
    timestamps = []

    log_file_path = join(baseurl, ds, 'log.txt')
    log_file = open(log_file_path, 'w')

    for v in tqdm.tqdm(l_videos, ncols=80, total=len(l_videos)):
        b = basename(v).split('.')[0]
        annot_file = join(annot_dir, f"{b}.csv")
        assert os.path.isfile(annot_file), annot_file

        truncate_one_video(v, annot_file, out_dir, log_file)

        timestamps = timestamps + gather_stats_video(annot_file, v)

    log_msg(80 * "=", log_file)
    msg = f"Total number of original videos: {len(l_videos)}\n" \
          f"Total number of trimmed videos: {len(timestamps)}"
    log_msg(msg, log_file)
    pull_stats(timestamps, log_file)

    dump_timestamps(timestamps, join(baseurl, ds, 'timestamps.txt'))

    log_file.close()


def get_list_clips_per_cl(clips_dir: str) -> dict:
    out = dict()
    folders = [name for name in os.listdir(clips_dir) if os.path.isdir(join(
        clips_dir, name))]

    set_seed(SEED)

    for name in folders:
        l_files = find_files_pattern(join(clips_dir, name), '*.mp4')

        for i in range(100):
            shuffle(l_files)

        out[name] = l_files

    return out


def create_folders(data: dict, n: int) -> dict:
    assert isinstance(n, int), type(n)
    assert n > 1, n

    out = dict()

    for cl in data:
        out[cl] = list(better_chunks_into_n(data[cl], n))
        for i, c in enumerate(out[cl]):
            assert len(c) > 0, f"{cl} | {i} | {len(c)}"

    folds = dict()

    for i in range(n):
        samples = []
        for cl in out:
            samples = samples + out[cl][i]

        folds[i] = samples

    return folds


def dump_split(data: list, dest_path: str, trimmed_dir_path: str,
               transcript: dict):
    fout = open(dest_path, 'w', encoding='utf-8')

    ds = constants.C_EXPR_DB
    ordered_labels = ORDERED_EMOTIONS[ds]

    for f in data:
        _id = f.replace(trimmed_dir_path, '')
        if _id.startswith(os.sep):
            _id = _id[1:]

        label = basename(_id).split('_')[1]
        label = label.replace('-', ' ')
        assert label in constants.EXPRESSIONS, label

        if label == constants.OTHER:
            label_int = 7
        else:
            label_int = ordered_labels[label]

        _id = _id.split('.')[0]  # remove video extension.
        tr = transcript[_id]
        line = f"{_id},{label_int},{tr}\n"
        fout.write(line)

    fout.close()


def split_data(transcript: dict):

    ds = constants.C_EXPR_DB
    baseurl = get_root_wsol_dataset()
    videos_folders = join(baseurl, ds, 'videos')
    trimmed_v_dir = join(baseurl, ds, 'trimmed_videos')

    all_clips_per_cl = get_list_clips_per_cl(trimmed_v_dir)

    n = 5  # number of folds.
    folds = create_folders(all_clips_per_cl, n)

    splits = dict()

    for i in range(n):
        _train = []
        _valid = []
        _valid = folds[i]
        for j in range(n):
            if i == j:
                continue

            _train = _train + folds[j]

        splits[i] = {
            constants.TRAINSET: copy.deepcopy(_train),
            constants.VALIDSET: copy.deepcopy(_valid)
        }

    # sanity check.
    for i in range(n):
        _train = splits[i][constants.TRAINSET]
        _valid = splits[i][constants.VALIDSET]

        for e in _train:
            assert e not in _valid, f"{i} | {e}"

    # dump splits.
    print('Dumping splits...')
    ordered_labels = ORDERED_EMOTIONS[ds]
    for s in splits:
        _train = splits[s][constants.TRAINSET]
        _valid = splits[s][constants.VALIDSET]

        _dir_fold = join(baseurl, ds, 'folds', f'split-{s}')
        os.makedirs(_dir_fold, exist_ok=True)

        with open(join(_dir_fold, "class_id.yaml"), 'w') as f:
            full_labels = copy.deepcopy(ordered_labels)
            full_labels[constants.OTHER] = 7  # include class 'Other'
            yaml.dump(full_labels, f)

        # train
        path_split = join(_dir_fold, f"{constants.TRAINSET}.txt")
        dump_split(_train, path_split, trimmed_v_dir, transcript)

        # valid
        path_split = join(_dir_fold, f"{constants.VALIDSET}.txt")
        dump_split(_valid, path_split, trimmed_v_dir, transcript)

        # test == valid
        path_split = join(_dir_fold, f"{constants.TESTSET}.txt")
        dump_split(_valid, path_split, trimmed_v_dir, transcript)

    # copy folds to root.
    dest = join(master_root, 'folds', ds)
    cmd = f"cp -r {join(baseurl, ds, 'folds')}/* {dest}"
    print(f"Copying folds to root {dest}")
    os.system(cmd)


def isascii(s):
    """Check if the characters in string s are in ASCII, U+0-U+7F."""
    return len(s) == len(s.encode())


def load_transcript() -> dict:
    ds = constants.C_EXPR_DB
    baseurl = get_root_wsol_dataset()
    transcript_org_pth = join(baseurl, ds, 'clips_transcription_cleaner.csv')
    with open(transcript_org_pth, 'r', encoding="UTF-8") as f:
        csv_reader = csv.reader(f, delimiter=',')
        data = dict()

        for row in csv_reader:
            _id, tr = row
            assert _id not in data, _id

            if not isascii(tr):
                print(f"NOT ASCII: {_id} | {tr}")

            data[_id] = tr

    return data


if __name__ == "__main__":
    root_dir = dirname(dirname(abspath(__file__)))
    sys.path.append(root_dir)

    # 4: crop and align faces.
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds", type=str, default=constants.AH_DB,
                        help="dataset name.")
    parser.add_argument("--split", type=str, default=constants.TRAINSET,
                        help="Name of the split: train, valid, test.")
    parser.add_argument("--nblocks", type=int, default=20,
                        help="Total number of blocks to divide the split into."
                             "You can call this code for each block "
                             "separately.")
    parser.add_argument("--process_block", type=int, default=0,
                        help="Among all the blocks, which block to process.")
    # a block is a list of videos from the split subset.

    parsedargs = parser.parse_args()
    ds = parsedargs.ds
    split = parsedargs.split
    nblocks = parsedargs.nblocks
    process_block = parsedargs.process_block

    assert split in [constants.TRAINSET], split

    crop_faces_align(ds=ds, split=split, nblocks=nblocks,
                     process_block=process_block)
