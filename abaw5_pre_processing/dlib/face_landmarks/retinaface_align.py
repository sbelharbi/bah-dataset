import os
import sys
import time
from os.path import dirname, abspath, join, basename
from typing import Optional, Union, Tuple, List
import warnings

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils import model_zoo


from PIL import Image

root_dir = dirname(dirname(dirname(dirname(abspath(__file__)))))
sys.path.append(root_dir)

# from abaw5_pre_processing.project.abaw5 import _constants
import constants
from abaw5_pre_processing.dlib.utils.tools import check_box_convention

_ALIGN_PATH = join(root_dir, 'face_evoLVe/applications/align')
sys.path.append(_ALIGN_PATH)

from face_evoLVe.applications.align.detector import detect_faces
from face_evoLVe.applications.align.align_trans import get_reference_facial_points
from face_evoLVe.applications.align.align_trans import warp_and_crop_face

from retinaface.pre_trained_models import get_model
from retinaface import pre_trained_models

__all__ = ['RetinaFaceAlign']


def retinaface_get_model(model_name: str, max_size: int, device: str = "cpu"):
    """
    Override retinaface/pre_trained_models.get_model
    https://github.com/ternaus/retinaface/blob/master/retinaface/pre_trained_models.py
    to avoid unzipping when doing parallel jobs as they unzip at the same time.

    :param model_name:
    :param max_size:
    :param device:
    :return:
    """
    assert model_name == "resnet50_2020-07-20", model_name

    _models = pre_trained_models.models
    model = _models[model_name].model(max_size=max_size, device=device)

    _d = torch.hub.get_dir()
    _p_w = join(_d, 'checkpoints', 'retinaface_resnet50_2020-07-20.pth')
    assert os.path.isfile(_p_w), _p_w
    state_dict = torch.load(_p_w, map_location="cpu")

    model.load_state_dict(state_dict)

    return model



class RetinaFaceAlign(object):
    """
    Crop and align faces using https://github.com/ternaus/retinaface.
    """
    def __init__(self,
                 out_size: int = constants.SZ256,
                 verbose: bool = False,
                 no_warnings: bool = False,
                 return_all_faces: bool = False,
                 confidence_threshold: float = 0.9
                 ):

        if no_warnings:
            warnings.filterwarnings("ignore")

        assert isinstance(confidence_threshold, float), type(confidence_threshold)
        assert 0 < confidence_threshold < 1, confidence_threshold
        self.confidence_threshold = confidence_threshold

        assert isinstance(out_size, int), type(out_size)
        assert out_size > 0, out_size

        self.out_size = out_size
        self.success = False

        self.return_all_faces = return_all_faces  # if true, all faces are
        # returned, otherwise, single top face is returned.

        assert isinstance(verbose, bool), type(verbose)
        self.verbose = verbose

        scale = out_size / 112.
        self.reference = get_reference_facial_points(
            default_square=True) * scale

        self.device = torch.device("cuda:0")
        _d = torch.hub.get_dir()
        _p_w = join(_d, 'checkpoints', 'retinaface_resnet50_2020-07-20.pth')

        if not os.path.isfile(_p_w):
            self.face_detector = get_model("resnet50_2020-07-20",
                                           max_size=2048,
                                           device=self.device
                                           )
        else:
            self.face_detector = retinaface_get_model("resnet50_2020-07-20",
                                                      max_size=2048,
                                                      device=self.device
                                                      )

        self.face_detector.eval()
        self.face_detector = self.face_detector

    def _reset_success(self):
        self.success = False

    @staticmethod
    def bb_iou(box_a, box_b):
        # intersection_over_union
        # ref: https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
        assert isinstance(box_a, np.ndarray), type(box_a)
        assert isinstance(box_b, np.ndarray), type(box_b)

        assert box_a.ndim == 1, box_a.ndim
        assert box_b.ndim == 1, box_b.ndim
        assert box_a.shape == box_b.shape
        assert box_a.size == 4, box_a.size


        xa = max(box_a[0], box_b[0])
        ya = max(box_a[1], box_b[1])
        xb = min(box_a[2], box_b[2])
        yb = min(box_a[3], box_b[3])

        inter_area = max(0, xb - xa + 1) * max(0, yb - ya + 1)

        box_a_area = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
        box_b_area = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)

        iou = inter_area / float(box_a_area + box_b_area - inter_area)

        return iou

    def get_closest_to_proposal(self,
                                bbxoes: np.ndarray,
                                p: np.ndarray) -> int:

        assert isinstance(bbxoes, np.ndarray), type(bbxoes)
        assert isinstance(p, np.ndarray), type(p)

        assert bbxoes.ndim == 2, bbxoes.ndim
        assert p.ndim == 1, p.ndim
        assert bbxoes.shape[1] == 4, bbxoes.shape[1]
        assert p.size == 4, p.size

        iou = []
        for i in range(bbxoes.shape[0]):
            iou.append(self.bb_iou(bbxoes[i], p))

        iou = np.array(iou)
        return iou.argmax()

    def align(self, img_path: str = None,
              img: np.ndarray = None) -> List[np.ndarray]:

        self._reset_success()

        if img is None:
            input_img = Image.open(img_path, 'r').convert('RGB')
            array_img = np.array(input_img)  # h, w, 3.
        else:
            assert img_path is None
            array_img = img
            input_img = Image.fromarray(img)

        success = True

        try:  # Handle exception
            with torch.no_grad():
                predictions = self.face_detector.predict_jsons(
                    array_img, confidence_threshold=self.confidence_threshold,
                    nms_threshold=0.4)

            # bounding_boxes, landmarks = self.face_detector.predict_jsons(
            #     array_img, confidence_threshold=0.7, nms_threshold=0.4)

            n = len(predictions)  # nbr faces.
            if n == 1:
                if predictions[0]['score'] == -1:
                    n = 0

            bounding_boxes = None
            landmarks = None

            if n > 0:

                bounding_boxes = np.zeros(shape=(n, 5), dtype=float)
                landmarks = [None for _ in range(n)]

                for kk in range(n):
                    bounding_boxes[kk, :-1] = np.array(predictions[kk]['bbox'])
                    bounding_boxes[kk, -1] = np.array(predictions[kk]['score'])

                    landmarks[kk] = predictions[kk]['landmarks']

            # bounding_boxes: np.ndarray (n, 5): n number of bbox.
            # 4 items:
            # 5th item: face score.

            if n > 0:
                check_box_convention(bounding_boxes[:, :4], 'x0y0x1y1',
                                     tolerate_neg=True)

            if n == 0:  # if there is none.
                success = False

            elif n > 0:  # if there is more than 1.
                success = True

        except Exception as e:
            success = False
            print('error ', repr(e))

        self.success = success

        if success:
            return self.get_output_faces(input_img, bounding_boxes, landmarks)

        else:
            if self.verbose:
                print(f"Failed at {img_path}")
            faces = [np.array(input_img).astype(np.uint8)]

            return faces

    def process_one_face(self,
                         input_img: Image.Image,
                         facial5points: list) -> np.ndarray:

        warped_face = warp_and_crop_face(
            np.array(input_img),
            facial5points,
            self.reference,
            crop_size=(self.out_size, self.out_size)
        )
        # <class 'numpy.ndarray'> [0, 255] (sz, sz, 3) uint8
        face: np.ndarray = warped_face

        return face

    def get_output_faces(self,
                         input_img: Image.Image,
                         bounding_boxes: np.ndarray,
                         landmarks: list) -> List[np.ndarray]:
        faces = []
        if self.return_all_faces:
            n = len(landmarks)
            scores = []
            for i in range(n):
                facial5points = landmarks[i]
                faces.append(self.process_one_face(input_img, facial5points))

                s = bounding_boxes[i, -1]
                scores.append(s)

            # order faces from the highest score to lowest
            idx = sorted(range(len(scores)), key=lambda k: scores[k],
                       reverse=True)

            tmp = []
            for i in idx:
                tmp.append(faces[i])

            faces = tmp

        else:
            i = bounding_boxes[:, -1].argmax()
            facial5points = landmarks[i]
            faces.append(self.process_one_face(input_img, facial5points))

        return faces


def test_RetinaFaceAlign():
    aligner = RetinaFaceAlign(out_size=constants.SZ256, verbose=True,
                              no_warnings=True, return_all_faces=True,
                              confidence_threshold=0.9)
    outd = join(root_dir, 'data/debug/out/cropped-faces')
    os.makedirs(outd, exist_ok=True)

    paths = [join(root_dir, 'data/debug/input/test_0006.jpg'),
             join(root_dir, 'data/debug/input/test_0038.jpg'),
             join(root_dir, 'data/debug/input/test_0049.jpg'),
             join(root_dir, 'data/debug/input/test_0067.jpg')
    ]

    for img_path in paths:
        faces = aligner.align(img_path, img=None)
        # plt.imshow(face)
        # plt.axis("off")
        # plt.show()

        print(f"Sucess-----------> {aligner.success}")

        for j in range(len(faces)):
            face = faces[j]
            Image.fromarray(face).save(
                join(outd, f"cropped-{j}-{basename(img_path)}"))

    from abaw5_pre_processing.dlib.utils.shared import find_files_pattern
    paths = find_files_pattern(join(root_dir, 'data/debug/input/faces'),
                               "*.jpg")

    for img_path in paths:
        faces = aligner.align(img_path, img=None)
        # plt.imshow(face)
        # plt.axis("off")
        # plt.show()
        print(f"Sucess-----------> {aligner.success}")
        for j in range(len(faces)):
            face = faces[j]
            Image.fromarray(face).save(
                join(outd, f"cropped-{j}-{basename(img_path)}"))


if __name__ == "__main__":
    test_RetinaFaceAlign()
