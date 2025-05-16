import os
import sys
from os.path import dirname, abspath, join
from typing import Type, Any, Callable, Union, List, Optional, Tuple
import fnmatch

from copy import deepcopy

import yaml
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision


root_dir = dirname(dirname(abspath(__file__)))
sys.path.append(root_dir)

import constants
from fer_models.apvit import PoolingVitClassifier


__all__ = ['FerModel', 'load_bah_visual_backbone']

_IMAGENET = {
    constants.RESNET18: 'checkpoints/resnet18-f37072fd.pth',
    constants.RESNET34: 'checkpoints/resnet34-b627a593.pth',
    constants.RESNET50: 'checkpoints/resnet50-11ad3fa6.pth',
    constants.RESNET101: 'checkpoints/resnet101-cd907fc2.pth',
    constants.RESNET152: 'checkpoints/resnet152-f82ba261.pth',
}

_DEFAULT_APVIT = {
    "k": 160,
    "r": 0.9,
    "dense_dims": 'None',
    "attn_method": constants.ATT_SUM_ABS_1,
    "normalize_att": False,
    "apply_self_att": False,
    "hid_att_dim": 128,
    "pretrained": None,
    "freeze_backbone": False
}


class _Dim_Reduc(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super(_Dim_Reduc, self).__init__()

        self.layer = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.BatchNorm1d(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.ReLU(inplace=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)



class FerModel(nn.Module):
    def __init__(self, ncls: int, model_type: str, pretrained: bool,
                 apvit_config: dict, reduce_dim: int = None):
        super(FerModel, self).__init__()

        dir_pre_w = join(root_dir, constants.FOLDER_PRETRAINED_IMAGENET)
        torch.hub.set_dir(dir_pre_w)
        self.ncls = ncls
        self.model_type = model_type
        self.num_ftrs = 0

        assert reduce_dim is None or isinstance(reduce_dim, int), reduce_dim
        self.reduce_dim = reduce_dim

        if model_type.startswith('resnet'):
            self.model = torch.hub.load('pytorch/vision', model_type,
                                        weights=None, force_reload=False)

            if pretrained:
                cpu_dev = torch.device("cpu")
                path_w = join(dir_pre_w, _IMAGENET[model_type])
                w = torch.load(path_w, map_location=cpu_dev)
                self.model.load_state_dict(w, strict=True)

            num_ftrs = self.model.fc.in_features
            self.num_ftrs = num_ftrs
            self.model.fc = nn.Identity()

            self.layer_reduce_dim = nn.Identity()

            if isinstance(reduce_dim, int):
                if num_ftrs != reduce_dim:
                    self.layer_reduce_dim = _Dim_Reduc(num_ftrs, reduce_dim)
                    self.num_ftrs = reduce_dim

            self.classifier = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(self.num_ftrs, ncls)
            )

        elif model_type == constants.APVIT:
            config = apvit_config
            self.model = PoolingVitClassifier(num_classes=ncls, **config)

            self.apvit_in_dim = 768
            self.layer_reduce_dim = nn.Identity()

            if isinstance(reduce_dim, int):
                if self.apvit_in_dim != reduce_dim:
                    self.layer_reduce_dim = _Dim_Reduc(self.apvit_in_dim,
                                                       reduce_dim)
                    self.apvit_in_dim = reduce_dim

            self.model.re_create_cl_head(self.ncls, in_dim=self.apvit_in_dim)

        else:
            raise NotImplementedError(model_type)

        self.compound_cl_head = None
        self.basic_ncls = ncls

    def create_compound_cl_head(self, ncls_c: int, dense_dims: str = 'None'):
        if self.model_type.startswith('resnet'):
            assert dense_dims == 'None', dense_dims

            self.compound_cl_head = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(self.num_ftrs, ncls_c)
            )

        elif self.model_type == constants.APVIT:
            self.compound_cl_head = self.model.create_compound_cl_head(
                ncls_c, dense_dims, in_dim=self.apvit_in_dim)

        else:
            raise NotImplementedError(self.model_type)

    def re_create_cl_head(self, ncls: int):
        assert self.compound_cl_head is None, "You cant do this case."

        if self.model_type.startswith('resnet'):
            self.classifier = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(self.num_ftrs, ncls)
            )

        elif self.model_type == constants.APVIT:
            self.model.re_create_cl_head(ncls, in_dim=self.apvit_in_dim)

        else:
            raise NotImplementedError(self.model_type)

    def flush(self):
        if hasattr(self.model, 'flush'):
            self.model.flush()

    @staticmethod
    def freeze_modules(modules):
        for module in modules:
            for param in module.parameters():
                param.requires_grad = False

            if isinstance(module, torch.nn.BatchNorm2d):
                module.eval()

            if isinstance(module, torch.nn.Dropout):
                module.eval()

    def freeze_classifier_head(self):
        if self.model_type.startswith('resnet'):
            modules = self.classifier.modules()

        elif self.model_type == constants.APVIT:
            modules = self.model.classification_head.modules()

        else:
            raise NotImplementedError

        self.freeze_modules(modules)

        if self.compound_cl_head is not None:
            modules = self.compound_cl_head.modules()
            self.freeze_modules(modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if self.model_type.startswith('resnet'):
            ft = self.model(x)
            ft = self.layer_reduce_dim(ft)

        elif self.model_type == constants.APVIT:
            ft = self.model(x)
            ft = self.layer_reduce_dim(ft)

        else:
            raise NotImplementedError

        return ft

    def _original_forward(self, x: torch.Tensor) -> Tuple[torch.Tensor,
                                                          torch.Tensor,
                                                          torch.Tensor]:

        if self.model_type.startswith('resnet'):
            ft = self.model(x)
            ft = self.layer_reduce_dim(ft)
            logits = self.classifier(ft)

        elif self.model_type == constants.APVIT:
            ft = self.model(x)
            ft = self.layer_reduce_dim(ft)
            logits = self.model.classification_head(ft)

        else:
            raise NotImplementedError

        basic_logits = logits

        if self.compound_cl_head is not None:
            logits = self.compound_cl_head(ft)

        return ft, logits, basic_logits

    def classify_ft(self, ft: torch.Tensor = None):
        # returns logits.
        if self.model_type.startswith('resnet'):
            logits = self.classifier(ft)

        elif self.model_type == constants.APVIT:
            logits = self.model.classification_head(ft)
        else:
            raise NotImplementedError

        basic_logits = logits
        if self.compound_cl_head is not None:
            logits = self.compound_cl_head(ft)

        return logits, basic_logits

    def predict_classes(self, logits: torch.Tensor):
        pass


def pre_load_all_resnet_imagenet():
    resnets = [constants.RESNET18, constants.RESNET34, constants.RESNET50,
               constants.RESNET101, constants.RESNET152]
    holder = {
        constants.RESNET18: 'ResNet18_Weights.DEFAULT',
        constants.RESNET34: 'ResNet34_Weights.DEFAULT',
        constants.RESNET50: 'ResNet50_Weights.DEFAULT',
        constants.RESNET101: 'ResNet101_Weights.DEFAULT',
        constants.RESNET152: 'ResNet152_Weights.DEFAULT',
    }

    destination = join(root_dir, constants.FOLDER_PRETRAINED_IMAGENET)
    os.makedirs(destination, exist_ok=True)
    torch.hub.set_dir(destination)

    for md in resnets:
        torch.hub.load('pytorch/vision', md, weights=holder[md],
                       force_reload=True)


def load_bah_visual_backbone(folder_w: str):

    assert os.path.isdir(folder_w), folder_w
    with open(join(folder_w, 'mini_config.yml'), 'r') as fx:
        config = yaml.safe_load(fx)

    ncls = config["num_classes"]
    cl_loss = config['cl_loss']

    model_type = config["model"]['model_type']
    reduce_dim = config["model"]['reduce_dim']

    if reduce_dim == 'None':
        reduce_dim = None
    else:
        reduce_dim = int(reduce_dim)

    pretrained = False
    apvit_config = None

    if model_type == constants.APVIT:
        apvit = config["model"]

        apvit_config = {
            "k": apvit['apvit_k'],
            "r": apvit['apvit_r'],
            "dense_dims": apvit['apvit_dense_dims'],
            "attn_method": apvit['apvit_attn_method'],
            "normalize_att": apvit['apvit_normalize_att'],
            "apply_self_att": apvit['apvit_apply_self_att'],
            "hid_att_dim": apvit['apvit_hid_att_dim'],
            "pretrained": None,
            "freeze_backbone": False
        }

    model = FerModel(ncls=ncls,
                     model_type=model_type,
                     pretrained=pretrained,
                     apvit_config=apvit_config,
                     reduce_dim=reduce_dim
                     )

    wp = join(folder_w, 'model.pt')
    assert os.path.isfile(wp), wp
    all_w = torch.load(wp, map_location=torch.device('cpu'))
    model.load_state_dict(all_w, strict=True)

    model.eval()

    return model


def find_files_pattern(fd_in_, pattern_):
    """
    Find paths to files with pattern within a folder recursively.
    :return:
    """
    msg = f"Folder {fd_in_} does not exist ... [NOT OK]"
    assert os.path.exists(fd_in_), msg

    print(f"Searching file pattern '{pattern_}' @ {fd_in_} ...")

    files = []
    for r, d, f in os.walk(fd_in_):
        for file in f:
            if fnmatch.fnmatch(file, pattern_):
                files.append(os.path.join(r, file))

    return files


def run_load_all_pretrained_models():
    in_fd = join(root_dir, constants.PRETRAINED_BACKBONES,
                 constants.VIDEO, constants.BAH_DB)
    models_paths = find_files_pattern(in_fd, 'model.pt')
    device = torch.device('cuda:0')
    x = torch.rand((32, 3, 112, 112), device=device)

    for f in models_paths:
        print(f"Loading {f} ...")
        folder_model = dirname(f)
        model = load_bah_visual_backbone(folder_model)
        model = model.to(device)
        # transforms not used.
        with torch.no_grad():
            ft = model(x)

        print(ft.shape, x.shape)


def run_resnet():
    resnets = [constants.RESNET18, constants.RESNET34, constants.RESNET50,
               constants.RESNET101, constants.RESNET152]
    ncls = 7
    cuda_dev = torch.device("cuda:1")
    dim = 112  # 224
    x = torch.rand((32, 3, dim, dim), device=cuda_dev)

    for md in resnets:
        model = FerModel(ncls=ncls, model_type=md, pretrained=True,
                         apvit_config=None)
        model = model.to(cuda_dev)
        with torch.no_grad():
            ft = model(x)
        print(ft.shape, x.shape)


def run_apvit_core():
    ncls = 7
    cuda_dev = torch.device("cuda:1")
    x = torch.rand((32, 3, 112, 112), device=cuda_dev)
    model = PoolingVitClassifier(num_classes=ncls, **_DEFAULT_APVIT).to(
        cuda_dev)
    out = model(x)
    print(x.shape, out.shape)


def run_apvit():
    ncls = 7
    cuda_dev = torch.device("cuda:3")
    bsz = 32
    x = torch.rand((bsz, 3, 112, 112), device=cuda_dev)

    model_type = constants.APVIT
    model = FerModel(ncls=ncls, model_type=model_type, pretrained=True,
                     apvit_config=_DEFAULT_APVIT)

    model.create_compound_cl_head(ncls_c=11)

    model.to(cuda_dev)
    with torch.no_grad():
        ft = model(x)
    print(model_type, ft.shape)


if __name__ == "__main__":
    # pre_load_all_resnet_imagenet()
    # run_resnet()
    # run_apvit_core()
    # run_apvit()

    run_load_all_pretrained_models()
