import sys
from os.path import dirname, abspath, join


import torch.nn as nn
from abc import ABCMeta, abstractmethod

root_dir = dirname(dirname(abspath(__file__)))
sys.path.append(root_dir)


__all__ = ['BaseBackbone']


class BaseBackbone(nn.Module, metaclass=ABCMeta):
    """Base backbone.

    This class defines the basic functions of a backbone.
    Any backbone that inherits this class should at least
    define its own `forward` function.

    """

    def __init__(self):
        super(BaseBackbone, self).__init__()

    def init_weights(self, pretrained=None):
        """Init backbone weights

        Args:
            pretrained (str | None): If pretrained is a string, then it
                initializes backbone weights by loading the pretrained
                checkpoint. If pretrained is None, then it follows default
                initializer or customized initializer in subclasses.
        """
        if isinstance(pretrained, str):
            raise NotImplementedError

        elif pretrained is None:
            # use default initializer or customized initializer in subclasses
            pass
        else:
            raise TypeError('pretrained must be a str or None.'
                            f' But received {type(pretrained)}.')

    @abstractmethod
    def forward(self, x):
        """Forward computation

        Args:
            x (tensor | tuple[tensor]): x could be a Torch.tensor or a tuple of
                Torch.tensor, containing input data for forward computation.
        """
        pass

    def train(self, mode=True):
        """Set module status before forward computation

        Args:
            mode (bool): Whether it is train_mode or test_mode
        """
        super(BaseBackbone, self).train(mode)