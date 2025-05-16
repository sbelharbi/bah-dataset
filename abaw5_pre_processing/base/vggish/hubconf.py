import sys
from os.path import dirname, abspath, join, basename, expanduser, normpath


root_dir = dirname(dirname(dirname((abspath(__file__)))))
sys.path.append(root_dir)


dependencies = ['torch', 'numpy', 'resampy', 'soundfile']

from abaw5_pre_processing.base.vggish.vggish import VGGish

model_urls = {
    'vggish': 'https://github.com/harritaylor/torchvggish/'
              'releases/download/v0.1/vggish-10086976.pth',
    'pca': 'https://github.com/harritaylor/torchvggish/'
           'releases/download/v0.1/vggish_pca_params-970ea276.pth'
}


def vggish(**kwargs):
    model = VGGish(urls=model_urls, **kwargs)
    return model
