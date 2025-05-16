import os
import sys
from os.path import dirname, abspath, join, basename, expanduser, normpath

import torch

root_dir = dirname(dirname(dirname((abspath(__file__)))))
sys.path.append(root_dir)
master_dir = dir(root_dir)
sys.path.append(master_dir)

import constants

__all__ = ['get_config']

emotion = ["Arousal", "Valence"]
task = "EXPR"
dataset_name = constants.BAH_DB

video_size = constants.SZ256  # 48
margin_size = 12
VIDEO_EMBEDDING_DIM = 512
MFCC_DIM = 39
EGEMAPS_DIM = 88
VGGISH_DIM = 128
BERT_DIM = 768
VIDEO_TEMPORAL_DIM = 128
MFCC_TEMPORAL_DIM = 32
VGGISH_TEMPORAL_DIM = 32
BERT_TEMPORAL_DIM = 512


def get_root_wsol_dataset():
    baseurl = None
    if "HOST_XXX" in os.environ.keys():
        if os.environ['HOST_XXX'] == 'lab':
            baseurl = f"{os.environ['DATASETSH']}/abaw8"
        elif os.environ['HOST_XXX'] == 'gsys':
            baseurl = f"{os.environ['DATASETSH']}/abaw8"
        elif os.environ['HOST_XXX'] == 'tay':
            baseurl = f"{os.environ['DATASETSH']}/abaw8"
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


def get_home_root_data():
    baseurl = None  # path where all datasets are located.
    if "HOST_XXX" in os.environ.keys():
        if os.environ['HOST_XXX'] == 'tay':
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


CROPPED_ALIGNED_FOLDER_DS = {
    constants.MELD: 'cropped_aligned',
    constants.C_EXPR_DB: 'cropped_aligned',
    constants.C_EXPR_DB_CHALLENGE: 'cropped_aligned',
    constants.AH_DB: 'cropped_aligned',
    constants.BAH_DB: 'cropped_aligned'
}


def get_config(ds: str) -> dict:
    config = {
        "dataset_name": ds,
        "dataset_root_path": join(get_home_root_data(), ds),
        "device": torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu'),

        "annotation_folder": "5th_ABAW_Annotations",

        "crop_align_face": 0,  # Always specify to 1 to use
        "use_mtcnn": 0,  # The author already cropped the faces, no need to do
        # it ourselves.
        "load_cropped_aligned_facial_image": 1,  # Therefore we load them
        # directly.
        "cropped_aligned_folder": CROPPED_ALIGNED_FOLDER_DS[ds],
        # Corresponding to the exact same folder from ABAW authors.
        "video_size": video_size,  # These are for face detection and cropping.
        # Not used since we already have it from ABAW authors.
        "margin_size": margin_size,
        "landmark": {
            "landmark_number": 68,
            "output_image_size": video_size,
            "margin_size": margin_size
        },

        "extract_cnn": 0,  # Not used. Directly feed CNN instead of video
        # to the model can greatly boost efficiency, but sacrifice
        # effectiveness.
        "cnn_folder": "cnn",
        "cnn": {
            "model": "ires100_75",
            "state_dict": "model_state_dict.pth",
            "batch_size": 32,
            "image_size": 48,
            "crop_size": 40,
            "mean": [0.5, 0.5, 0.5],
            "std": [0.5, 0.5, 0.5]
        },

        # Corresponding to the three sub-tracks of ABAW contest.
        # name of the columns in the csv file.
        "AU_label_cols": ["AU1", "AU2", "AU4", "AU6", "AU7", "AU10", "AU12",
                          "AU15", "AU23", "AU24", "AU25", "AU26"],
        "VA_label_cols": ["valence", "arousal"],  # This is what we used.
        "EXPR_label_cols": ["Neutral"],

        # Generate txt labels to npy format.
        "extract_continuous_label": 1,

        # Not used, because we directly generate 16K hz wav from raw videos.
        "change_audio_frequency": 0,
        "load_freq_changed_audio": 0,
        "freq_changed_audio_folder": "freq_changed_audio",
        "target_frequency": 16000,

        # This is used to convert mp4 to 16K mono wav.
        "convert_to_wav": 1,
        "wav_folder": "wav",

        "extract_mfcc": 0,  # Extract MFCC.
        "mfcc_folder": "mfcc",
        "mfcc_config_file": "opensmile_mfcc.conf",

        "extract_egemaps": 0,  # This one can be really slow to extract.
        "egemaps_folder": "egemaps",

        "extract_logmel": 0,  # Not used, because it's too large
        "logmel_folder": "logmel",

        "extract_vggish": 1,  # Used to extract vggish
        "vggish_folder": "vggish",

        "extract_transcript": 0,  # Use Vosk to transcribe_audio the video with
        # timestamp.
        "transcript_folder": "transcript",

        "add_punctuation": 0,  # Then add punctuation on the extracted
        # transcripts.
        "punctuation_folder": "punctuation",

        "extract_word_embedding": 1,  # Then extract bert
        "word_embedding_folder": "word_embedding",

        "align_word_embedding": 1,  # Finally populate the bert feature
        # according to the timestamps
        "aligned_word_embedding_folder": "aligned_word_embedding",

        "save_npy": 1,  # Save all extracted feature to npy

        "emotion_dimension": {
            "AU": 12,
            "EXPR": constants.NUM_CLASSES[ds],
            "VA_V": 1,
            "VA_A": 1
        },

        # The root path of your dataset
        # It should include three sub folders coming from ABAW authors:
        # `5th_ABAW_Annotations`, `cropped_aligned`, `raw_video`.
        # Inside `5th_ABAW_Annotations`, there should be three folders, each
        # contains the labels for a sub-track.
        # For the Valence-arousal (VA) one, I don't remember if it was named VA
        # by the ABAW authors. If not, name it as VA by yourself. My code will
        # read it by `VA`.
        # Inside folder `VA`, rename the two sub-folders to 'train' and
        # 'validate'. My code will load them, do things, and generate the
        # partition information accordingly.

        "root_directory": join(get_home_root_data(), ds),

        # Set to the same as root_directory so that the intermediate data will
        # also be saved in here.
        "output_root_directory": join(get_home_root_data(), ds, 'features'),

        # Put all the model state dict here, such as Vosk, and opensmile config
        # file.
        "load_directory": join(root_dir, "load"),

        # The VA task.
        "task": task,

        # Raw videos from the ABAW authors.
        "raw_data_folder": "raw_videos",

        # All the npys go here.
        "npy_folder": "compacted_48",

        "frequency": {
            "video": None,
            "continuous_label": None,
            "mfcc": None,
            "egemaps": None,
            "vggish": None,
            "logmel": None,
            "bert": None,
        },

        "multiplier": {
            "video": 1,
            "cnn": 1,
            # "AU_continuous_label": 1,
            "EXPR_continuous_label": 1,
            # "VA_continuous_label": 1,
            "continuous_label": 1,
            "mfcc": 1,
            "egemaps": 1,
            "vggish": 1,
            "logmel": 1,
            "bert": 1,
        },

        "feature_dimension": {
            "video": (video_size, video_size, 3),
            "cnn": (512,),
            "AU_continuous_label": (12,),
            "EXPR_continuous_label": (1,),
            "VA_continuous_label": (1,),
            "continuous_label": (1,),
            "mfcc": (39,),
            "egemaps": (88,),
            "vggish": (128,),
            "logmel": (96, 64),
            "bert": (768,)
        },

        "speech_model": {
            # Download it from https://alphacephei.com/vosk/models/
            # vosk-model-en-us-0.22.zip
            # Then put it in the  folder "load"
            "path": join(root_dir, "load/vosk-model-en-us-0.22")
        },
    }

    return config
