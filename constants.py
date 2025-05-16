# Datasets
MELD = 'MELD'
C_EXPR_DB = 'C-EXPR-DB'  # test set of AWAW 7th, compound
# emotion, annotated by our team and split into 125 videos.
C_EXPR_DB_CHALLENGE = 'C-EXPR-DB-CHALLENGE'  # test set of AWAW 7th, compound
# emotion
AH_DB = 'AH'  # ambivalence/hesitancy challenge dataset.
BAH_DB = 'BAH_DB'  # ambivalence/hesitancy dataset: neurips

DATASETS = [MELD, C_EXPR_DB, C_EXPR_DB_CHALLENGE, AH_DB, BAH_DB]

NUM_CLASSES = {
    MELD: 7,
    C_EXPR_DB: 7,
    C_EXPR_DB_CHALLENGE: 7,
    AH_DB: 2,
    BAH_DB: 2
}

# Task
CLASSFICATION = 'CLASSIFICATION'
REGRESSION = 'REGRESSION'

TASKS = [CLASSFICATION, REGRESSION]

DS_TASK = {
    MELD: CLASSFICATION,
    C_EXPR_DB: CLASSFICATION,
    C_EXPR_DB_CHALLENGE: CLASSFICATION,
    AH_DB: CLASSFICATION,
    BAH_DB: CLASSFICATION
}

# Fusion methods:
LFAN = 'LFAN'  # https://arxiv.org/pdf/2203.13031
CAN = 'CAN'  # https://arxiv.org/pdf/2203.13031
JMT = 'JMT'  # https://arxiv.org/pdf/2403.10488
MT = 'MT'  # https://arxiv.org/pdf/2403.10488

FUSION_METHODS = [LFAN, CAN, JMT, MT]

# optimizers
SGD = 'SGD'
ADAM = 'ADAM'

OPTIMIZERS = [SGD, ADAM]


# LR scheduler
STEP = 'STEP'
MULTISTEP = 'MULTISTEP'
MYSTEP = 'MYSTEP'
MYWARMUP = 'MYWARMUP'
COSINE = 'COSINE'
MYCOSINE = 'MYCOSINE'

LR_SCHEDULERS = [STEP, MULTISTEP, MYSTEP, MYWARMUP, COSINE, MYCOSINE]

# Mode for lr scheduler
MAX_MODE = 'MAX'
MIN_MODE = 'MIN'

LR_MODES = [MAX_MODE, MIN_MODE]

# Modes
TRAINING = "TRAINING"
EVALUATION = 'EVALUATION'

MODES = [TRAINING, EVALUATION]


CROP_SIZE = 224
RESIZE_SIZE = 256

SZ224 = 224
SZ256 = 256
SZ112 = 112


# EXPRESSIONS
# basic
SURPRISE = 'Surprise'
FEAR = 'Fear'
DISGUST = 'Disgust'
HAPPINESS = 'Happiness'
SADNESS = 'Sadness'
ANGER = 'Anger'
NEUTRAL = 'Neutral'

# compound
FEARFULLY_SURPRISED = "Fearfully Surprised"
HAPPILY_SURPRISED = "Happily Surprised"
SADLY_SURPRISED = "Sadly Surprised"
DISGUSTEDLY_SURPRISED = "Disgustedly Surprised"
ANGRILY_SURPRISED = "Angrily Surprised"
SADLY_FEARFUL = "Sadly Fearful"
SADLY_ANGRY = "Sadly Angry"
OTHER = "Other"

# Ambivalence/ Hesitancy
W_AH = 'With A-H'
N_AH = 'No A-H'


EXPRESSIONS = [SURPRISE, FEAR, DISGUST, SADNESS, HAPPINESS, ANGER, NEUTRAL,
               FEARFULLY_SURPRISED,
               HAPPILY_SURPRISED,
               SADLY_SURPRISED,
               DISGUSTEDLY_SURPRISED,
               ANGRILY_SURPRISED,
               SADLY_FEARFUL,
               SADLY_ANGRY,
               OTHER,
               W_AH,
               N_AH
               ]

# dataset splits:
TRAINSET = 'train'
VALIDSET = 'val'
TESTSET = 'test'

SPLITS = [TRAINSET, VALIDSET, TESTSET]

# Data modalities
VGGISH = 'vggish'  # audio: vggish features
VIDEO = 'video'  # video: row frames
BERT = 'bert'  # text: BERT features
EXPR = 'EXPR_continuous_label'

MODALITIES = [VGGISH, VIDEO, BERT, EXPR]

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

# METRICS = [MACRO_F1, W_F1, CL_ACC, CFUSE_MARIX, F1_POS, F1_NEG, AP_POS]
METRICS = [MACRO_F1, W_F1, CL_ACC, CFUSE_MARIX]

# levels
FRAME_LEVEL = 'FRAME_LEVEL'  # frame level
VIDEO_LEVEL = 'VIDEO_LEVEL'  # video level

EVAL_LEVELS = [FRAME_LEVEL, VIDEO_LEVEL]

# how to predict at video level from frame prediction
FRM_VOTE = 'FRAMES_VOTE'  # video label = majority voting at frame level
FRM_AVG_PROBS = 'FRAMES_AVG_PROBS'  # compute probs of each frame,
# then average them, then predict for video.
FRM_AVG_LOGITS = 'FRAMES_AVG_LOGITS'  # compute logits of each frame,
# then average them, then predict for video

VIDEO_PREDS = [FRM_VOTE, FRM_AVG_PROBS, FRM_AVG_LOGITS]

# train supervision
TR_SUP_VIDEO_GL = 'video_global_sup'  # video-global level supervision for
# training
TR_SUP_VIDEO_FR = 'video_fr_sup'  # frame level supervision for training
TR_SUP = [TR_SUP_VIDEO_FR, TR_SUP_VIDEO_GL]


# new constants
PRETRAINED_WEIGHTS_DIR = 'pretrained-weights'
FOLDER_PRETRAINED_IMAGENET = 'pretrained-imgnet'  # todo: redundant.
PRETRAINED_BACKBONES = 'pretrained_backbones'  # where backbones reside.

# APVIT attention type.
ATT_LA = 'LA'
ATT_SUM = 'SUM'
ATT_SUM_ABS_1 = 'SUM_ABS_1'
ATT_SUM_ABS_2 = 'SUM_ABS_2'
ATT_MAX = 'MAX'
ATT_MAX_ABS_1 = 'MAX_ABS_1'
ATT_RAND = 'Random'

ATT_MEAN = 'mean_ATT'
ATT_PARAM_ATT = 'PARAM_ATT'  # parametric
ATT_PARAM_G_ATT = 'PARAM_G_ATT'  # parametric (gated attention)

ATT_METHODS = [ATT_LA, ATT_SUM, ATT_SUM_ABS_1, ATT_SUM_ABS_2, ATT_MAX,
               ATT_MAX_ABS_1, ATT_RAND,
               ATT_MEAN,
               ATT_PARAM_ATT, ATT_PARAM_G_ATT]


# ecnoders

APVIT = 'apvit'  # https://arxiv.org/pdf/2212.05463.pdf

#  resnet
RESNET18 = 'resnet18'
RESNET34 = 'resnet34'
RESNET50 = 'resnet50'
RESNET101 = 'resnet101'
RESNET152 = 'resnet152'


STD_CL = "STD_CL"  # standard classification using only the encoder features.
METHOD_APVIT = 'APVIT'
APVITCLASSIFIER = 'APVITClassifier'