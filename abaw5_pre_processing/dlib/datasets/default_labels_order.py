"""
Default order of emotions
"""
import sys
from os.path import join, dirname, abspath

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

# from abaw5_pre_processing.project.abaw5 import _constants
import constants

ORDERED_EMOTIONS = {

    constants.MELD: {
        constants.NEUTRAL: 0,
        constants.HAPPINESS: 1,
        constants.SURPRISE: 2,
        constants.FEAR: 3,
        constants.ANGER: 4,
        constants.DISGUST: 5,
        constants.SADNESS: 6
    },
    constants.C_EXPR_DB: {
        constants.FEARFULLY_SURPRISED: 0,
        constants.HAPPILY_SURPRISED: 1,
        constants.SADLY_SURPRISED: 2,
        constants.DISGUSTEDLY_SURPRISED: 3,
        constants.ANGRILY_SURPRISED: 4,
        constants.SADLY_FEARFUL: 5,
        constants.SADLY_ANGRY: 6
    },
    constants.C_EXPR_DB_CHALLENGE: {
        constants.FEARFULLY_SURPRISED: 0,
        constants.HAPPILY_SURPRISED: 1,
        constants.SADLY_SURPRISED: 2,
        constants.DISGUSTEDLY_SURPRISED: 3,
        constants.ANGRILY_SURPRISED: 4,
        constants.SADLY_FEARFUL: 5,
        constants.SADLY_ANGRY: 6
    },
    constants.AH_DB: {
        constants.N_AH: 0,
        constants.W_AH: 1
    },
    constants.BAH_DB: {
        constants.N_AH: 0,
        constants.W_AH: 1
    }
}
