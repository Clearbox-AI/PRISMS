from enum import Enum

class DatasetType(Enum):
    NACC = "nacc"
    NACC_LATENTS = "nacc_latents"
    TOY_MNIST = "toy_mnist"
    EXP_LUMIR = "exp_lumir"
    LDM_ONE_H = "ldm_one_h"

class ImageRange(Enum):
    minus1to1 = "minus1to1"
    plus0to1 = "plus0to1"
    none = "none"