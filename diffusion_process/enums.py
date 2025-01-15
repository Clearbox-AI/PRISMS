from enum import Enum

class DatasetType(Enum):
    IMAGE_TABULAR = "image_tabular"
    NACC_LATENTS = "nacc_latents"
    TOY_MNIST = "toy_mnist"
    EXP_LUMIR = "exp_lumir"
    LDM_ONE_H = "ldm_one_h"