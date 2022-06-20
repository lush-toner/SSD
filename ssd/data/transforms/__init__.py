from ssd.modeling.anchors.prior_box import PriorBox
from .target_transform import SSDTargetTransform
from .transforms import *


def build_transforms(cfg, is_train=True):
    if is_train:
        transform = [
            ConvertFromInts(), # image : int -> float
            # PhotometricDistort(), # several random augmentation, 
            # Expand(cfg.INPUT.PIXEL_MEAN), # enlarge data and box
            # RandomSampleCrop(),
            # RandomMirror(),
            ToPercentCoords(),
            Resize(cfg.INPUT.IMAGE_SIZE),
            SubtractMeans(cfg.INPUT.PIXEL_MEAN),
            ToTensor(),
        ]
    else:
        transform = [
            Resize(cfg.INPUT.IMAGE_SIZE),
            SubtractMeans(cfg.INPUT.PIXEL_MEAN),
            ToTensor()
        ]
    transform = Compose(transform) # Composes several augmentations #
    return transform


def build_target_transform(cfg):
    transform = SSDTargetTransform(PriorBox(cfg)(),
                                   cfg.MODEL.CENTER_VARIANCE, # 0.1
                                   cfg.MODEL.SIZE_VARIANCE, # 0.2
                                   cfg.MODEL.THRESHOLD) # 0.5
    return transform
