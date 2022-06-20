from .ssd_detector import SSDDetector

_DETECTION_META_ARCHITECTURES = {
    "SSDDetector": SSDDetector
}

def build_detection_model(cfg):
    meta_arch = _DETECTION_META_ARCHITECTURES[cfg.MODEL.META_ARCHITECTURE]
    return meta_arch(cfg)



"""
.....
MODEL:
  BACKBONE:
    NAME: vgg
    OUT_CHANNELS: (512, 1024, 512, 256, 256, 256)
    PRETRAINED: True
  BOX_HEAD:
    NAME: SSDBoxHead
    PREDICTOR: SSDBoxPredictor
  CENTER_VARIANCE: 0.1
  DEVICE: cuda
  META_ARCHITECTURE: SSDDetector
  NEG_POS_RATIO: 3
  NUM_CLASSES: 21
  PRIORS:
  .....
"""