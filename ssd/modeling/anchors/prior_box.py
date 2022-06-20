from itertools import product

import torch
from math import sqrt


class PriorBox:
    def __init__(self, cfg):
        self.image_size = cfg.INPUT.IMAGE_SIZE
        prior_config = cfg.MODEL.PRIORS
        self.feature_maps = prior_config.FEATURE_MAPS # [38, 19, 10, 5, 3, 1] -> multi-scale feature map size
        self.min_sizes = prior_config.MIN_SIZES # [30, 60, 111, 162, 213, 264]
        self.max_sizes = prior_config.MAX_SIZES # [60, 111, 162, 213, 264, 315]
        self.strides = prior_config.STRIDES # [8, 16, 32, 64, 100, 300]
        self.aspect_ratios = prior_config.ASPECT_RATIOS # [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
        self.clip = prior_config.CLIP # True

    def __call__(self):
        """Generate SSD Prior Boxes.
            It returns the center, height and width of the priors. The values are relative to the image size
            Returns:
                priors (num_priors, 4): The prior boxes represented as [[center_x, center_y, w, h]]. All the values
                    are relative to the image size.
        """
        priors = []
        for k, maps in enumerate(self.feature_maps):
            scale = self.image_size / self.strides[k] # how much decrease compare to original size
            
            # product -> combination, when product(range(2), repeat=2) -> 0 0, 0 1, 1 0, 1 1 
            for i, j in product(range(maps), repeat=2): 
                # unit center x,y , center coordinate, 0~1 normalize
                cx = (j + 0.5) / scale 
                cy = (i + 0.5) / scale

                """ SAVE BOX SHAPE IN EACH COORDINATE"""

                # save coordinate

                # SQUARE BOX : SMALL SIZE
                # min_sizes : [30, 60, 111, 162, 213, 264]
                h = w = self.min_sizes[k] / self.image_size # original to box ratio [0.1 0.2 0.37 0.54 0.71 0.88]
                priors.append([cx, cy, w, h])

                # SQUARE BOX : BIG SIZE
                # [60, 111, 162, 213, 264, 315]
                h = w = sqrt(self.min_sizes[k] * self.max_sizes[k]) / self.image_size # original to box ratio [0.2 0.37 0.54 0.71 0.88, 1.05]
                priors.append([cx, cy, w, h])

                # change h/w ratio of the small sized box
                h = w = self.min_sizes[k] / self.image_size # original to box ratio [0.1 0.2 0.37 0.54 0.71 0.88]
                for ratio in self.aspect_ratios[k]: # aspect ratio [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
                    ratio = sqrt(ratio)
                    priors.append([cx, cy, w * ratio, h / ratio]) # increase width, decrease height
                    priors.append([cx, cy, w / ratio, h * ratio]) # decrease width, increase height

        priors = torch.tensor(priors)
        if self.clip:
            priors.clamp_(max=1, min=0)
        return priors
