import torch
import math


def convert_locations_to_boxes(locations, priors, center_variance,
                               size_variance):
    """Convert regressional location results of SSD into boxes in the form of (center_x, center_y, h, w).

    The conversion:
        $$predicted\_center * center_variance = \frac {real\_center - prior\_center} {prior\_hw}$$
        $$exp(predicted\_hw * size_variance) = \frac {real\_hw} {prior\_hw}$$
    We do it in the inverse direction here.
    Args:
        locations (batch_size, num_priors, 4): the regression output of SSD. It will contain the outputs as well.
        priors (num_priors, 4) or (batch_size/1, num_priors, 4): prior boxes.
        center_variance: a float used to change the scale of center.
        size_variance: a float used to change of scale of size.
    Returns:
        boxes:  priors: [[center_x, center_y, w, h]]. All the values
            are relative to the image size.
    """
    # priors can have one dimension less.
    if priors.dim() + 1 == locations.dim():
        priors = priors.unsqueeze(0)
    return torch.cat([
        locations[..., :2] * center_variance * priors[..., 2:] + priors[..., :2],
        torch.exp(locations[..., 2:] * size_variance) * priors[..., 2:]
    ], dim=locations.dim() - 1)


def convert_boxes_to_locations(center_form_boxes, center_form_priors, center_variance, size_variance):
    # priors can have one dimension less
    if center_form_priors.dim() + 1 == center_form_boxes.dim():
        center_form_priors = center_form_priors.unsqueeze(0)
    return torch.cat([
        (center_form_boxes[..., :2] - center_form_priors[..., :2]) / center_form_priors[..., 2:] / center_variance,
        torch.log(center_form_boxes[..., 2:] / center_form_priors[..., 2:]) / size_variance
    ], dim=center_form_boxes.dim() - 1)


def area_of(left_top, right_bottom) -> torch.Tensor:
    """Compute the areas of rectangles given two corners.

    Args:
        left_top (N, 2): left top corner.
        right_bottom (N, 2): right bottom corner.

    Returns:
        area (N): return the area.
    """
    hw = torch.clamp(right_bottom - left_top, min=0.0) # right - left -> width / bottom - top -> height
    return hw[..., 0] * hw[..., 1]


def iou_of(gt_boxes, prior_boxes, eps=1e-5):
    """Return intersection-over-union (Jaccard index) of boxes.

    Args:
        gt_boxes (N, 4): ground truth boxes.
        prior_boxes (N or 1, 4): predicted boxes.
        eps: a small number to avoid 0 as denominator.
    Returns:
        iou (N): IoU values.
    """
    # gt_boxes.shape ([1, num GT boxes, 4]) / prior_boxes.shape ([8732, 1, 4])
    # left top max -> overlap x1, y1
    # right bottom min -> overlap x2, y2
    """Calculate overlap area"""
    overlap_left_top     = torch.max(gt_boxes[..., :2], prior_boxes[..., :2]) # x1(xmin), y1(ymin) torch.Size([8732, 1, 4])
    overlap_right_bottom = torch.min(gt_boxes[..., 2:], prior_boxes[..., 2:]) # x2(xmax), y2(ymax) torch.Size([1, boxes, 4])
    overlap_area = area_of(overlap_left_top, overlap_right_bottom) # [8732, num GT boxes]

    """Calculate GT box and prior box"""
    gt_area = area_of(gt_boxes[..., :2], gt_boxes[..., 2:]) # x1, y1 & x2, y2
    prior_area = area_of(prior_boxes[..., :2], prior_boxes[..., 2:]) # x1, y1 & x2, y2
    return overlap_area / (gt_area + prior_area - overlap_area + eps)


def assign_priors(gt_boxes, gt_labels, corner_form_priors,
                  iou_threshold):
    """Assign ground truth boxes and targets to priors.

    Args:
        gt_boxes (num_targets, 4): ground truth boxes.
        gt_labels (num_targets): labels of targets.
        priors (num_priors, 4): corner form priors
    Returns:
        boxes (num_priors, 4): real values for priors.
        labels (num_priros): labels for priors.
    """
    # size: num_priors x num_targets
    # torch.Size([boxes, 4]) -> torch.Size([1, boxes, 4]) / torch.Size([8732, 4]) -> torch.Size([8732, 1, 4])
    ious = iou_of(gt_boxes.unsqueeze(0), corner_form_priors.unsqueeze(1)) # ([8732, boxes])
    # size: num_priors -> just pick 1 iou for gt & prior iou -> [8732] -> compare with row direction(num boxes) -> consider just 1 box for every prior
    best_label_per_prior, best_label_per_prior_index = ious.max(1) # -> get best iou from dim 1 - element
    # size: num_targets ->pick 1 iou of overall gt & prior iou->[boxes]-> compare with column direction(num priors) -> consider all GT num boxes for all prior
    best_prior_per_label, best_prior_per_label_index = ious.max(0) # -> get best iou from dim 0 - all dim
    """2022-06-07- TO assign BEST PRIOR BOX THAT HAS MAX IOU EACH GT """
    for target_index, prior_index in enumerate(best_prior_per_label_index):
        best_label_per_prior_index[prior_index] = target_index
    # 2.0 is used to make sure every target has a prior assigned
    best_label_per_prior.index_fill_(0, best_prior_per_label_index, 2) # change element of "index" in dim "0" to "2"
    # size: num_priors
    labels = gt_labels[best_label_per_prior_index] # assign labels
    labels[best_label_per_prior < iou_threshold] = 0  # the backgournd id
    boxes = gt_boxes[best_label_per_prior_index]
    return boxes, labels


def hard_negative_mining(loss, labels, neg_pos_ratio):
    """
    It used to suppress the presence of a large number of negative prediction.
    It works on image level not batch level.
    For any example/image, it keeps all the positive predictions and
     cut the number of negative predictions to make sure the ratio
     between the negative examples and positive examples is no more
     the given ratio for an image.

    Args:
        loss (N, num_priors): the loss for each example.
        labels (N, num_priors): the labels.
        neg_pos_ratio:  the ratio between the negative examples and positive examples.
    """
    pos_mask = labels > 0
    num_pos = pos_mask.long().sum(dim=1, keepdim=True)
    num_neg = num_pos * neg_pos_ratio

    loss[pos_mask] = -math.inf
    _, indexes = loss.sort(dim=1, descending=True)
    _, orders = indexes.sort(dim=1)
    neg_mask = orders < num_neg
    return pos_mask | neg_mask


def center_form_to_corner_form(locations):
    return torch.cat([locations[..., :2] - locations[..., 2:] / 2,
                      locations[..., :2] + locations[..., 2:] / 2], locations.dim() - 1)


def corner_form_to_center_form(boxes):
    return torch.cat([
        (boxes[..., :2] + boxes[..., 2:]) / 2, # xmin, ymin + xmax, ymax
        boxes[..., 2:] - boxes[..., :2]
    ], boxes.dim() - 1)
