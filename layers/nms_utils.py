import torch
import torch.nn.functional as F
from ..box_utils import jaccard

import numpy as np


def cc_fast_nms(boxes, masks, scores, iou_threshold: float = 0.5, top_k: int = 200):
    # Collapse all the classes into 1 
    scores, classes = scores.max(dim=0)

    _, idx = scores.sort(0, descending=True)
    idx = idx[:top_k]

    boxes_idx = boxes[idx]

    # Compute the pairwise IoU between the boxes
    iou = jaccard(boxes_idx, boxes_idx)
    
    # Zero out the lower triangle of the cosine similarity matrix and diagonal
    iou.triu_(diagonal=1)

    # Now that everything in the diagonal and below is zeroed out, if we take the max
    # of the IoU matrix along the columns, each column will represent the maximum IoU
    # between this element and every element with a higher score than this element.
    iou_max, _ = torch.max(iou, dim=0)

    # Now just filter out the ones greater than the threshold, i.e., only keep boxes that
    # don't have a higher scoring box that would supress it in normal NMS.
    idx_out = idx[iou_max <= iou_threshold]
    
    return boxes[idx_out], masks[idx_out], classes[idx_out], scores[idx_out]

def fast_nms(boxes, masks, scores, iou_threshold: float = 0.5, top_k: int = 200,
             conf_thresh: float = 0.05, max_num_detections: int = 200):
    scores, idx = scores.sort(1, descending=True)

    idx = idx[:, :top_k].contiguous()
    scores = scores[:, :top_k]

    num_classes, num_dets = idx.size()

    boxes = boxes[idx.view(-1), :].view(num_classes, num_dets, 4)
    masks = masks[idx.view(-1), :].view(num_classes, num_dets, -1)

    iou = jaccard(boxes, boxes)
    iou.triu_(diagonal=1)
    iou_max, _ = iou.max(dim=1)

    # Now just filter out the ones higher than the threshold
    keep = (iou_max <= iou_threshold)

    # We should also only keep detections over the confidence threshold, but at the cost of
    # maxing out your detection count for every image, you can just not do that. Because we
    # have such a minimal amount of computation per detection (matrix mulitplication only),
    # this increase doesn't affect us much (+0.2 mAP for 34 -> 33 fps), so we leave it out.
    # However, when you implement this in your method, you should do this second threshold.
    if conf_thresh > 0:
        keep *= (scores > conf_thresh)

    # Assign each kept detection to its corresponding class
    classes = torch.arange(num_classes, device=boxes.device)[:, None].expand_as(keep)
    classes = classes[keep]

    boxes = boxes[keep]
    masks = masks[keep]
    scores = scores[keep]
    
    # Only keep the top max_num_detections highest scores across all classes
    scores, idx = scores.sort(0, descending=True)
    idx = idx[:max_num_detections]
    scores = scores[:max_num_detections]

    classes = classes[idx]
    boxes = boxes[idx]
    masks = masks[idx]

    return boxes, masks, classes, scores

def traditional_nms(boxes, masks, scores, iou_threshold: float = 0.5, top_k: int = 200,
                    conf_thresh: float = 0.05, max_num_detections: int = 200, max_size: int = 400):
    import pyximport
    pyximport.install(setup_args={"include_dirs":np.get_include()}, reload_support=True)

    from utils.cython_nms import nms as cnms

    num_classes = scores.size(0)

    idx_lst = []
    cls_lst = []
    scr_lst = []

    # Multiplying by max_size is necessary because of how cnms computes its area and intersections
    boxes = boxes * max_size

    for _cls in range(num_classes):
        cls_scores = scores[_cls, :]
        conf_mask = cls_scores > conf_thresh
        idx = torch.arange(cls_scores.size(0), device=boxes.device)

        cls_scores = cls_scores[conf_mask]
        idx = idx[conf_mask]

        if cls_scores.size(0) == 0:
            continue
        
        preds = torch.cat([boxes[conf_mask], cls_scores[:, None]], dim=1).cpu().numpy()
        keep = cnms(preds, iou_threshold)
        keep = torch.Tensor(keep, device=boxes.device).long()

        idx_lst.append(idx[keep])
        cls_lst.append(keep * 0 + _cls)
        scr_lst.append(cls_scores[keep])
    
    idx     = torch.cat(idx_lst, dim=0)
    classes = torch.cat(cls_lst, dim=0)
    scores  = torch.cat(scr_lst, dim=0)

    scores, idx2 = scores.sort(0, descending=True)
    idx2 = idx2[:max_num_detections]
    scores = scores[:max_num_detections]

    idx = idx[idx2]
    classes = classes[idx2]

    # Undo the multiplication above
    return boxes[idx] / max_size, masks[idx], classes, scores
