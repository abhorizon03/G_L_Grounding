from typing import Any, List, Tuple, Optional
from copy import deepcopy
import torch
from torchvision.ops import box_iou
from sklearn.metrics import f1_score, jaccard_score
# from torcheval.metrics import BinaryBinnedAUROC
from torcheval.metrics.functional import binary_binned_auroc
from monai.transforms import KeepLargestConnectedComponent

from .utils import nanvar, largest_connected_component, get_bb_from_largest_component


lcc = KeepLargestConnectedComponent()


def CNR(sim_map: torch.Tensor, bboxes: List[Tuple], non_absolute: bool = False):
    '''
    Contrast-to-Noise Ratio per given bounding box
    '''
    sim_map = deepcopy(sim_map)
    cnrs = []
    
    for bbox in bboxes:
        x, y, width, height = bbox
        area_in = sim_map[y:y+height, x:x+width]
        mu_in, var_in = torch.nanmean(area_in), nanvar(area_in)
        sim_map[y:y+height, x:x+width] = float("NaN")
        mu_out, var_out = torch.nanmean(sim_map), nanvar(sim_map)
        if non_absolute:
            means_diff = mu_in - mu_out
        else:
            means_diff = torch.abs(mu_in - mu_out)
        cnr = means_diff / torch.sqrt(var_in + var_out)
        cnrs.append(torch.nan_to_num(cnr).item())
    
    return sum(cnrs) / len(cnrs)


def mIoU_old(sim_map: torch.Tensor, gt_boxes: List[List], thresholds: List = [.1, .2, .3, .4, .5]):
    # Old version of mIoU --> extracts a bbox via largest connected component

    iou_per_bb = []
    for gt_box in gt_boxes:
        gt_box = torch.tensor(gt_box)[None, :]
        gt_box[:, 2] += gt_box[:, 0]
        gt_box[:, 3] += gt_box[:, 1]
    
        ious = torch.empty((len(thresholds),))
        for i, threshold in enumerate(thresholds):
            # get predicted bounding box from similarity map
            sim_map_bin = sim_map > threshold
            largest_comp = lcc(sim_map_bin[None, ...]).squeeze(0)
            try:
                y_inds, x_inds = torch.nonzero(largest_comp, as_tuple=True)
                pred_box = [x_inds[0].item(), y_inds[0].item(), x_inds[-1].item(), y_inds[-1].item()]
            except IndexError:
                ious[i] = 0
                continue

            # previous implementation
            # largest_comp = largest_connected_component(sim_map_bin)
            # pred_box = get_bb_from_largest_component(largest_comp)

            pred_box = torch.tensor(pred_box)[None, :]
            # pred_box[:, 2] += pred_box[:, 0]
            # pred_box[:, 3] += pred_box[:, 1]
            # IoU @ threshold
            ious[i] = box_iou(pred_box, gt_box).squeeze().item()
        
        iou_per_bb.append(ious.mean().item())

    return sum(iou_per_bb) / len(iou_per_bb)


def mIoU(sim_map: torch.Tensor, gt_boxes: List[List], thresholds: List = [.1, .2, .3, .4, .5]):
    # iou_per_bb = []
    trg = torch.zeros_like(sim_map, dtype=torch.bool)
    for gt_box in gt_boxes:
        x, y, w, h = gt_box
        trg[y:y+h, x:x+w] = True
    
    ious = torch.empty((len(thresholds),))
    for i, threshold in enumerate(thresholds):
        bin_mask = sim_map > threshold
        intersection = torch.logical_and(bin_mask, trg)
        union = torch.logical_or(bin_mask, trg)
        # IoU @ threshold
        ious[i] = intersection.sum().item() / union.sum().item()
    
    # iou_per_bb.append(ious.mean().item())
    # return sum(iou_per_bb) / len(iou_per_bb)
    return ious.mean().item()


def mIoU_scikit(sim_map: torch.Tensor, gt_boxes: List[List], thresholds: List = [.1, .2, .3, .4, .5]):
    iou_per_bb = []
    for bb in gt_boxes:
        x, y, w, h = bb
        trg = torch.zeros_like(sim_map, dtype=torch.bool)
        trg[y:y+h, x:x+w] = True
    
        ious = torch.empty((len(thresholds),))
        for i, threshold in enumerate(thresholds):
            bin_mask = sim_map > threshold
            ious[i] = jaccard_score(trg.flatten().numpy(), bin_mask.flatten().numpy())
        
        iou_per_bb.append(ious.mean().item())

    return sum(iou_per_bb) / len(iou_per_bb)


class AUC_ROC:
    def __init__(self):
        # self.metric = BinaryBinnedAUROC(threshold=5)
        pass
    
    def __call__(self, sim_map: torch.Tensor, gt_boxes: List[List]):
        trg = torch.zeros_like(sim_map, dtype=torch.bool)
        for bb in gt_boxes:
            x, y, w, h = bb
            trg[y:y+h, x:x+w] = True
        
        # self.metric.update(sim_map.flatten(), trg.long().flatten())
        # return self.metric.compute()[0].item()
        return binary_binned_auroc(sim_map.flatten(), trg.long().flatten(), threshold=5)[0].item()
    
def _make_binned_thresholds(num_bins: int,
                            low: float = 0.0,
                            high: float = 1.0) -> torch.Tensor:
    num_bins = max(1, int(num_bins))
    return torch.linspace(low, high, steps=num_bins, dtype=torch.float32)


def _build_masks(sim_map: torch.Tensor,
                 gt_boxes: List[List[int]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    H, W = sim_map.shape[-2:]
    valid = torch.isfinite(sim_map)

    pos = torch.zeros_like(valid, dtype=torch.bool)
    for x, y, w, h in gt_boxes:
        # 边界裁剪，防越界
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(W, x + w), min(H, y + h)
        if x0 < x1 and y0 < y1:
            pos[y0:y1, x0:x1] = True

    # 仅在有效像素内统计
    pos = pos & valid
    neg = (~pos) & valid
    return valid, pos, neg


def Precision(
    sim_map: torch.Tensor,
    gt_boxes: List[List[int]],
    thresholds: Optional[List[float]] = None,
    num_bins: int = 5,
) -> float:
    thr = _make_binned_thresholds(num_bins) if thresholds is None else (
        torch.tensor(thresholds, dtype=sim_map.dtype, device=sim_map.device)
        if isinstance(thresholds, list) else thresholds
    )

    valid, pos, neg = _build_masks(sim_map, gt_boxes)

    vals = torch.empty(len(thr), dtype=torch.float32)
    for i, t in enumerate(thr):
        T = (sim_map > t) & valid             # 只在有效像素上产生预测
        TP = int((T & pos).sum().item())      # (热力图>thr ∧ 框内)
        FP = int((T & neg).sum().item())      # (热力图>thr ∧ 框外)
        denom = TP + FP
        vals[i] = 0.0 if denom == 0 else float(TP) / float(denom)

    return vals.mean().item()


def Recall(
    sim_map: torch.Tensor,
    gt_boxes: List[List[int]],
    thresholds: Optional[List[float]] = None,
    num_bins: int = 5,
) -> float:
    thr = _make_binned_thresholds(num_bins) if thresholds is None else (
        torch.tensor(thresholds, dtype=sim_map.dtype, device=sim_map.device)
        if isinstance(thresholds, list) else thresholds
    )

    valid, pos, _ = _build_masks(sim_map, gt_boxes)

    vals = torch.empty(len(thr), dtype=torch.float32)
    for i, t in enumerate(thr):
        T = (sim_map > t) & valid             # 只在有效像素上产生预测
        TP = int((T & pos).sum().item())      # (热力图>thr ∧ 框内)
        FN = int(((~T) & pos).sum().item())   # (热力图<=thr ∧ 框内)
        denom = TP + FN
        vals[i] = 0.0 if denom == 0 else float(TP) / float(denom)

    return vals.mean().item()

def Dice(
    sim_map: torch.Tensor,
    gt_boxes: List[List[int]],
    thresholds: Optional[List[float]] = [0.1, 0.2, 0.3, 0.4, 0.5],
    num_bins: int = 5,
) -> float:
    thr = _make_binned_thresholds(num_bins) if thresholds is None else (
        torch.tensor(thresholds, dtype=sim_map.dtype, device=sim_map.device)
        if isinstance(thresholds, list) else thresholds
    )

    valid, pos, neg = _build_masks(sim_map, gt_boxes)

    vals = torch.empty(len(thr), dtype=torch.float32)
    for i, t in enumerate(thr):
        T = (sim_map > t) & valid
        TP = int((T & pos).sum().item())
        FP = int((T & neg).sum().item())
        FN = int((~T & pos).sum().item())
        denom = 2 * TP + FP + FN
        vals[i] = 0.0 if denom == 0 else (2.0 * TP) / float(denom)

    return vals.mean().item()
