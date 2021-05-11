import torch
import math
import numpy as np
import matplotlib.pyplot as plt

def colored_depthmap(depth, d_min=None, d_max=None):
    if d_min is None:
        d_min = np.min(depth)
    if d_max is None:
        d_max = np.max(depth)
    depth_relative = (depth - d_min) / (d_max - d_min)
    return 255 * plt.cm.jet(depth_relative)[:, :, :3]  # H, W, C


def log10(x):
    """Convert a new tensor with the base-10 logarithm of the elements of x. """
    return torch.log(x) / math.log(10)

def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    valid_mask = gt > 0
    pred = pred[valid_mask]
    gt = gt[valid_mask]

    thresh = torch.max((gt / pred), (pred / gt))
    d1 = float((thresh < 1.25).float().mean())
    d2 = float((thresh < 1.25 ** 2).float().mean())
    d3 = float((thresh < 1.25 ** 3).float().mean())

    rmse = (gt - pred) ** 2
    rmse = math.sqrt(rmse.mean())

    rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
    rmse_log = math.sqrt(rmse_log.mean())

    abs_rel = ((gt - pred).abs() / gt).mean()
    sq_rel = (((gt - pred) ** 2) / gt).mean()

    return abs_rel, sq_rel, rmse, rmse_log, d1, d2, d3