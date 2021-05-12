import torch
import math
import numpy as np
import matplotlib.pyplot as plt


def colorize(value, vmin=10, vmax=1000, cmap='plasma'):
    # normalize
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    if vmin!=vmax:
        value = (value - vmin) / (vmax - vmin) # vmin..vmax
    else:
        # Avoid 0-division
        value = value*0.
    # squeeze last dim if it exists
    #value = value.squeeze(axis=0)

    cmapper = plt.cm.get_cmap(cmap)
    value = cmapper(value) # (nxmx4)

    img = value[:,:,:3]
    return img

def colored_depthmap(depth, d_min=None, d_max=None):
    if d_min is None:
        d_min = np.min(depth)
    if d_max is None:
        d_max = np.max(depth)
    depth_relative = (depth - d_min) / (d_max - d_min)
    return 255 * plt.cm.jet(depth_relative)[:, :, :3]  # H, W, C

def colored_batch_depthmap(batch, d_min = None, d_max=None):
    batch = batch.cpu().numpy()
    colored_batch = []
    for i in range(batch.shape[0]):
        depth = batch[i][0]
        colored_batch.append(colorize(depth, vmin=d_min, vmax=d_max))
    colored_batch = torch.stack([torch.tensor(colored) for colored in colored_batch], axis=0)
    return colored_batch.permute(0, 3, 1, 2)

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