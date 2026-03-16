import numpy as np
from scipy.spatial import cKDTree


def guard_sqrt(tensor):
    """
    Guarded square root to avoid negative values under the sqrt.
    """
    import torch
    return torch.sqrt(torch.clamp(tensor, min=1e-12))


def chamfer_distance_single_shape_kdtree(pred, gt, one_side=False, sqrt=False):
    """
    Computes average Chamfer distance between prediction and ground truth
    using KDTree for nearest-neighbour search.

    Args:
        pred (np.ndarray): Predicted point cloud, shape (N, 3).
        gt   (np.ndarray): Ground-truth point cloud, shape (M, 3).
        one_side (bool): If True, returns only gt→pred distances.
        sqrt     (bool): If True, use L2 distance instead of squared L2.

    Returns:
        np.ndarray: Per-point distances.
    """
    pred = np.asarray(pred)
    gt   = np.asarray(gt)

    tree_pred = cKDTree(pred)
    tree_gt   = cKDTree(gt)

    dist_pred_to_gt, _ = tree_gt.query(pred)
    if not sqrt:
        dist_pred_to_gt = dist_pred_to_gt ** 2
    if not one_side:
        return dist_pred_to_gt

    dist_gt_to_pred, _ = tree_pred.query(gt)
    if not sqrt:
        dist_gt_to_pred = dist_gt_to_pred ** 2

    return dist_gt_to_pred
