import torch
import numpy as np
import trimesh
from .chamfer_distance import ChamferDistance


def compute_chamfer_distance(
    pred: torch.Tensor, gt: torch.Tensor, chamfer_distance_cls: torch.nn.Module = None
):
    """
    计算预测和真实点云之间的Chamfer距离
    
    Args:
        pred: 预测的点云，形状为 (B, N, 3)
        gt: 真实的点云，形状为 (B, M, 3)
        chamfer_distance_cls: Chamfer距离计算类实例
        
    Returns:
        torch.Tensor: 每个批次的Chamfer距离
    """
    if chamfer_distance_cls is None:
        chamfer_distance_cls = ChamferDistance().to(pred.device)
    
    # 计算Chamfer距离
    dist1, dist2 = chamfer_distance_cls(pred, gt)
    
    # 平均距离
    dist1 = dist1.mean(dim=1)
    dist2 = dist2.mean(dim=1)
    
    return dist1 + dist2, dist1, dist2


def compute_fscore(pred, gt, tau=0.1, chunk_size=2048):
    """
    计算预测和真实点云之间的F-score
    
    Args:
        pred: 预测的点云，形状为 (B, N, 3)
        gt: 真实的点云，形状为 (B, M, 3)
        tau: F-score的距离阈值
        chunk_size: 每次处理的点数
        
    Returns:
        torch.Tensor: 每个批次的F-score
    """
    B, N, _ = pred.shape
    _, M, _ = gt.shape
    
    # 初始化张量来存储最小距离
    min_dists_pred_to_gt = torch.zeros(B, N, device=pred.device)
    min_dists_gt_to_pred = torch.zeros(B, M, device=gt.device)
    
    for b in range(B):
        pred_b = pred[b]  # (N, 3)
        gt_b = gt[b]  # (M, 3)
        
        # 分块处理pred到gt
        for i in range(0, N, chunk_size):
            pred_chunk = pred_b[i : i + chunk_size]  # (chunk_size, 3)
            # 计算距离并获取最小值
            dists = torch.cdist(
                pred_chunk.unsqueeze(0), gt_b.unsqueeze(0), p=2
            )  # (1, chunk_size, M)
            min_dists = dists.min(dim=2).values.squeeze(0)  # (chunk_size,)
            min_dists_pred_to_gt[b, i : i + min(chunk_size, N-i)] = min_dists[:min(chunk_size, N-i)]
        
        # 分块处理gt到pred
        for i in range(0, M, chunk_size):
            gt_chunk = gt_b[i : i + chunk_size]  # (chunk_size, 3)
            # 计算距离并获取最小值
            dists = torch.cdist(
                gt_chunk.unsqueeze(0), pred_b.unsqueeze(0), p=2
            )  # (1, chunk_size, N)
            min_dists = dists.min(dim=2).values.squeeze(0)  # (chunk_size,)
            min_dists_gt_to_pred[b, i : i + min(chunk_size, M-i)] = min_dists[:min(chunk_size, M-i)]
    
    # 确定在tau距离内的匹配项
    precision_matches = (min_dists_pred_to_gt < tau).float()  # (B, N)
    recall_matches = (min_dists_gt_to_pred < tau).float()  # (B, M)
    
    # 计算精确度和召回率
    precision = precision_matches.sum(dim=1) / N  # (B,)
    recall = recall_matches.sum(dim=1) / M  # (B,)
    
    # 计算F-Score
    fscore = (
        2 * (precision * recall) / (precision + recall + 1e-8)
    )  # 避免除零
    
    return fscore


def compute_volume_iou(pred, gt, mode="bbox"):
    """
    计算预测和真实点云之间的体积IoU
    
    Args:
        pred: 预测的点云，形状为 (B, N, 3)
        gt: 真实的点云，形状为 (B, N, 3)
        mode: 计算体积IoU的模式，可以是"bbox"
        
    Returns:
        torch.Tensor: 每个批次的体积IoU
    """
    if mode == "bbox":
        # 计算边界框
        pred_min = pred.min(dim=1).values
        pred_max = pred.max(dim=1).values
        gt_min = gt.min(dim=1).values
        gt_max = gt.max(dim=1).values
        
        # 计算交集
        intersection_min = torch.max(pred_min, gt_min)
        intersection_max = torch.min(pred_max, gt_max)
        inter_dims = (intersection_max - intersection_min).clamp(min=0)
        inter_vol = inter_dims[:, 0] * inter_dims[:, 1] * inter_dims[:, 2]
        
        # 计算并集
        pred_dims = (pred_max - pred_min).clamp(min=0)
        pred_vol = pred_dims[:, 0] * pred_dims[:, 1] * pred_dims[:, 2]  # (B,)
        gt_dims = (gt_max - gt_min).clamp(min=0)
        gt_vol = gt_dims[:, 0] * gt_dims[:, 1] * gt_dims[:, 2]
        
        # 计算IoU
        union_vol = pred_vol + gt_vol - inter_vol
        iou = inter_vol / (union_vol + 1e-8)
    else:
        raise ValueError(f"不支持的模式: {mode}")
    
    return iou


def normalize_points(tensor):
    """
    将点云归一化到[-0.95, 0.95]范围内
    
    Args:
        tensor: 输入点云，形状为 (B, N, 3)
        
    Returns:
        tensor: 归一化后的点云，形状为 (B, N, 3)
    """
    min_vals = tensor.min(dim=1, keepdim=True)[0]
    max_vals = tensor.max(dim=1, keepdim=True)[0]
    
    ranges = max_vals - min_vals
    ranges = torch.where(ranges == 0, torch.ones_like(ranges), ranges)
    
    normalized_tensor = 1.9 * (tensor - min_vals) / ranges - 0.95
    
    return normalized_tensor


def sample_points_from_meshes(meshes, num_samples=20000):
    """
    从三角网格中采样点
    
    Args:
        meshes: 网格列表或单个网格
        num_samples: 采样点数量
        
    Returns:
        torch.Tensor: 采样点，形状为 (B, num_samples, 3)
    """
    if not isinstance(meshes, list):
        meshes = [meshes]
    
    vertices = []
    for mesh in meshes:
        # 使用trimesh采样表面点
        try:
            vert = trimesh.sample.sample_surface(mesh, num_samples)[0]
        except:
            # 如果表面采样失败，转为体积采样
            vert = trimesh.sample.volume_mesh(mesh, num_samples)
        
        vertices.append(torch.from_numpy(vert).float())
    
    return torch.stack(vertices) if len(vertices) > 1 else vertices[0].unsqueeze(0) 