import torch


def compute_nearest_neighbors(src, dst):
    """
    计算从src到dst的最近邻点

    Args:
        src (torch.Tensor): 源点云，形状为 (B, N, 3)
        dst (torch.Tensor): 目标点云，形状为 (B, M, 3)

    Returns:
        torch.Tensor: dst中每个src点的最近邻索引，形状为 (B, N)
    """
    # 为每个点计算欧几里德距离
    B, N, _ = src.shape
    _, M, _ = dst.shape
    
    nn_indices = torch.zeros((B, N), dtype=torch.long, device=src.device)
    
    for b in range(B):
        # 计算当前批次中每个点的距离
        src_expanded = src[b].unsqueeze(1)  # (N, 1, 3)
        dst_expanded = dst[b].unsqueeze(0)  # (1, M, 3)
        
        # 计算欧几里德距离的平方
        dist = torch.sum((src_expanded - dst_expanded) ** 2, dim=2)  # (N, M)
        
        # 获取最小距离对应的索引
        min_idx = torch.argmin(dist, dim=1)  # (N,)
        nn_indices[b] = min_idx
    
    return nn_indices


def compute_rigid_transform(A, B):
    """
    计算将A对齐到B的刚性变换(R, t)

    Args:
        A (torch.Tensor): 源点云，形状为 (B, N, 3)
        B (torch.Tensor): 目标点云，形状为 (B, N, 3)

    Returns:
        R (torch.Tensor): 旋转矩阵，形状为 (B, 3, 3)
        t (torch.Tensor): 平移向量，形状为 (B, 3, 1)
    """
    # 计算质心
    centroid_A = A.mean(dim=1, keepdim=True)
    centroid_B = B.mean(dim=1, keepdim=True)
    
    # 中心化点
    AA = A - centroid_A
    BB = B - centroid_B
    
    # 计算协方差矩阵
    H = torch.matmul(AA.transpose(1, 2), BB)
    
    # 计算SVD分解
    U, S, Vt = torch.svd(H)
    V = Vt.transpose(1, 2)
    R = torch.matmul(V, U.transpose(1, 2))
    
    # 处理反射情况
    det_R = torch.det(R)
    ones = torch.ones_like(det_R)
    eye = torch.eye(3, device=R.device).unsqueeze(0).repeat(R.size(0), 1, 1)
    diag = torch.diag_embed(torch.stack([ones, ones, det_R], dim=1))
    R = torch.matmul(torch.matmul(V, diag), U.transpose(1, 2))
    
    # 计算平移
    t = centroid_B.transpose(1, 2) - torch.matmul(
        R, centroid_A.transpose(1, 2)
    )  # (B, 3, 1)
    
    return R, t


def icp(src, dst, max_iterations=20, tolerance=1e-5):
    """
    执行ICP算法将src对齐到dst

    Args:
        src (torch.Tensor): 源点云，形状为 (B, N, 3)
        dst (torch.Tensor): 目标点云，形状为 (B, M, 3)
        max_iterations (int): ICP迭代的最大次数
        tolerance (float): 收敛容差

    Returns:
        torch.Tensor: 变换后的源点云
        torch.Tensor: 累积旋转矩阵，形状为 (B, 3, 3)
        torch.Tensor: 累积平移向量，形状为 (B, 3, 1)
    """
    src_transformed = src.clone()
    prev_error = float("inf")
    
    # 初始化累积旋转和平移
    cumulative_R = (
        torch.eye(3, device=src.device).unsqueeze(0).repeat(src.shape[0], 1, 1)
    )
    cumulative_t = torch.zeros((src.shape[0], 3, 1), device=src.device)
    
    for i in range(max_iterations):
        # 步骤1：找到最近邻点
        nn_indices = compute_nearest_neighbors(src_transformed, dst)
        B, N, _ = src_transformed.shape
        
        # 收集最近邻点
        nearest_neighbors = torch.zeros_like(src_transformed)
        for b in range(B):
            nearest_neighbors[b] = dst[b][nn_indices[b]]
        
        # 步骤2：计算变换
        R, t = compute_rigid_transform(src_transformed, nearest_neighbors)
        
        # 步骤3：应用变换
        src_transformed = torch.matmul(
            src_transformed, R.transpose(1, 2)
        ) + t.transpose(1, 2)
        
        # 更新累积变换
        cumulative_R = torch.matmul(R, cumulative_R)
        cumulative_t = torch.matmul(R, cumulative_t) + t
        
        # 步骤4：检查收敛性
        mean_error = torch.mean(torch.norm(src_transformed - nearest_neighbors, dim=2))
        if torch.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error
    
    return src_transformed, cumulative_R, cumulative_t 