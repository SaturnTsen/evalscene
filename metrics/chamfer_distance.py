import torch


class ChamferDistance(torch.nn.Module):
    """
    计算点云之间的Chamfer距离
    """
    def __init__(self):
        super().__init__()

    def forward(self, xyz1, xyz2):
        """
        计算两个点云之间的Chamfer距离
        
        Args:
            xyz1: 第一个点云, 形状为 (B, N, 3)
            xyz2: 第二个点云, 形状为 (B, M, 3)
            
        Returns:
            dist1: 从xyz1到xyz2的距离 (B, N)
            dist2: 从xyz2到xyz1的距离 (B, M)
        """
        B, N, C = xyz1.shape
        _, M, _ = xyz2.shape
        
        # 计算每个点云中的点与另一个点云中所有点的欧几里德距离
        xyz1 = xyz1.unsqueeze(2)  # (B, N, 1, C)
        xyz2 = xyz2.unsqueeze(1)  # (B, 1, M, C)
        
        # 计算距离矩阵
        dist = torch.sum((xyz1 - xyz2) ** 2, dim=-1)  # (B, N, M)
        
        # 获取最小距离
        dist1, _ = torch.min(dist, dim=2)  # (B, N)
        dist2, _ = torch.min(dist, dim=1)  # (B, M)
        
        return dist1, dist2 