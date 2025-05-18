from dataclasses import dataclass, field
from typing import List, Optional, Union
from omegaconf import MISSING


@dataclass
class BaseSystemConfig:
    """评测系统基础配置"""
    # 输入输出配置
    input_dir: str = MISSING  # 输入文件夹路径，必须指定
    output_dir: str = MISSING  # 输出文件夹路径，必须指定
    save_dir: str = "outputs/eval_results"  # 评测结果保存路径
    
    # 评测配置
    metrics: List[str] = field(default_factory=lambda: [
        "scene_cd", "scene_fscore", "object_cd", "object_fscore", "iou_bbox"
    ])
    device: str = "cuda"


@dataclass
class SceneSystemConfig(BaseSystemConfig):
    """场景评测系统配置"""
    # 数据配置
    input_format: str = "glb"  # 输入文件格式
    gt_format: str = "glb"  # 真实数据文件格式
    
    # 点云采样配置
    num_points: int = 20000  # 每个模型采样的点数
    
    # 评测配置
    eval_scene_level: bool = True  # 是否在场景级别评测
    eval_object_level: bool = True  # 是否在对象级别评测
    use_icp: bool = True  # 是否使用ICP进行配准
    icp_max_iterations: int = 50  # ICP最大迭代次数
    icp_tolerance: float = 1e-5  # ICP收敛容差
    fscore_threshold: float = 0.1  # F-score阈值 