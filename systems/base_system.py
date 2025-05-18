import os
import torch
import hydra
import logging
import pandas as pd
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


@dataclass
class BaseSystem:
    """场景评测系统基类，所有评测系统都应继承此类"""
    
    @dataclass
    class Config:
        """基础配置类，使用Structured Config"""
        # 输入输出配置
        input_dir: str = ""  # 输入文件夹路径
        output_dir: str = ""  # 输出文件夹路径
        save_dir: str = ""  # 评测结果保存路径
        
        # 评测配置
        metrics: List[str] = None  # 要计算的指标列表
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    def __init__(self, cfg: Config):
        """
        初始化评测系统
        
        Args:
            cfg: 系统配置
        """
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        
        # 检查并创建必要的目录
        self._setup_directories()
        
        logger.info(f"评测系统已初始化，设备: {cfg.device}")
    
    def _setup_directories(self):
        """设置并创建必要的目录"""
        if self.cfg.save_dir:
            os.makedirs(self.cfg.save_dir, exist_ok=True)
            logger.info(f"结果将保存到: {self.cfg.save_dir}")
    
    def setup(self):
        """设置评测环境，子类可以覆盖此方法"""
        pass
    
    def compute_metrics(self, pred_data, gt_data, **kwargs) -> Dict[str, torch.Tensor]:
        """
        计算评测指标，子类应该实现此方法
        
        Args:
            pred_data: 预测数据
            gt_data: 真实数据
            
        Returns:
            Dict[str, torch.Tensor]: 指标名称到指标值的映射
        """
        raise NotImplementedError("子类必须实现compute_metrics方法")
    
    def test_step(self, batch, batch_idx):
        """
        执行测试步骤，子类应该实现此方法
        
        Args:
            batch: 批次数据
            batch_idx: 批次索引
        """
        raise NotImplementedError("子类必须实现test_step方法")
    
    def on_test_end(self):
        """测试结束时的回调，子类可以覆盖此方法"""
        logger.info("评测完成")
    
    def get_save_dir(self) -> str:
        """获取结果保存目录"""
        return self.cfg.save_dir
    
    def save_metrics_to_csv(self, metrics: Dict[str, Any], filename: str = "metrics.csv"):
        """
        将指标保存到CSV文件
        
        Args:
            metrics: 指标字典
            filename: 输出文件名
        """
        # 将指标转换为DataFrame并保存
        metrics_dict = {k: [v.item() if torch.is_tensor(v) else v] for k, v in metrics.items()}
        df = pd.DataFrame(metrics_dict)
        save_path = os.path.join(self.get_save_dir(), filename)
        df.to_csv(save_path, index=False)
        logger.info(f"指标已保存到: {save_path}")
        
        # 也打印到控制台
        logger.info("评测指标:")
        for k, v in metrics.items():
            logger.info(f"{k}: {v.item() if torch.is_tensor(v) else v}") 