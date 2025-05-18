from dataclasses import dataclass, field
from typing import List, Optional

from ..base import SceneSystemConfig


@dataclass
class SceneEvaluationConfig:
    """场景评测配置"""
    # 系统配置
    system: SceneSystemConfig = field(default_factory=SceneSystemConfig)
    
    # 常规配置
    experiment_name: str = "scene_evaluation"
    seed: int = 42
    
    def __post_init__(self):
        """初始化后处理"""
        # 如果保存目录未指定，则使用默认命名规则
        if self.system.save_dir == "outputs/eval_results":
            self.system.save_dir = f"outputs/eval_results/{self.experiment_name}" 