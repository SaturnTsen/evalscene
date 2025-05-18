#!/usr/bin/env python
import os
import sys
import logging
import random
import numpy as np
import torch
import hydra
from omegaconf import OmegaConf
from dataclasses import dataclass, field
from typing import Optional

from systems import SceneSystem
from configs import SceneEvaluationConfig


# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def set_seed(seed):
    """设置随机种子以确保结果可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@hydra.main(version_base=None, config_name="scene_evaluation", config_path="configs/test")
def main(cfg: SceneEvaluationConfig):
    """主评测函数"""
    logger.info(f"评测配置:\n{OmegaConf.to_yaml(cfg)}")
    
    # 设置随机种子
    set_seed(cfg.seed)
    
    # 确保输入和输出目录有效
    if not cfg.system.input_dir or not os.path.exists(cfg.system.input_dir):
        logger.error(f"输入目录无效: {cfg.system.input_dir}")
        return
    
    if not cfg.system.output_dir or not os.path.exists(cfg.system.output_dir):
        logger.error(f"输出目录无效: {cfg.system.output_dir}")
        return
    
    # 创建保存目录
    os.makedirs(cfg.system.save_dir, exist_ok=True)
    
    # 创建评测系统
    logger.info("初始化评测系统...")
    system = SceneSystem(cfg.system)
    system.setup()
    
    # 执行评测
    logger.info("开始评测...")
    system.test_directory()
    
    # 完成评测
    system.on_test_end()
    logger.info(f"评测完成，结果保存在: {cfg.system.save_dir}")


if __name__ == "__main__":
    main() 