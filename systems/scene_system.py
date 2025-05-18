import os
import torch
import logging
import trimesh
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field

from .base_system import BaseSystem
from ..metrics import (
    compute_chamfer_distance,
    compute_fscore,
    compute_volume_iou,
    normalize_points,
    sample_points_from_meshes,
    icp,
)

logger = logging.getLogger(__name__)


class SceneSystem(BaseSystem):
    """3D场景生成评测系统"""
    
    @dataclass
    class Config(BaseSystem.Config):
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
        
        # 标准指标列表
        metrics: List[str] = field(default_factory=lambda: [
            "scene_cd", "scene_fscore", "object_cd", "object_fscore", "iou_bbox"
        ])
    
    def __init__(self, cfg: Config):
        """初始化场景评测系统"""
        super().__init__(cfg)
        self.cfg = cfg
        self.metrics_values = {}  # 存储评测指标结果
    
    def setup(self):
        """设置评测环境"""
        super().setup()
        logger.info(f"场景评测系统已设置，采样点数: {self.cfg.num_points}")
        logger.info(f"ICP配准: {'启用' if self.cfg.use_icp else '禁用'}")
    
    @torch.no_grad()
    def compute_metrics(self, pred_meshes, gt_meshes, use_icp=None):
        """
        计算评测指标
        
        Args:
            pred_meshes: 预测的网格列表
            gt_meshes: 真实的网格列表
            use_icp: 是否使用ICP进行配准
            
        Returns:
            Dict[str, torch.Tensor]: 评测指标
        """
        if use_icp is None:
            use_icp = self.cfg.use_icp
        
        # 采样点云
        vertices_pred = [
            torch.from_numpy(
                sample_points_from_meshes(mesh, self.cfg.num_points)
            ).to(self.device)
            for mesh in pred_meshes
        ]
        vertices_pred = torch.cat(vertices_pred) if len(vertices_pred) > 1 else vertices_pred[0]
        
        vertices_gt = [
            torch.from_numpy(
                sample_points_from_meshes(mesh, self.cfg.num_points)
            ).to(self.device)
            for mesh in gt_meshes
        ]
        vertices_gt = torch.cat(vertices_gt) if len(vertices_gt) > 1 else vertices_gt[0]
        
        # 确保形状正确
        if vertices_pred.dim() == 2:
            vertices_pred = vertices_pred.unsqueeze(0)
        if vertices_gt.dim() == 2:
            vertices_gt = vertices_gt.unsqueeze(0)
        
        metrics = {}
        
        # 场景级别评测
        if self.cfg.eval_scene_level:
            scene_metrics = self._compute_scene_metrics(
                vertices_pred, vertices_gt, use_icp=use_icp
            )
            metrics.update(scene_metrics)
        
        # 对象级别评测
        if self.cfg.eval_object_level and len(pred_meshes) > 1 and len(gt_meshes) > 1:
            object_metrics = self._compute_object_metrics(
                vertices_pred, vertices_gt, use_icp=use_icp
            )
            metrics.update(object_metrics)
        
        return metrics
    
    def _compute_scene_metrics(self, vertices_pred, vertices_gt, use_icp=True):
        """计算场景级别指标"""
        # 将所有对象点合并为场景点云
        if vertices_pred.size(0) > 1:
            vertices_scene_pred = vertices_pred.reshape(-1, 3).unsqueeze(0)
            vertices_scene_gt = vertices_gt.reshape(-1, 3).unsqueeze(0)
        else:
            vertices_scene_pred = vertices_pred
            vertices_scene_gt = vertices_gt
        
        # 应用ICP配准
        if use_icp:
            vertices_scene_pred, R, t = icp(
                vertices_scene_pred, 
                vertices_scene_gt,
                max_iterations=self.cfg.icp_max_iterations,
                tolerance=self.cfg.icp_tolerance
            )
            logger.info(f"ICP配准完成，旋转矩阵: {R}")
            logger.info(f"ICP配准完成，平移向量: {t}")
        
        # 计算Chamfer距离
        cds = compute_chamfer_distance(vertices_scene_pred, vertices_scene_gt)
        
        # 计算F-score
        fscore = compute_fscore(
            vertices_scene_pred, vertices_scene_gt, tau=self.cfg.fscore_threshold
        )
        
        return {
            "scene_cd": cds[0],
            "scene_cd_1": cds[1],
            "scene_cd_2": cds[2],
            "scene_fscore": fscore,
        }
    
    def _compute_object_metrics(self, vertices_pred, vertices_gt, use_icp=True):
        """计算对象级别指标"""
        # 规范化点云以便对象级别比较
        vertices_object_pred = normalize_points(vertices_pred)
        vertices_object_gt = normalize_points(vertices_gt)
        
        # 应用ICP配准
        if use_icp:
            vertices_object_pred, _, _ = icp(
                vertices_object_pred, 
                vertices_object_gt,
                max_iterations=self.cfg.icp_max_iterations,
                tolerance=self.cfg.icp_tolerance
            )
        
        # 计算Chamfer距离
        cd_object = compute_chamfer_distance(
            vertices_object_pred, vertices_object_gt
        )[0]
        
        # 计算F-score
        fscore_object = compute_fscore(
            vertices_object_pred, vertices_object_gt, tau=self.cfg.fscore_threshold
        )
        
        # 计算边界框IoU
        iou_bbox = compute_volume_iou(
            vertices_pred, vertices_gt, mode="bbox"
        )
        
        return {
            "object_cd": cd_object,
            "object_fscore": fscore_object,
            "iou_bbox": iou_bbox,
        }
    
    def load_mesh(self, file_path):
        """加载网格文件"""
        try:
            mesh = trimesh.load(file_path)
            return mesh
        except Exception as e:
            logger.error(f"加载文件 {file_path} 时出错: {e}")
            return None
    
    def test_step(self, pred_path, gt_path):
        """
        执行评测步骤
        
        Args:
            pred_path: 预测模型路径
            gt_path: 真实模型路径
        """
        logger.info(f"评测: {pred_path} vs {gt_path}")
        
        # 加载模型
        pred_mesh = self.load_mesh(pred_path)
        gt_mesh = self.load_mesh(gt_path)
        
        if pred_mesh is None or gt_mesh is None:
            logger.error("模型加载失败，跳过此评测")
            return None
        
        # 计算指标
        metrics = self.compute_metrics([pred_mesh], [gt_mesh])
        
        # 更新指标值
        for k, v in metrics.items():
            if k not in self.metrics_values:
                self.metrics_values[k] = []
            self.metrics_values[k].append(v.item())
        
        return metrics
    
    def test_directory(self):
        """评测目录中的所有模型"""
        input_dir = Path(self.cfg.input_dir)
        gt_dir = Path(self.cfg.output_dir)
        
        if not input_dir.exists() or not gt_dir.exists():
            logger.error(f"输入目录 {input_dir} 或输出目录 {gt_dir} 不存在")
            return
        
        # 获取所有预测文件
        pred_files = list(input_dir.glob(f"*.{self.cfg.input_format}"))
        total = len(pred_files)
        logger.info(f"找到 {total} 个预测文件")
        
        for i, pred_file in enumerate(pred_files):
            logger.info(f"评测进度: {i+1}/{total}")
            
            # 查找对应的真实文件
            file_name = pred_file.stem
            gt_file = gt_dir / f"{file_name}.{self.cfg.gt_format}"
            
            if not gt_file.exists():
                logger.warning(f"找不到 {file_name} 的真实文件")
                continue
            
            # 执行评测
            self.test_step(str(pred_file), str(gt_file))
    
    def on_test_end(self):
        """测试结束时计算最终指标"""
        # 计算所有指标的平均值
        final_metrics = {}
        for k, values in self.metrics_values.items():
            if values:
                final_metrics[k] = sum(values) / len(values)
        
        # 保存结果
        self.save_metrics_to_csv(final_metrics)
        
        # 打印结果
        logger.info("评测完成，最终指标:")
        for k, v in final_metrics.items():
            logger.info(f"{k}: {v}")
        
        super().on_test_end() 