# EvalScene - 3D场景生成评测框架

EvalScene是一个基于Hydra配置系统的通用3D场景生成模型评测框架，支持多种评测指标和ICP配准功能。

## 功能特点

- **多种评测指标**：支持Chamfer距离、F-score、体积IoU等多种评测指标
- **ICP配准**：自动对齐预测与参考点云，提供更准确的形状相似度评估
- **场景级与对象级评测**：同时支持整体场景和单个对象的评测
- **结构化配置**：使用Hydra的Structured Config而非YAML配置
- **灵活可扩展**：易于添加新的评测指标和评测任务

## 安装

首先，克隆仓库并安装依赖：

```bash
git clone https://github.com/yourusername/evalscene.git
cd evalscene
pip install -r requirements.txt
```

## 使用方法

### 基本用法

最简单的用法是通过命令行直接运行评测：

```bash
python evaluate.py system.input_dir=/path/to/predictions system.output_dir=/path/to/groundtruth
```

其中：
- `system.input_dir`：包含预测模型的文件夹路径（必需）
- `system.output_dir`：包含真实模型（参考模型）的文件夹路径（必需）

### 目录结构要求

输入和输出目录应具有相同的文件名结构。例如：

```
预测目录：
/path/to/predictions/
  ├── scene_001.glb
  ├── scene_002.glb
  └── scene_003.glb

参考目录：
/path/to/groundtruth/
  ├── scene_001.glb
  ├── scene_002.glb
  └── scene_003.glb
```

系统会自动匹配同名文件进行评测。

## 配置选项

可以通过命令行覆盖配置文件中的任何参数：

```bash
python evaluate.py \
  system.input_dir=/path/to/predictions \
  system.output_dir=/path/to/groundtruth \
  system.save_dir=outputs/custom_results \
  system.num_points=30000 \
  system.fscore_threshold=0.05 \
  experiment_name=我的测试
```

### 主要配置参数

#### 基础参数
- `system.input_dir`：预测模型目录（必需）
- `system.output_dir`：参考模型目录（必需）
- `system.save_dir`：结果保存目录（默认为`outputs/eval_results/实验名称`）
- `experiment_name`：实验名称（默认为`scene_evaluation`）

#### 数据参数
- `system.input_format`：输入文件格式（默认为`glb`）
- `system.gt_format`：参考文件格式（默认为`glb`）
- `system.num_points`：每个模型采样的点数（默认为20000）

#### 评测参数
- `system.eval_scene_level`：是否进行场景级评估（默认为`true`）
- `system.eval_object_level`：是否进行对象级评估（默认为`true`）
- `system.fscore_threshold`：F-score阈值（默认为0.1）

#### ICP配准参数
- `system.use_icp`：是否使用ICP配准（默认为`true`）
- `system.icp_max_iterations`：ICP最大迭代次数（默认为50）
- `system.icp_tolerance`：ICP收敛容差（默认为1e-5）

## 评测指标

EvalScene支持以下主要评测指标：

- **Chamfer距离（CD）**：衡量两个点云之间的平均距离，值越小越好
  - `scene_cd`：场景级Chamfer距离
  - `scene_cd_1`：从预测到真实点云的平均距离
  - `scene_cd_2`：从真实到预测点云的平均距离
  - `object_cd`：对象级Chamfer距离

- **F-score**：精确度和召回率的调和平均数，值越高越好
  - `scene_fscore`：场景级F-score
  - `object_fscore`：对象级F-score

- **体积IoU**：交并比，衡量体积重叠程度，值越高越好
  - `iou_bbox`：边界框IoU

## ICP配准

迭代最近点（ICP）算法用于对齐预测与参考点云，减少由坐标系差异或初始姿态引起的误差。ICP流程：

1. 在每次迭代中，找到源点云中每个点在目标点云中的最近点
2. 计算将源点云映射到最近点集合的最佳刚性变换（旋转矩阵R和平移向量t）
3. 应用变换到源点云
4. 重复直到收敛或达到最大迭代次数

EvalScene会在评测前自动执行ICP配准，并报告变换参数。

## 项目结构

```
evalscene/
├── metrics/             # 评测指标实现
│   ├── __init__.py
│   ├── chamfer_distance.py  # Chamfer距离计算
│   ├── icp.py           # ICP配准算法
│   └── metrics.py       # 其他评测指标
├── systems/             # 评测系统
│   ├── __init__.py
│   ├── base_system.py   # 评测系统基类
│   └── scene_system.py  # 3D场景评测系统
├── configs/             # 配置
│   ├── __init__.py
│   ├── base/            # 基础配置
│   └── test/            # 测试配置
├── evaluate.py          # 主评测程序
├── requirements.txt     # 依赖项
└── README.md            # 本文档
```

## 示例用法

### 基本评测

```bash
python evaluate.py \
  system.input_dir=/data/predictions \
  system.output_dir=/data/references
```

### 自定义评测参数

```bash
python evaluate.py \
  system.input_dir=/data/predictions \
  system.output_dir=/data/references \
  system.save_dir=outputs/my_results \
  system.num_points=30000 \
  system.fscore_threshold=0.05 \
  system.use_icp=true \
  system.icp_max_iterations=100 \
  experiment_name=高精度评测
```

### 禁用ICP配准

```bash
python evaluate.py \
  system.input_dir=/data/predictions \
  system.output_dir=/data/references \
  system.use_icp=false
```

## 评测结果

评测完成后，结果将保存在指定的`save_dir`目录中：

```
outputs/eval_results/experiment_name/
  └── metrics.csv  # 包含所有评测指标
```

同时，评测结果也会在控制台输出。 