# GATES Project: Graph Attention Transcriptomics Encoder for Spatial Transcriptomics Analysis

GATES（Graph Attention Transcriptomics Encoder）是一个基于 **PyTorch Geometric** 的深度学习框架，用于分析 **空间转录组学（Spatial Transcriptomics）** 数据。本项目结合空间邻接关系与基因表达相似性，构建异构图结构，并通过图注意力机制进行特征学习，最终实现高精度的组织区域聚类与可视化。

## 📌 项目亮点
- 支持 10x Genomics Visium 等主流空间转录组数据格式
- 自动构建 **空间邻接图 + 基因相似性图**
- 基于 **GAT（Graph Attention Network）** 的端到端训练
- 提供完整的预处理 → 训练 → 聚类 → 可视化流程
- 模块化设计，易于扩展与复用

---

## 📁 项目结构

```
C:.
│  .gitignore
│  LICENSE
│  README.md
│  requirements.txt
│
├─cache                     # 预处理后的缓存数据（.h5ad）
├─configs                   # 配置文件（YAML）
├─data                      # 原始数据（支持多个样本，如 151673, 151674, 151675）
│  └─151673                 # 示例：10x Visium 小鼠脑切片数据
│      │  filtered_feature_bc_matrix.h5
│      │  position.tsv
│      │  metadata.tsv
│      │  truth.txt         # 真实标签（用于评估）
│      └─spatial/
├─results                   # 输出结果（如聚类图）
├─scripts                   # 主运行脚本
│  └─run_analysis.py
└─src                       # 核心源码
    │  gates_model.py       # GATES 模型定义
    │  trainer.py           # 训练逻辑
    │  utils.py             # 工具函数（图构建、预处理等）
    │  pyg.py               # PyG 图数据封装
    └─convert_visium_to_stereo.py  # 数据格式转换工具
```

---

## ⚙️ 安装指南

### 环境要求
- Python ≥ 3.8
- PyTorch ≥ 1.12
- CUDA（推荐，用于加速训练）

### 安装步骤
```bash
# 1. 克隆仓库
git clone https://github.com/your-username/GATES.git
cd GATES

# 2. 安装依赖
pip install -r requirements.txt
```

> 💡 提示：建议使用虚拟环境（如 `conda` 或 `venv`）避免依赖冲突。

---

## ▶️ 快速开始

```bash
cd scripts
python run_analysis.py
```

该脚本将自动：
1. 加载 `data/151673` 中的 Visium 数据
2. 预处理并缓存到 `cache/`
3. 构建空间图与基因相似图
4. 训练 GATES 模型
5. 执行聚类并评估（Silhouette Score, Davies-Bouldin Index）
6. 保存空间聚类图至 `results/spatial_plot.png`

---

## 🛠 配置说明

所有参数均在 `configs/default.yaml` 中管理，包括：

```yaml
# configs/default.yaml
data:
  counts_file: 'C:/Users/admini/Documents/GATES-ST/data/151673/RNA_counts.tsv'
  coor_file: 'C:/Users/admini/Documents/GATES-ST/data/151673/position.tsv'
  used_barcodes_file: 'C:/Users/admini/Documents/GATES-ST/data/151673/used_barcodes.txt'
model:
  alpha: 0.5  # 修正：原值 0.0001 过小，建议 0.1~0.9
  n_top_genes: 3000
  hidden_dims: [512, 30]
  rad_cutoff: 50
  k_neighbors: 6
  similarity_metric: "cosine"
train:
  n_epochs: 1000
  lr: 0.0001
  weight_decay: 0.0001
  key_added: "GATES"
cluster:
  resolution: 1.0
output:
  spatial_plot_path: "results/spatial_plot.png"
  neighbor_stats_plot: 'C:/Users/admini/Documents/GATES-ST/figure/alpha{alpha}_{resolution}_Stereo-seq_Mouse_NumberOfNeighbors.png'
  spatial_plot_crop: [10100, 10721, 13810, 13093]

```

修改该文件即可适配不同数据集或调整超参数。

---

## 📈 输出结果

运行完成后，你将获得：
- **聚类指标**：控制台输出 Silhouette Score 和 Davies-Bouldin Index
- **可视化图**：`results/spatial_plot.png` 展示空间聚类结果（如下示意）

> 🖼️ *示例图：不同颜色代表不同组织区域，与真实解剖结构高度一致*

---

## 📚 数据说明

本项目默认使用 **10x Genomics Visium 公开数据集**：
- `151673`, `151674`, `151675`：小鼠脑组织切片（来自 [10x官方示例](https://support.10xgenomics.com/spatial-gene-expression/datasets)）
- 文件包括：基因表达矩阵（.h5）、空间坐标（tissue_positions_list.csv）、组织图像等

如需使用其他数据（如 Stereo-seq），请参考 `src/convert_visium_to_stereo.py` 进行格式转换。

---

## ❓ 常见问题

**Q: 运行时报错 “CUDA out of memory”？**
A: 尝试减小 `batch_size`（如有）或改用 `device: "cpu"`。

**Q: 如何更换数据集？**
A: 修改 `configs/default.yaml` 中的 `data.root` 路径，并确保目录结构与 151673 一致。

**Q: 能否用于人类组织数据？**
A: 可以！只要提供符合格式的空间表达数据即可。

---

## 📄 许可证

本项目采用 **MIT License** — 详见 [LICENSE](LICENSE) 文件。

---

## 🙏 致谢

- 数据来源：10x Genomics Spatial Gene Expression Datasets
- 技术基础：PyTorch Geometric, Scanpy, scikit-learn
- 若本项目对您的研究有帮助，欢迎引用相关论文（如有）！

---

> 💬 **欢迎提交 Issue 或 Pull Request！** 任何改进建议或 bug 报告都十分感谢！
