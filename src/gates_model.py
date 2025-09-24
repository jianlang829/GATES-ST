# src/gates_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GATES(nn.Module):
    """
    GATES (Graph Attention Encoder) 模型
    使用两个独立的 GAT 层分别处理空间网络和基因相似性网络，然后进行加权融合。
    """

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, alpha: float = 0.5):
        """
        初始化模型。

        Args:
            in_channels: 输入特征维度
            hidden_channels: 隐藏层维度
            out_channels: 输出嵌入维度
            alpha: 基因相似性网络的权重，最终融合为: alpha * gene_sim + (1-alpha) * spatial
        """
        super().__init__()
        self.alpha = alpha

        # 定义两个独立的 GAT 层
        self.spatial_gat = GATConv(in_channels, hidden_channels, add_self_loops=True, concat=False)
        self.gene_sim_gat = GATConv(in_channels, hidden_channels, add_self_loops=True, concat=False)

        # 融合后的第二层 GAT
        self.fusion_gat = GATConv(hidden_channels, out_channels, add_self_loops=True, concat=False)

    def forward(self, x: torch.Tensor, spatial_edge_index: torch.Tensor, gene_sim_edge_index: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        Args:
            x: 节点特征 [N, in_channels]
            spatial_edge_index: 空间网络的边索引 [2, E_spatial]
            gene_sim_edge_index: 基因相似性网络的边索引 [2, E_gene]

        Returns:
            节点嵌入 [N, out_channels]
        """
        # 分别通过两个GAT层
        h_spatial = F.elu(self.spatial_gat(x, spatial_edge_index))
        h_gene_sim = F.elu(self.gene_sim_gat(x, gene_sim_edge_index))

        # 按 alpha 加权融合
        h_fused = self.alpha * h_gene_sim + (1 - self.alpha) * h_spatial

        # 通过第二层GAT
        out = self.fusion_gat(h_fused, spatial_edge_index)  # 这里可以选择用哪种边，或再构建融合边

        return out
