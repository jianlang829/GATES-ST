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
        # 编码器部分
        self.spatial_gat = GATConv(in_channels, hidden_channels, add_self_loops=True, concat=False)
        self.gene_sim_gat = GATConv(in_channels, hidden_channels, add_self_loops=True, concat=False)
        self.fusion_gat = GATConv(hidden_channels, out_channels, add_self_loops=True, concat=False)
        # 解码器：用于重构输入（仅训练时使用）
        self.decoder = nn.Sequential(
            nn.Linear(out_channels, hidden_channels),
            nn.ELU(),
            nn.Linear(hidden_channels, in_channels)
        )

    def forward(self, x: torch.Tensor, spatial_edge_index: torch.Tensor, gene_sim_edge_index: torch.Tensor):
        """
        返回嵌入和重构输出（用于训练）
        """
        h_spatial = F.elu(self.spatial_gat(x, spatial_edge_index))
        h_gene_sim = F.elu(self.gene_sim_gat(x, gene_sim_edge_index))
        h_fused = self.alpha * h_gene_sim + (1 - self.alpha) * h_spatial
        # 使用空间图进行最终融合（符合空间转录组特性）
        embeddings = self.fusion_gat(h_fused, spatial_edge_index)
        recon = self.decoder(embeddings)
        return embeddings, recon

    def encode(self, x: torch.Tensor, spatial_edge_index: torch.Tensor, gene_sim_edge_index: torch.Tensor):
        """
        仅返回嵌入（用于推理）
        """
        h_spatial = F.elu(self.spatial_gat(x, spatial_edge_index))
        h_gene_sim = F.elu(self.gene_sim_gat(x, gene_sim_edge_index))
        h_fused = self.alpha * h_gene_sim + (1 - self.alpha) * h_spatial
        embeddings = self.fusion_gat(h_fused, spatial_edge_index)
        return embeddings
