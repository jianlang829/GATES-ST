import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class ImprovedGATES(nn.Module):
    """增强版GATES模型，吸收TensorFlow版本精华"""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 alpha: float = 0.5, spatial_att_heads: int = 4, gene_att_heads: int = 4):
        super().__init__()
        self.alpha = alpha
        self.spatial_att_heads = spatial_att_heads
        self.gene_att_heads = gene_att_heads

        # 多头注意力机制（借鉴TF版本）
        self.spatial_gat = GATConv(in_channels, hidden_channels, heads=spatial_att_heads, concat=True)
        self.gene_sim_gat = GATConv(in_channels, hidden_channels, heads=gene_att_heads, concat=True)

        # 注意力融合层（TF版本的层级注意力思想）
        self.attention_fusion = nn.MultiheadAttention(hidden_channels * spatial_att_heads, num_heads=2)

        # 最终编码层
        self.fusion_gat = GATConv(hidden_channels * spatial_att_heads, out_channels, add_self_loops=True, concat=False)

        # 解码器（增强重构能力）
        self.decoder = nn.Sequential(
            nn.Linear(out_channels, hidden_channels),
            nn.ELU(),
            nn.Linear(hidden_channels, in_channels),
            nn.ReLU()  # 防止负值
        )

        # 注意力权重记录（用于可视化）
        self.attention_weights = {}

    def forward(self, x: torch.Tensor, spatial_edge_index: torch.Tensor, gene_sim_edge_index: torch.Tensor):
        # 空间注意力编码
        h_spatial, spatial_att = self.spatial_gat(x, spatial_edge_index, return_attention_weights=True)
        h_spatial = F.elu(h_spatial)

        # 基因相似性注意力编码
        h_gene_sim, gene_att = self.gene_sim_gat(x, gene_sim_edge_index, return_attention_weights=True)
        h_gene_sim = F.elu(h_gene_sim)

        # 记录注意力权重（TF版本的可解释性思想）
        self.attention_weights['spatial'] = spatial_att
        self.attention_weights['gene_sim'] = gene_att

        # 动态融合（TF版本的alpha加权思想）
        h_fused = self.alpha * h_gene_sim + (1 - self.alpha) * h_spatial

        # 最终编码
        embeddings = self.fusion_gat(h_fused, spatial_edge_index)

        # 重构
        recon = self.decoder(embeddings)

        return embeddings, recon, self.attention_weights

    def encode(self, x: torch.Tensor, spatial_edge_index: torch.Tensor, gene_sim_edge_index: torch.Tensor):
        """仅返回嵌入（用于推理）"""
        with torch.no_grad():
            h_spatial = F.elu(self.spatial_gat(x, spatial_edge_index))
            h_gene_sim = F.elu(self.gene_sim_gat(x, gene_sim_edge_index))
            h_fused = self.alpha * h_gene_sim + (1 - self.alpha) * h_spatial
            embeddings = self.fusion_gat(h_fused, spatial_edge_index)
        return embeddings
