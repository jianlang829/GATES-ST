import scanpy as sc
import pandas as pd
import numpy as np

class GraphPruning:
    """从TF版本学习的图剪枝技术"""

    @staticmethod
    def prune_by_preclustering(adata, spatial_net, resolution=0.2):
        """基于预聚类结果修剪空间图（TF核心创新）"""
        print('Pruning graph by pre-clustering...')

        # 预聚类
        sc.tl.pca(adata)
        sc.pp.neighbors(adata)
        sc.tl.louvain(adata, resolution=resolution, key_added='pre_cluster')

        # 修剪：只保留同类别间的边
        cluster_dict = dict(zip(adata.obs_names, adata.obs['pre_cluster']))
        spatial_net['Cell1_Cluster'] = spatial_net['Cell1'].map(cluster_dict)
        spatial_net['Cell2_Cluster'] = spatial_net['Cell2'].map(cluster_dict)

        pruned_net = spatial_net[spatial_net['Cell1_Cluster'] == spatial_net['Cell2_Cluster']].copy()

        print(f'Pruned from {len(spatial_net)} to {len(pruned_net)} edges')
        return pruned_net

class AttentionAnalyzer:
    """注意力机制分析工具（TF版本的可解释性思想）"""

    @staticmethod
    def analyze_attention_weights(attention_weights, adata):
        """分析注意力权重分布"""
        analysis = {}

        for graph_type, (edge_index, weights) in attention_weights.items():
            analysis[graph_type] = {
                'mean_attention': weights.mean().item(),
                'std_attention': weights.std().item(),
                'sparsity': (weights < 0.1).float().mean().item()  # 稀疏度
            }

        return analysis

    @staticmethod
    def visualize_attention_patterns(attention_weights, adata, save_path=None):
        """可视化注意力模式"""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        for idx, (graph_type, (edge_index, weights)) in enumerate(attention_weights.items()):
            axes[idx].hist(weights.cpu().numpy(), bins=50, alpha=0.7)
            axes[idx].set_title(f'{graph_type} Attention Distribution')
            axes[idx].set_xlabel('Attention Weight')
            axes[idx].set_ylabel('Frequency')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
