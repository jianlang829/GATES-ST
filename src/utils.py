import os
import numpy as np
import pandas as pd
import scanpy as sc
import anndata
from scipy.sparse import csr_matrix, issparse
from sklearn.neighbors import NearestNeighbors
from typing import Tuple, Dict, List, Optional
import warnings

def load_data(counts_file: str, coor_file: str) -> Tuple[np.ndarray, np.ndarray]:
    """加载基因表达矩阵和空间坐标."""
    counts = pd.read_csv(counts_file, sep='\t', index_col=0).values
    coors = pd.read_csv(coor_file, sep='\t', index_col=0).values
    return counts, coors

def filter_cells_genes(counts: np.ndarray, min_genes: int = 200, min_cells: int = 3) -> np.ndarray:
    """过滤低质量细胞和基因."""
    # 过滤基因：在至少 min_cells 个细胞中表达
    gene_counts = np.sum(counts > 0, axis=0)
    valid_genes = gene_counts >= min_cells
    counts = counts[:, valid_genes]

    # 过滤细胞：至少有 min_genes 个基因表达
    cell_counts = np.sum(counts > 0, axis=1)
    valid_cells = cell_counts >= min_genes
    counts = counts[valid_cells, :]

    print(f"Filtered: {valid_cells.sum()} cells, {valid_genes.sum()} genes remain.")
    return counts

def build_spatial_network(coors: np.ndarray, radius_cutoff: float = 50) -> pd.DataFrame:
    """基于空间距离构建邻接网络（半径过滤）."""
    n_cells = coors.shape[0]
    edges = []

    for i in range(n_cells):
        dists = np.sqrt(np.sum((coors[i] - coors) ** 2, axis=1))
        neighbors = np.where(dists <= radius_cutoff)[0]
        for j in neighbors:
            if i != j:
                edges.append([i, j])
    return pd.DataFrame(edges, columns=['Cell1', 'Cell2'])

def build_knn_graph(features: np.ndarray, k: int = 6, metric: str = 'cosine') -> pd.DataFrame:
    """构建 KNN 图（支持 cosine / euclidean，未来可扩展 pearson）."""
    n_cells = features.shape[0]
    edges = []

    if metric == 'cosine':
        from sklearn.metrics.pairwise import cosine_similarity
        sim_mat = cosine_similarity(features)
        for i in range(n_cells):
            sims = sim_mat[i]
            topk_indices = np.argsort(sims)[-k-1:-1][::-1]  # 去掉自己，取topk
            for j in topk_indices:
                edges.append([i, j])

    elif metric == 'euclidean':
        from sklearn.metrics.pairwise import euclidean_distances
        dist_mat = euclidean_distances(features)
        for i in range(n_cells):
            dists = dist_mat[i]
            topk_indices = np.argsort(dists)[:k]  # 取最近的k个
            for j in topk_indices:
                if i != j:
                    edges.append([i, j])

    elif metric == 'pearson':
        from scipy.stats import pearsonr
        for i in range(n_cells):
            for j in range(i + 1, n_cells):
                corr, _ = pearsonr(features[i], features[j])
                edges.append([i, j, corr, 'pearson'])
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    if metric == 'pearson':
        return pd.DataFrame(edges, columns=['Cell1', 'Cell2', 'Weight', 'Metric'])
    else:
        return pd.DataFrame(edges, columns=['Cell1', 'Cell2'])

def enhanced_gene_similarity_net(
    adata: anndata.AnnData,
    k_neighbors: int = 6,
    metrics: List[str] = ['cosine', 'pearson'],
    verbose: bool = True
) -> pd.DataFrame:
    """
    构建增强版基因相似性网络，支持多相似度度量，返回融合后的边表。
    """
    X = adata.X
    if issparse(X):
        X = X.toarray()
    genes_filter = adata.var.get('highly_variable', np.ones(X.shape[1], dtype=bool))
    X = X[:, genes_filter]

    all_edges = []

    for metric in metrics:
        if metric == 'cosine':
            from sklearn.metrics.pairwise import cosine_similarity
            sim_mat = cosine_similarity(X)
            for i in range(sim_mat.shape[0]):
                sims = sim_mat[i]
                topk_indices = np.argsort(sims)[-k_neighbors-1:-1][::-1]
                for j in topk_indices:
                    all_edges.append([i, j])

        elif metric == 'euclidean':
            from sklearn.metrics.pairwise import euclidean_distances
            dist_mat = euclidean_distances(X)
            for i in range(dist_mat.shape[0]):
                dists = dist_mat[i]
                topk_indices = np.argsort(dists)[:k_neighbors]
                for j in topk_indices:
                    if i != j:
                        all_edges.append([i, j])

        elif metric == 'pearson':
            from scipy.stats import pearsonr
            for i in range(X.shape[0]):
                for j in range(i + 1, X.shape[0]):
                    corr, _ = pearsonr(X[i], X[j])
                    all_edges.append([i, j, corr, 'pearson'])

        else:
            warnings.warn(f"Metric {metric} not implemented, skipping.")
            continue

    if metrics == ['pearson']:  # 仅当只有 pearson 时才返回带权重的边
        df = pd.DataFrame(all_edges, columns=['Cell1', 'Cell2', 'Weight', 'Metric'])
    else:
        df = pd.DataFrame(all_edges, columns=['Cell1', 'Cell2'])

    if verbose:
        print(f"Built gene similarity network with {len(df)} edges.")
    return df

# 兼容旧函数（默认只使用 cosine）
def Cal_Gene_Similarity_Net(adata, k_neighbors=6, metric='cosine'):
    return enhanced_gene_similarity_net(adata, k_neighbors=k_neighbors, metrics=[metric])

def calculate_neighbor_stats(adata, spatial_net):
    cell_degrees = {}
    for _, row in spatial_net.iterrows():
        c1, c2 = int(row['Cell1']), int(row['Cell2'])
        cell_degrees[c1] = cell_degrees.get(c1, 0) + 1
        cell_degrees[c2] = cell_degrees.get(c2, 0) + 1
    degrees = [cell_degrees.get(i, 0) for i in range(adata.n_obs)]
    print(f"Neighbor stats: mean={np.mean(degrees):.2f}, max={np.max(degrees)}, min={np.min(degrees)}")
