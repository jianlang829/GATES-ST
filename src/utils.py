# src/utils.py
import os
import pandas as pd
import numpy as np
import scanpy as sc
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from typing import Tuple
import torch
from torch_geometric.data import Data
import sklearn

def load_and_preprocess_data(config: dict) -> sc.AnnData:
    # 1. 检查并加载计数矩阵
    counts_file = config['data']['counts_file']
    if not os.path.exists(counts_file):
        raise FileNotFoundError(f"计数矩阵文件不存在: {counts_file}")

    try:
        counts = pd.read_csv(counts_file, sep='\t', index_col=0)
    except Exception as e:
        raise ValueError(f"无法读取计数矩阵文件 '{counts_file}'，请检查文件格式是否为制表符分隔的文本文件（TSV），且第一列为基因名。错误详情: {e}")

    # 2. 检查并加载坐标文件
    coor_file = config['data']['coor_file']
    if not os.path.exists(coor_file):
        raise FileNotFoundError(f"空间坐标文件不存在: {coor_file}")

    try:
        coor_df = pd.read_csv(coor_file, sep='\t', header=None)
    except Exception as e:
        raise ValueError(f"无法读取坐标文件 '{coor_file}'，请确保其为制表符分隔的三列文本文件（barcode, x, y），无表头。错误详情: {e}")

    if coor_df.shape[1] < 3:
        raise ValueError(f"坐标文件 '{coor_file}' 至少应包含三列（barcode, x, y），当前只有 {coor_df.shape[1]} 列。")

    coor_df.columns = ['barcode', 'x', 'y']
    coor_df = coor_df.set_index('barcode')

    # 3. 确保索引类型一致（字符串）
    counts.columns = counts.columns.astype(str)
    coor_df.index = coor_df.index.astype(str)

    # 4. 创建 AnnData（cells × genes）
    adata = sc.AnnData(counts.T)
    adata.var_names_make_unique()

    # 5. 对齐表达数据和坐标（取交集）
    common_barcodes = adata.obs_names.intersection(coor_df.index)
    if len(common_barcodes) == 0:
        raise ValueError("计数矩阵的列名（细胞barcode）与坐标文件的barcode无交集，请检查两者是否匹配。")

    adata = adata[common_barcodes, :]
    coor_df = coor_df.loc[common_barcodes]

    # 6. 设置空间坐标 [x, y]
    adata.obsm["spatial"] = coor_df[['x', 'y']].values

    # 7. QC 和标准化（保持不变）
    sc.pp.calculate_qc_metrics(adata, inplace=True)

    # 8. 可选：加载已使用的barcode列表
    if 'used_barcodes_file' in config['data'] and config['data']['used_barcodes_file']:
        used_barcodes_file = config['data']['used_barcodes_file']
        if not os.path.exists(used_barcodes_file):
            raise FileNotFoundError(f"指定的 used_barcodes_file 不存在: {used_barcodes_file}")

        try:
            used_barcode = pd.read_csv(used_barcodes_file, sep='\t', header=None)[0].astype(str)
        except Exception as e:
            raise ValueError(f"无法读取 used_barcodes_file '{used_barcodes_file}'，应为单列文本文件。错误详情: {e}")

        used_barcode = used_barcode[used_barcode.isin(adata.obs_names)]
        if len(used_barcode) == 0:
            raise ValueError("used_barcodes_file 中的 barcode 与当前 AnnData 无交集。")
        adata = adata[used_barcode, :]

    # 9. 后续预处理
    sc.pp.filter_genes(adata, min_cells=50)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=config['model']['n_top_genes'])
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    return adata

def Cal_Spatial_Net(adata, rad_cutoff=None, k_cutoff=None, model='Radius', verbose=True):
    """构建空间邻居网络"""
    assert model in ['Radius', 'KNN']
    if verbose:
        print('------Calculating spatial graph...')
    coor = pd.DataFrame(adata.obsm['spatial'])
    coor.index = adata.obs.index
    coor.columns = ['imagerow', 'imagecol']
    if model == 'Radius':
        nbrs = sklearn.neighbors.NearestNeighbors(radius=rad_cutoff).fit(coor)
        distances, indices = nbrs.radius_neighbors(coor, return_distance=True)
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([it] * indices[it].shape[0], indices[it], distances[it])))
    elif model == 'KNN':
        nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=k_cutoff + 1).fit(coor)
        distances, indices = nbrs.kneighbors(coor)
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([it] * indices.shape[1], indices[it, :], distances[it, :])))
    KNN_df = pd.concat(KNN_list)
    KNN_df.columns = ['Cell1', 'Cell2', 'Distance']
    Spatial_Net = KNN_df.copy()
    Spatial_Net = Spatial_Net.loc[Spatial_Net['Distance'] > 0,]
    id_cell_trans = dict(zip(range(coor.shape[0]), np.array(coor.index)))
    Spatial_Net['Cell1'] = Spatial_Net['Cell1'].map(id_cell_trans)
    Spatial_Net['Cell2'] = Spatial_Net['Cell2'].map(id_cell_trans)
    if verbose:
        print('The graph contains %d edges, %d cells.' % (Spatial_Net.shape[0], adata.n_obs))
        print('%.4f neighbors per cell on average.' % (Spatial_Net.shape[0] / adata.n_obs))
    adata.uns['Spatial_Net'] = Spatial_Net

def Stats_Spatial_Net(adata, save_path=None, show_plot=True):
    """统计并可视化空间网络属性"""
    Num_edge = adata.uns['Spatial_Net']['Cell1'].shape[0]
    Mean_edge = Num_edge / adata.shape[0]
    plot_df = pd.value_counts(pd.value_counts(adata.uns['Spatial_Net']['Cell1']))
    plot_df = plot_df / adata.shape[0]
    if show_plot:
        fig, ax = plt.subplots(figsize=[3, 2])
        plt.ylabel('Percentage')
        plt.xlabel('')
        plt.title('Number of Neighbors (Mean=%.2f)' % Mean_edge)
        ax.bar(plot_df.index, plot_df)
        if save_path:
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

def Cal_Gene_Similarity_Net(adata, k_neighbors=6, metric='cosine', verbose=True):
    """计算基因表达相似度网络"""
    if 'highly_variable' in adata.var.columns:
        adata_Vars = adata[:, adata.var['highly_variable']]
    else:
        adata_Vars = adata
    X = adata_Vars.X.toarray() if hasattr(adata_Vars.X, 'toarray') else adata_Vars.X
    X = X.astype(np.float32)
    # 向量化计算相似度，避免双重循环
    if metric == 'cosine':
        similarity_matrix = cosine_similarity(X)
    elif metric == 'euclidean':
        similarity_matrix = -euclidean_distances(X)
    elif metric == 'pearson':
        X_mean = X.mean(axis=1, keepdims=True)
        X_centered = X - X_mean
        X_std = X.std(axis=1, keepdims=True)
        X_normalized = np.divide(X_centered, X_std, out=np.zeros_like(X_centered), where=X_std!=0)
        similarity_matrix = np.dot(X_normalized, X_normalized.T) / (X.shape[1] - 1)
    else:
        raise ValueError(f"未知的相似度度量: {metric}")
    KNN_list = []
    for i in range(similarity_matrix.shape[0]):
        sorted_indices = np.argsort(-similarity_matrix[i, :])
        closest_cells = sorted_indices[1:k_neighbors + 1]
        closest_distances = similarity_matrix[i, closest_cells]
        KNN_list.append(pd.DataFrame({
            'Cell1': [i] * k_neighbors,
            'Cell2': closest_cells,
            'Distance': closest_distances
        }))
    KNN_df = pd.concat(KNN_list, ignore_index=True)
    id_cell_trans = dict(zip(range(X.shape[0]), adata_Vars.obs.index))
    KNN_df['Cell1'] = KNN_df['Cell1'].map(id_cell_trans)
    KNN_df['Cell2'] = KNN_df['Cell2'].map(id_cell_trans)
    if verbose:
        print('The graph contains %d edges, %d cells.' % (KNN_df.shape[0], adata.n_obs))
        print('%.4f neighbors per cell on average.' % (KNN_df.shape[0] / adata.n_obs))
    adata.uns['Gene_Similarity_Net'] = KNN_df

def build_pyg_graph_from_df(adata: sc.AnnData, net_key: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    从 adata.uns 中的 DataFrame 网络构建 PyG 所需的 edge_index。
    返回: (edge_index, edge_weight)
    """
    df = adata.uns[net_key].copy()
    cell_to_idx = {cell: idx for idx, cell in enumerate(adata.obs_names)}
    df['Cell1'] = df['Cell1'].map(cell_to_idx)
    df['Cell2'] = df['Cell2'].map(cell_to_idx)
    # 移除映射失败的行 (NaN)
    df = df.dropna(subset=['Cell1', 'Cell2'])
    df['Cell1'] = df['Cell1'].astype(int)
    df['Cell2'] = df['Cell2'].astype(int)
    edge_index = torch.tensor([df['Cell1'].values, df['Cell2'].values], dtype=torch.long)
    edge_weight = torch.tensor(df['Distance'].values, dtype=torch.float32)
    return edge_index, edge_weight

def create_pyg_data(adata: sc.AnnData, config: dict) -> Data:
    """
    将 AnnData 对象转换为 PyTorch Geometric 的 Data 对象。
    包含特征矩阵、空间边和基因相似性边。
    """
    # 特征矩阵 (仅高变基因)
    if 'highly_variable' in adata.var.columns:
        X = adata[:, adata.var['highly_variable']].X
    else:
        X = adata.X
    X = torch.tensor(X.toarray() if hasattr(X, 'toarray') else X, dtype=torch.float)
    # 构建两种类型的边
    spatial_edge_index, _ = build_pyg_graph_from_df(adata, 'Spatial_Net')  # 忽略权重（GATConv 不使用）
    gene_sim_edge_index, _ = build_pyg_graph_from_df(adata, 'Gene_Similarity_Net')
    # 创建 PyG Data 对象
    data = Data(
        x=X,
        spatial_edge_index=spatial_edge_index,
        gene_sim_edge_index=gene_sim_edge_index
    )
    return data

def Prune_Spatial_By_GeneSim(adata, k=6, metric='cosine', verbose=True):
    """基于基因表达相似性对空间图进行剪枝：仅保留 top-k 最相似的空间邻居"""
    # 1. 获取空间边
    spatial_net = adata.uns['Spatial_Net'].copy()
    cell_to_idx = {cell: i for i, cell in enumerate(adata.obs_names)}
    idx_to_cell = {i: cell for cell, i in cell_to_idx.items()}

    # 2. 计算基因相似度矩阵（仅高变基因）
    if 'highly_variable' in adata.var:
        X = adata[:, adata.var['highly_variable']].X
    else:
        X = adata.X
    X = X.toarray() if hasattr(X, 'toarray') else X
    X = X.astype(np.float32)

    if metric == 'cosine':
        sim = cosine_similarity(X)
    else:
        raise NotImplementedError("目前仅支持 cosine 相似度用于剪枝")

    # 3. 对每个细胞的空间邻居，按基因相似度选 top-k
    pruned_edges = []
    grouped = spatial_net.groupby('Cell1')
    for cell1, group in grouped:
        neighbors = group['Cell2'].values
        if len(neighbors) == 0:
            continue
        idx1 = cell_to_idx[cell1]
        neighbor_indices = [cell_to_idx[c] for c in neighbors if c in cell_to_idx]
        if not neighbor_indices:
            continue
        sims = sim[idx1, neighbor_indices]
        top_k_idx = np.argsort(-sims)[:k]
        top_neighbors = [idx_to_cell[neighbor_indices[i]] for i in top_k_idx]
        for cell2 in top_neighbors:
            pruned_edges.append((cell1, cell2))

    # 4. 更新 Spatial_Net
    pruned_df = pd.DataFrame(pruned_edges, columns=['Cell1', 'Cell2'])
    pruned_df['Distance'] = 1.0  # 距离不再重要，可设为1
    adata.uns['Spatial_Net'] = pruned_df

    if verbose:
        print(f"After pruning, spatial graph has {pruned_df.shape[0]} edges.")


def build_graphs(adata, config):
    """根据 graph_strategy 构建图"""
    strategy = config.get('graph_strategy', 'spatial-similarity')
    rad = config['model']['rad_cutoff']
    k = config['model']['k_neighbors']
    metric = config['model']['similarity_metric']

    if strategy == "spatial-prune":
        Cal_Spatial_Net(adata, rad_cutoff=rad, model='Radius', verbose=True)
        Prune_Spatial_By_GeneSim(adata, k=k, metric=metric, verbose=True)
        # 不构建 Gene_Similarity_Net，alpha 应设为 0
    elif strategy == "spatial-similarity":
        Cal_Spatial_Net(adata, rad_cutoff=rad, model='Radius', verbose=True)
        Cal_Gene_Similarity_Net(adata, k_neighbors=k, metric=metric, verbose=True)
    elif strategy == "prune-similarity":
        Cal_Spatial_Net(adata, rad_cutoff=rad, model='Radius', verbose=True)
        Prune_Spatial_By_GeneSim(adata, k=k, metric=metric, verbose=True)
        Cal_Gene_Similarity_Net(adata, k_neighbors=k, metric=metric, verbose=True)
    else:
        raise ValueError(f"Unknown graph_strategy: {strategy}")
