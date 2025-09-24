# scripts/run_analysis.py
import os
import yaml
import scanpy as sc
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, davies_bouldin_score
from src.utils import load_and_preprocess_data, Cal_Spatial_Net, Stats_Spatial_Net, Cal_Gene_Similarity_Net, create_pyg_data
from src.gates_model import GATES
from src.trainer import GATESTrainer

def main():
    # --- 1. 加载配置 ---
    with open('../configs/default.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    alpha = config['model']['alpha']
    resolution = config['cluster']['resolution']

    # --- 2. 数据加载与预处理 ---
    print("Loading and preprocessing data...")
    adata = load_and_preprocess_data(config)
    print(f'After filtering: {adata.shape}')

    # --- 3. 构建图网络 ---
    print("Building spatial network...")
    Cal_Spatial_Net(adata, rad_cutoff=config['model']['rad_cutoff'], model='Radius', verbose=True)
    Stats_Spatial_Net(adata, save_path=config['output']['neighbor_stats_plot'].format(alpha=alpha, resolution=resolution), show_plot=False)

    print("Building gene similarity network...")
    Cal_Gene_Similarity_Net(adata, k_neighbors=config['model']['k_neighbors'], metric=config['model']['similarity_metric'], verbose=True)

    # --- 4. 准备 PyG 数据 ---
    print("Preparing PyG data...")
    pyg_data = create_pyg_data(adata, config)

    # --- 5. 初始化并训练模型 ---
    print("Initializing and training GATES model...")
    model = GATES(
        in_channels=adata.var['highly_variable'].sum() if 'highly_variable' in adata.var else adata.n_vars,
        hidden_channels=config['model']['hidden_dims'][0],
        out_channels=config['model']['hidden_dims'][1],
        alpha=alpha
    )

    trainer = GATESTrainer(model, config)
    trainer.train(pyg_data, n_epochs=config['train']['n_epochs'])

    # --- 6. 推理获取嵌入 ---
    print("Inferring embeddings...")
    embeddings = trainer.infer(pyg_data)
    adata.obsm[config['train']['key_added']] = embeddings

    # --- 7. 聚类与降维 ---
    print("Performing clustering and UMAP...")
    sc.pp.neighbors(adata, use_rep=config['train']['key_added'])
    sc.tl.umap(adata)
    sc.tl.louvain(adata, resolution=resolution)

    # --- 8. 评估聚类效果 ---
    umap_embedding = adata.obsm['X_umap']
    louvain_labels = adata.obs['louvain'].astype(int)

    sc_score = silhouette_score(umap_embedding, louvain_labels)
    db_score = davies_bouldin_score(umap_embedding, louvain_labels)

    print(f'Silhouette Coefficient: {sc_score:.4f}')
    print(f'Davies-Bouldin Index: {db_score:.4f}')

    # --- 9. 可视化 ---
    print("Generating plots...")
    crop_coord = config['output']['spatial_plot_crop']

    plt.rcParams["figure.figsize"] = (5, 4)
    sc.pl.embedding(adata, basis="spatial", color="louvain", crop_coord=crop_coord, s=6, show=False, title=f'Ours SC{sc_score:.2f} DB{db_score:.2f}')
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()
