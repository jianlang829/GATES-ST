# scripts/run_analysis.py
import os
import yaml
import scanpy as sc
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, davies_bouldin_score
from src.utils import load_and_preprocess_data, Cal_Spatial_Net, Stats_Spatial_Net, Cal_Gene_Similarity_Net, create_pyg_data
from src.gates_model import GATES
from src.trainer import GATESTrainer
import squidpy as sq

def main():
    with open('./configs/default.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # === 新增：定义缓存路径 ===
    cache_dir = "./cache"
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(
        cache_dir,
        f"preprocessed_adata_rad{config['model']['rad_cutoff']}_k{config['model']['k_neighbors']}.h5ad"
    )

    print('------------------')
    alpha = config['model']['alpha']
    resolution = config['cluster']['resolution']

    # === 检查缓存是否存在 ===
    if os.path.exists(cache_file):
        print(f"Loading cached preprocessed data from {cache_file}...")
        adata = sc.read_h5ad(cache_file)
        print(f'Loaded cached data: {adata.shape}')
    else:
        print("Loading and preprocessing data...")
        adata = load_and_preprocess_data(config)
        print(f'After filtering: {adata.shape}')

        print("Building spatial network...")
        Cal_Spatial_Net(adata, rad_cutoff=config['model']['rad_cutoff'], model='Radius', verbose=True)
        Stats_Spatial_Net(adata, save_path=config['output']['neighbor_stats_plot'].format(alpha=alpha, resolution=resolution), show_plot=False)

        print("Building gene similarity network...")
        Cal_Gene_Similarity_Net(
            adata,
            k_neighbors=config['model']['k_neighbors'],
            metric=config['model']['similarity_metric'],
            verbose=True
        )

        # === 保存到缓存 ===
        print(f"Saving preprocessed data to cache: {cache_file}")
        adata.write_h5ad(cache_file)
    print("Preparing PyG data...")
    pyg_data = create_pyg_data(adata, config)
    # 确保输入维度正确：高变基因数量
    in_channels = adata.var['highly_variable'].sum() if 'highly_variable' in adata.var else adata.n_vars
    hidden_channels = config['model']['hidden_dims'][0]
    out_channels = config['model']['hidden_dims'][1]
    print(f"Model input dim: {in_channels}, hidden: {hidden_channels}, output: {out_channels}")
    print("Initializing and training GATES model...")
    print("hidden_dims:", config['model']['hidden_dims'])
    print("type:", type(config['model']['hidden_dims']))
    model = GATES(
        in_channels = int(adata.var['highly_variable'].sum()) if 'highly_variable' in adata.var else int(adata.n_vars),
        hidden_channels = int(config['model']['hidden_dims'][0]),
        out_channels = int(config['model']['hidden_dims'][1]),
        alpha=alpha
    )
    trainer = GATESTrainer(model, config)
    trainer.train(pyg_data, n_epochs=config['train']['n_epochs'])
    print("Inferring embeddings...")
    embeddings = trainer.infer(pyg_data)
    adata.obsm[config['train']['key_added']] = embeddings
    print("Performing clustering and UMAP...")
    sc.pp.neighbors(adata, use_rep=config['train']['key_added'])
    sc.tl.umap(adata)
    sc.tl.louvain(adata, resolution=resolution)
    adata.obs['louvain'] = adata.obs['louvain'].astype('category')  # 👈 新增这行！
    louvain_labels = adata.obs['louvain'].astype(int)

    # 修正：使用 GATES 嵌入计算指标，而非 UMAP
    sc_score = silhouette_score(embeddings, louvain_labels)
    db_score = davies_bouldin_score(embeddings, louvain_labels)

    print(f'Silhouette Coefficient: {sc_score:.4f}')
    print(f'Davies-Bouldin Index: {db_score:.4f}')
    print("Generating plots...")
    crop_coord = config['output']['spatial_plot_crop']
    plt.rcParams["figure.figsize"] = (5, 4)

    print("Generating plots...")
    crop_coord = config['output']['spatial_plot_crop']
    plt.rcParams["figure.figsize"] = (5, 4)

    # 优先从 config 获取完整路径，否则用默认名 + 输出目录
    output_path = config['output'].get('spatial_plot_path')
    if output_path is None:
        output_dir = config['output'].get('dir', '.')
        output_path = os.path.join(output_dir, "spatial_louvain.png")

    # 确保目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print("Spatial coordinates range:")
    print("x:", adata.obsm["spatial"][:, 0].min(), "to", adata.obsm["spatial"][:, 0].max())
    print("y:", adata.obsm["spatial"][:, 1].min(), "to", adata.obsm["spatial"][:, 1].max())
    print("Crop coord:", crop_coord)
    print("Spatial coordinates shape:", adata.obsm["spatial"].shape)
    print("First few spatial coords:\n", adata.obsm["spatial"][:5])
    print("Louvain labels info:")
    print("Unique labels:", adata.obs['louvain'].unique())
    print("Number of NaNs:", adata.obs['louvain'].isna().sum())
    print("Data type:", adata.obs['louvain'].dtype)

    # 替换原来的 sc.pl.spatial 调用
    sq.pl.spatial_scatter(
        adata,
        color="louvain",
        shape=None,  # 不显示组织轮廓（可选）
        size=20,     # 对应 spot_size
        title=f'Ours SC{sc_score:.2f} DB{db_score:.2f}',
        save=output_path  # 自动保存，无需 plt.savefig
    )

    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Spatial plot saved to: {output_path}")
    success_art = r"""


    ___ _   _  ___ ___ ___  ___ ___
    / __| | | |/ __/ __/ _ \/ __/ __|
    \__ \ |_| | (_| (_|  __/\__ \__ \
    |___/\__,_|\___\___\___||___/___/


    """

    print("\033[1;32m" + success_art + "\033[0m")
    print("\033[1;36m✨ Analysis completed successfully! All results saved. ✨\033[0m")
    print("\033[1;33m🎉 You're awesome! Go celebrate with a coffee! ☕\033[0m")

if __name__ == "__main__":
    main()
