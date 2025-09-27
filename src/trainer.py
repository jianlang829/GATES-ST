# src/trainer.py
import torch
import torch.optim as optim
from tqdm import tqdm
from .gates_model import GATES
from typing import Dict, Any
import torch.nn.functional as F
from torch_geometric.data import Data

class GATESTrainer:
    """
    GATES 模型训练器
    """
    def __init__(self, model: GATES, config: Dict[str, Any]):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['train']['lr'],
            weight_decay=config['train']['weight_decay']
        )
        self.loss_history = []

    def train(self, data: Data, n_epochs: int) -> None:
        """
        训练模型。
        Args:
            data: PyG Data 对象（不是 DataLoader）
            n_epochs: 训练轮数
        """
        self.model.train()
        data = data.to(self.device)
        spatial_edge_index = data.spatial_edge_index  # [2, E]

        # 从配置中读取空间正则化权重（默认为 0.0，即不启用）
        lambda_spatial = self.config['train'].get('lambda_spatial', 0.0)

        for epoch in tqdm(range(n_epochs), desc="Training"):
            self.optimizer.zero_grad()
            embeddings, recon = self.model(data.x, spatial_edge_index, data.gene_sim_edge_index)

            # 1. 重构损失（MSE）
            recon_loss = F.mse_loss(recon, data.x)

            # 2. 空间一致性正则项（仅当 lambda_spatial > 0 时计算）
            spatial_loss = 0.0
            if lambda_spatial > 0.0:
                z_i = embeddings[spatial_edge_index[0]]  # source 节点嵌入
                z_j = embeddings[spatial_edge_index[1]]  # target 节点嵌入
                spatial_loss = F.mse_loss(z_i, z_j)      # ||z_i - z_j||^2 的均值

            # 3. 总损失
            total_loss = recon_loss + lambda_spatial * spatial_loss

            total_loss.backward()
            self.optimizer.step()
            self.loss_history.append(total_loss.item())

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Recon Loss: {recon_loss.item():.4f}", end="")
                if lambda_spatial > 0.0:
                    print(f", Spatial Loss: {spatial_loss.item():.4f}", end="")
                print(f", Total Loss: {total_loss.item():.4f}")

    def infer(self, data: Data) -> torch.Tensor:
        """
        使用训练好的模型进行推理。
        """
        self.model.eval()
        data = data.to(self.device)
        with torch.no_grad():
            embeddings = self.model.encode(data.x, data.spatial_edge_index, data.gene_sim_edge_index)
        return embeddings.cpu().numpy()
