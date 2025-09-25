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
        for epoch in tqdm(range(n_epochs), desc="Training"):
            self.optimizer.zero_grad()
            _, recon = self.model(data.x, data.spatial_edge_index, data.gene_sim_edge_index)
            loss = F.mse_loss(recon, data.x)
            loss.backward()
            self.optimizer.step()
            self.loss_history.append(loss.item())
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    def infer(self, data: Data) -> torch.Tensor:
        """
        使用训练好的模型进行推理。
        """
        self.model.eval()
        data = data.to(self.device)
        with torch.no_grad():
            embeddings = self.model.encode(data.x, data.spatial_edge_index, data.gene_sim_edge_index)
        return embeddings.cpu().numpy()
