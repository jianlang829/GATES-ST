# src/trainer.py
import torch
import torch.optim as optim
from tqdm import tqdm
from .gates_model import GATES
from typing import Dict, Any
import torch.nn.functional as F

class GATESTrainer:
    """
    GATES 模型训练器
    """

    def __init__(self, model: GATES, config: Dict[str, Any]):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # 设置优化器
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['train']['lr'],
            weight_decay=config['train']['weight_decay']
        )

        self.loss_history = []

    def train(self, data: torch.utils.data.DataLoader, n_epochs: int) -> None:
        """
        训练模型。

        Args:
            data: 包含特征和边信息的 PyG Data 对象
            n_epochs: 训练轮数
        """
        self.model.train()
        data = data.to(self.device)

        for epoch in tqdm(range(n_epochs), desc="Training"):
            self.optimizer.zero_grad()

            # 前向传播
            embeddings = self.model(data.x, data.spatial_edge_index, data.gene_sim_edge_index)

            # 计算损失：这里使用简单的自编码损失（重构输入）
            # 在实际应用中，您可以根据需要设计更复杂的损失函数
            loss = F.mse_loss(embeddings, data.x)

            # 反向传播
            loss.backward()
            self.optimizer.step()

            self.loss_history.append(loss.item())

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    def infer(self, data: torch.utils.data.DataLoader) -> torch.Tensor:
        """
        使用训练好的模型进行推理。

        Args:
            data: 包含特征和边信息的 PyG Data 对象

        Returns:
            节点嵌入 [N, out_channels]
        """
        self.model.eval()
        data = data.to(self.device)

        with torch.no_grad():
            embeddings = self.model(data.x, data.spatial_edge_index, data.gene_sim_edge_index)

        return embeddings.cpu().numpy()
