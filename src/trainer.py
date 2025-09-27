import torch
import torch.optim as optim
from tqdm import tqdm
from .gates_model import ImprovedGATES
from typing import Dict, Any
import torch.nn.functional as F

class ImprovedGATESTrainer:
    """增强版训练器，吸收TF版本多目标优化思想"""

    def __init__(self, model: ImprovedGATES, config: Dict[str, Any]):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['train']['lr'],
            weight_decay=config['train']['weight_decay']
        )

        # 从TF版本学习的多目标损失权重
        self.lambda_spatial = config['train'].get('lambda_spatial', 0.1)
        self.lambda_att = config['train'].get('lambda_att', 0.01)
        self.loss_history = []

    def calc_spatial_consistency(self, embeddings, spatial_edge_index):
        """TF版本的空间一致性损失"""
        z_i = embeddings[spatial_edge_index[0]]
        z_j = embeddings[spatial_edge_index[1]]
        return F.mse_loss(z_i, z_j)

    def attention_regularization(self, attention_weights):
        """TF版本的注意力正则化"""
        reg_loss = 0
        for key, (edge_index, att_weights) in attention_weights.items():
            # 鼓励注意力权重分布均匀（避免过度集中）
            entropy = -torch.sum(att_weights * torch.log(att_weights + 1e-8))
            reg_loss += entropy
        return -reg_loss  # 最大化熵

    def train(self, data, n_epochs: int) -> None:
        self.model.train()
        data = data.to(self.device)

        for epoch in tqdm(range(n_epochs), desc="Training"):
            self.optimizer.zero_grad()

            # 前向传播（返回注意力权重）
            embeddings, recon, attention_weights = self.model(
                data.x, data.spatial_edge_index, data.gene_sim_edge_index
            )

            # 多目标损失函数（TF版本思想）
            recon_loss = F.mse_loss(recon, data.x)
            spatial_loss = self.calc_spatial_consistency(embeddings, data.spatial_edge_index)
            att_reg_loss = self.attention_regularization(attention_weights)

            # 综合损失（TF版本的多目标优化）
            total_loss = (recon_loss +
                         self.lambda_spatial * spatial_loss +
                         self.lambda_att * att_reg_loss)

            total_loss.backward()
            self.optimizer.step()
            self.loss_history.append(total_loss.item())

            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Recon={recon_loss:.4f}, "
                      f"Spatial={spatial_loss:.4f}, AttReg={att_reg_loss:.4f}, "
                      f"Total={total_loss:.4f}")

    def infer(self, data):
        """推理并返回注意力权重用于分析"""
        self.model.eval()
        data = data.to(self.device)
        with torch.no_grad():
            embeddings = self.model.encode(data.x, data.spatial_edge_index, data.gene_sim_edge_index)
        return embeddings.cpu().numpy(), self.model.attention_weights
