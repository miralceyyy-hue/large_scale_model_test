import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


# ==========================================
# 1. 辅助组件：相对位置编码器
# ==========================================
class RelativePositionalEncoder(nn.Module):
    """
    将连续的 2D 相对坐标 (dx, dy) 映射到高维 d_model。
    使用 MLP 学习空间距离的非线性表示。
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model)
        )

    def forward(self, rel_coords):
        # rel_coords: (Batch, Seq_Len, 2)
        return self.mlp(rel_coords)


# ==========================================
# 2. 辅助组件：渲染头 (保留你原有的 VAE 风格)
# ==========================================
class RenderHeadR(nn.Module):
    """
    负责将隐变量 h 映射到 RGB 空间。
    输出 mu, logvar 用于 KL Loss，z 用于可视化。
    """

    def __init__(self, h_dim: int, z_dim: int = 3, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(h_dim, hidden),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Linear(hidden, 2 * z_dim)  # 输出 mu 和 logvar
        )

    def forward(self, h):
        # h: (Batch, h_dim)
        y = self.net(h)
        mu, logvar = y.chunk(2, dim=-1)

        # 限制 logvar 范围，防止训练初期数值爆炸
        logvar = torch.clamp(logvar, min=-5.0, max=3.0)

        # Reparameterization Trick (训练时采样，推理时用均值)
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu

            # Sigmoid 归一化到 [0, 1] 以匹配 RGB
        # 注意：这里对 mu 也做 Sigmoid，确保均值也在合法范围内
        z = torch.sigmoid(z)
        mu = torch.sigmoid(mu)

        return {"mu": mu, "logvar": logvar, "z": z}


# ==========================================
# 3. 核心模型：Spatial Transformer
# ==========================================
class SpatialTransformer(nn.Module):
    def __init__(
            self,
            dim_pca: int = 50,  # 输入 PCA 维度
            d_model: int = 128,  # 内部隐维度 (Transformer width)
            nhead: int = 4,  # Attention Heads
            num_layers: int = 2,  # Transformer Layers
            dim_feedforward: int = 256,
            dropout: float = 0.1,
            k_spatial: int = 6  # 物理邻居数量 (用于确定 Type Embedding)
    ):
        super().__init__()

        self.d_model = d_model
        self.k_s = k_spatial

        # --- A. 嵌入层 (Embeddings) ---
        # 1. 内容投影 (PCA -> d_model)
        self.input_proj = nn.Linear(dim_pca, d_model)

        # 2. 相对位置编码 (Coords -> d_model)
        self.pos_encoder = RelativePositionalEncoder(d_model)

        # 3. 类型/身份编码 (Center=0, Spatial=1, Feature=2)
        # 0: 中心点
        # 1: 空间邻居 (近邻)
        # 2: 特征邻居 (远亲)
        self.type_embedding = nn.Embedding(3, d_model)

        # --- B. 核心编码器 (Core) ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # 关键：输入形状为 (Batch, Seq, Dim)
            norm_first=True  # Pre-Norm 通常收敛更稳
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # --- C. 输出头 (Heads) ---
        # 1. 重构头 (解码回 PCA)
        self.recon_head = nn.Sequential(
            nn.Linear(d_model, dim_pca)
        )

        # 2. 视觉头 (渲染 RGB)
        self.render_head = RenderHeadR(h_dim=d_model, z_dim=3)

        # 初始化
        self._init_weights()

    def _init_weights(self):
        # Xavier 初始化 Linear，Embedding 初始化为 0 或小随机数
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, seq_pca, rel_coords):
        """
        Args:
            seq_pca: (Batch, Seq_Len, 50) - 包含中心点和所有邻居的 PCA
            rel_coords: (Batch, Seq_Len, 2) - 相对中心点的坐标偏移
        Returns:
            Dict: h_fuse, pca_recon, visual
        """
        B, L, _ = seq_pca.shape

        # 1. 构建 Input Embeddings
        # ------------------------
        # (A) Feature Embedding
        e_feat = self.input_proj(seq_pca)  # (B, L, D)

        # (B) Positional Embedding
        e_pos = self.pos_encoder(rel_coords)  # (B, L, D)

        # (C) Type Embedding
        # 构造类型索引张量
        # index 0 -> Type 0 (Center)
        # index 1..k_s -> Type 1 (Spatial)
        # index k_s+1..end -> Type 2 (Feature)
        type_ids = torch.zeros(L, dtype=torch.long, device=seq_pca.device)
        type_ids[1: self.k_s + 1] = 1
        type_ids[self.k_s + 1:] = 2

        # 扩展到 Batch 维度 (虽然 Embedding 可以自动广播，但为了稳妥)
        e_type = self.type_embedding(type_ids).unsqueeze(0).expand(B, -1, -1)  # (B, L, D)

        # 叠加所有信息
        x = e_feat + e_pos + e_type

        # 2. Transformer 交互 (Niche Modeling)
        # ------------------------
        # Self-Attention 在这里发生：中心点与邻居交互，邻居之间也交互
        x_transformed = self.transformer(x)  # (B, L, D)

        # 3. 聚合/提取 (Aggregation)
        # ------------------------
        # 我们只关心中心点 (Index 0) 更新后的状态
        # 因为在 Self-Attention 中，位置 0 已经聚合了整个序列的信息
        h_center = x_transformed[:, 0, :]  # (B, D)

        # 4. 输出预测 (Heads)
        # ------------------------
        # (A) 重构 PCA (Self-Supervised Task)
        pca_recon = self.recon_head(h_center)  # (B, 50)

        # (B) 视觉约束 (Visual Task)
        visual_out = self.render_head(h_center)

        return {
            "h_fuse": h_center,
            "pca_recon": pca_recon,
            "visual": visual_out
        }