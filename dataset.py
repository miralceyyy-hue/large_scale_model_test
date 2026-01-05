import os
import torch
import numpy as np
import scanpy as sc
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import Dataset
from tqdm import tqdm

# ================= 配置参数 =================
# 仅保留 PCA 和 RGB
KEY_PCA = 'X_pca'  # 原始数据的低维表征 (50维)


# KEY_FM 已移除

class SpatialSequenceDataset(Dataset):
    def __init__(
            self,
            h5ad_path: str,
            k_spatial: int = 6,  # 物理邻居数 (捕捉微环境结构)
            k_feature: int = 10,  # 特征邻居数 (基于 PCA 相似度，用于去噪)
            k_candidate: int = 100,  # 搜索范围
    ):
        super().__init__()
        self.path = h5ad_path
        self.k_s = k_spatial
        self.k_f = k_feature
        self.k_c = k_candidate

        print(f"[Dataset] Loading data from {os.path.basename(h5ad_path)}...")
        self.adata = sc.read(h5ad_path)
        self.n_obs = self.adata.n_obs

        # 1. 准备数据矩阵
        # -------------------------------------------
        # (A) 空间坐标
        if 'spatial' in self.adata.obsm:
            self.coords = self.adata.obsm['spatial'].astype(np.float32)
        else:
            # 兼容 Xenium 的 centroids
            if 'x_centroid' in self.adata.obs:
                self.coords = self.adata.obs[['x_centroid', 'y_centroid']].values.astype(np.float32)
            else:
                raise KeyError("Cannot find coordinates (obsm['spatial'] or x/y_centroid)")

        # (B) PCA 特征 (既做输入，也做 Target，也做构图依据)
        if KEY_PCA not in self.adata.obsm:
            raise KeyError(f"{KEY_PCA} not found. Please run PCA first.")
        self.pca_data = self.adata.obsm[KEY_PCA].astype(np.float32)

        # (C) RGB 目标 (用于视觉 Loss)
        try:
            r = self.adata.obs['rgb_mean_R'].values
            g = self.adata.obs['rgb_mean_G'].values
            b = self.adata.obs['rgb_mean_B'].values
            self.rgb_mean = np.stack([r, g, b], axis=1).astype(np.float32)

            vr = self.adata.obs['rgb_var_R'].values
            vg = self.adata.obs['rgb_var_G'].values
            vb = self.adata.obs['rgb_var_B'].values
            self.rgb_var = np.stack([vr, vg, vb], axis=1).astype(np.float32)

            # 归一化 RGB 到 0-1
            if self.rgb_mean.max() > 1.1:
                self.rgb_mean /= 255.0
                self.rgb_var /= (255.0 ** 2)
        except KeyError:
            raise KeyError("Run preprocess_rgb_single.py first to generate RGB stats.")

        # 2. 构建混合图 (逻辑不变)
        # -------------------------------------------
        self.neighbor_indices = self._build_hybrid_graph()

        print(f"[Dataset] Ready. Total cells: {self.n_obs}")

    def _build_hybrid_graph(self):
        """
        构建混合邻居索引：
        1. 空间邻居 (Spatial): 距离最近的前 K_s 个
        2. 特征邻居 (Feature): 在 K_c 范围内，PCA Cosine 相似度最高的前 K_f 个
        """
        print(f"[Graph] Building Hybrid Graph (Spatial={self.k_s}, Feature={self.k_f})...")

        # 步骤 A: 大范围空间搜索
        nbrs_engine = NearestNeighbors(n_neighbors=self.k_c + 1, algorithm='kd_tree', n_jobs=-1)
        nbrs_engine.fit(self.coords)
        spatial_dists, spatial_indices = nbrs_engine.kneighbors(self.coords)

        # 结果容器 (N, 1 + Ks + Kf)
        final_indices = np.zeros((self.n_obs, 1 + self.k_s + self.k_f), dtype=np.int64)
        final_indices[:, 0] = np.arange(self.n_obs)  # 第0列是自己

        # 步骤 B: 批量填充
        batch_size = 4096

        for i in tqdm(range(0, self.n_obs, batch_size), desc="Computing Neighbors"):
            end = min(i + batch_size, self.n_obs)

            # 1. 空间邻居: 直接取最近的 (排除自己)
            # spatial_indices[:, 0] 是自己，所以取 1:k_s+1
            batch_spatial_nbrs = spatial_indices[i:end, 1: self.k_s + 1]
            final_indices[i:end, 1: 1 + self.k_s] = batch_spatial_nbrs

            # 2. 特征邻居: 在 candidates 里找最像的
            # 取出中心点特征
            center_feats = self.pca_data[i:end].reshape(-1, 1, self.pca_data.shape[1])

            # 取出候选点索引 (排除自己)
            candidate_idx = spatial_indices[i:end, 1:]
            # 取出候选点特征
            candidate_feats = self.pca_data[candidate_idx]

            # 计算 Cosine 相似度
            center_norm = center_feats / (np.linalg.norm(center_feats, axis=2, keepdims=True) + 1e-8)
            cand_norm = candidate_feats / (np.linalg.norm(candidate_feats, axis=2, keepdims=True) + 1e-8)

            # (B, 1, D) @ (B, D, Kc) -> (B, 1, Kc) -> (B, Kc)
            sim_matrix = np.matmul(center_norm, cand_norm.transpose(0, 2, 1)).squeeze(1)

            # 排序取 Top K (从大到小)
            top_k_args = np.argsort(sim_matrix, axis=1)[:, -self.k_f:]
            top_k_args = np.flip(top_k_args, axis=1)

            # 映射回全局索引
            rows = np.arange(end - i)[:, None]
            batch_feature_nbrs = candidate_idx[rows, top_k_args]

            final_indices[i:end, 1 + self.k_s:] = batch_feature_nbrs

        return final_indices

    def __len__(self):
        return self.n_obs

    def __getitem__(self, idx):
        """
        返回单一样本数据 (不含大模型特征)
        """
        # 1. 获取序列索引
        indices = self.neighbor_indices[idx]

        # 2. 抓取 PCA 特征 (Input)
        # 形状: (17, 50)
        seq_pca = self.pca_data[indices]

        # 3. 计算相对坐标 (Positional Encoding)
        # 形状: (17, 2)
        seq_coords = self.coords[indices]
        center_coord = seq_coords[0:1, :]
        relative_coords = seq_coords - center_coord

        # 4. 抓取目标 (Targets)
        # 用于重构 Loss (取中心点)
        target_pca = self.pca_data[idx]
        # 用于视觉 Loss
        target_rgb_mu = self.rgb_mean[idx]
        target_rgb_var = self.rgb_var[idx]

        return {
            "seq_pca": torch.tensor(seq_pca, dtype=torch.float32),
            "rel_coords": torch.tensor(relative_coords, dtype=torch.float32),
            "target_pca": torch.tensor(target_pca, dtype=torch.float32),
            "target_rgb_mu": torch.tensor(target_rgb_mu, dtype=torch.float32),
            "target_rgb_var": torch.tensor(target_rgb_var, dtype=torch.float32),
            "center_idx": torch.tensor(idx, dtype=torch.long)
        }
