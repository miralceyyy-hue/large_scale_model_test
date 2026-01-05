import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import umap
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# ==========================================
# 1. 基础配置
# ==========================================
# 20色调色板，用于聚类显示
TAB20 = np.array(plt.get_cmap('tab20').colors)


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ==========================================
# 2. 损失函数
# ==========================================

def reconstruction_loss(pred_pca, target_pca):
    return F.mse_loss(pred_pca, target_pca)


def kl_rgb_loss(mu_pred, logvar_pred, mu_target, var_target, eps=1e-6):
    var_pred = torch.exp(logvar_pred)
    var_obs = torch.clamp(var_target, min=eps)
    log_ratio = torch.log(var_obs + eps) - torch.log(var_pred + eps)
    frac = (var_pred + (mu_pred - mu_target) ** 2) / (var_obs + eps)
    kl = 0.5 * (log_ratio + frac - 1.0).sum(dim=-1)
    return kl.mean()


# ==========================================
# 3. 可视化工具
# ==========================================

def _z_to_uint8_by_percentile(z_all: np.ndarray, p_lo=1.0, p_hi=99.0) -> np.ndarray:
    """动态拉伸对比度"""
    lo = np.percentile(z_all, p_lo, axis=0, keepdims=True)
    hi = np.percentile(z_all, p_hi, axis=0, keepdims=True)
    rng = np.maximum(hi - lo, 1e-8)
    z01 = np.clip((z_all - lo) / rng, 0.0, 1.0)
    return (np.round(z01 * 255.0)).astype(np.uint8)


def _plot_scatter(coords, colors, save_path, title, point_size=5.0, is_discrete=False):
    """通用散点图绘制"""
    plt.figure(figsize=(10, 10), dpi=150)

    # 随机打乱绘制顺序，防止某一类遮挡另一类
    order = np.arange(len(coords))
    np.random.shuffle(order)
    coords = coords[order]

    if is_discrete:
        # 离散聚类标签 -> 颜色
        labels = colors[order].astype(int)
        unique_labels = np.unique(labels)
        for lbl in unique_labels:
            mask = (labels == lbl)
            c = TAB20[lbl % len(TAB20)]
            plt.scatter(coords[mask, 0], coords[mask, 1], c=[c], s=point_size, label=f"{lbl}", linewidth=0)
        # 如果类别不多，显示图例
        if len(unique_labels) <= 15:
            plt.legend(markerscale=3, bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        # 连续 RGB 颜色
        c = colors[order]
        plt.scatter(coords[:, 0], coords[:, 1], c=c, s=point_size, marker='.', linewidth=0)

    plt.gca().set_aspect('equal', adjustable='box')
    plt.axis('off')
    if "UMAP" not in title:
        plt.gca().invert_yaxis()  # 空间坐标通常反转Y轴

    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_visualization(
        full_coords: np.ndarray,
        full_z_rgb: np.ndarray,
        full_h_fuse: np.ndarray,  # [新增] 高维隐变量
        epoch: int,
        base_dir: str,
        cluster_k: int = 7,  # [新增] 聚类数
        point_size: float = 5.0
):
    """
    保存可视化结果：
    1. RGB 重构图 (Spatial)
    2. UMAP 投影图 (Latent Space)
    3. 聚类图 (Spatial + UMAP)
    """
    epoch_dir = os.path.join(base_dir, f"epoch_{epoch:03d}")
    os.makedirs(epoch_dir, exist_ok=True)

    # --- 1. 保存 Raw Data ---
    np.save(os.path.join(epoch_dir, "z_rgb.npy"), full_z_rgb)
    np.save(os.path.join(epoch_dir, "h_fuse.npy"), full_h_fuse)

    # --- 2. 绘制 RGB 图 (所见即所得) ---
    rgb_uint8 = _z_to_uint8_by_percentile(full_z_rgb, p_lo=0.5, p_hi=99.5)
    rgb_norm = rgb_uint8 / 255.0

    _plot_scatter(
        full_coords, rgb_norm,
        os.path.join(epoch_dir, "spatial_rgb.png"),
        title=f"Epoch {epoch} | Spatial RGB",
        point_size=point_size, is_discrete=False
    )

    # --- 3. 聚类分析 (KMeans) ---
    print(f"  [Vis] Running KMeans (k={cluster_k})...")
    # 为了速度，可以直接在 h_fuse 上跑 KMeans
    kmeans = KMeans(n_clusters=cluster_k, n_init=3, random_state=42)
    labels = kmeans.fit_predict(full_h_fuse)

    # 保存标签
    np.save(os.path.join(epoch_dir, "labels.npy"), labels)

    # 绘制空间聚类图
    _plot_scatter(
        full_coords, labels,
        os.path.join(epoch_dir, "spatial_cluster.png"),
        title=f"Epoch {epoch} | Spatial Clusters (k={cluster_k})",
        point_size=point_size, is_discrete=True
    )

    # --- 4. UMAP 分析 (Latent Space Structure) ---
    # 为了节省时间，可以每隔 50 轮才跑一次 UMAP，或者每次都跑
    # 这里默认每次都跑，但先用 PCA 降维加速
    print(f"  [Vis] Running UMAP...")
    if full_h_fuse.shape[1] > 50:
        pca = PCA(n_components=30)
        h_pca = pca.fit_transform(full_h_fuse)
    else:
        h_pca = full_h_fuse

    reducer = umap.UMAP(n_components=2, n_jobs=-1, random_state=42)
    umap_emb = reducer.fit_transform(h_pca)

    # 绘制 UMAP (按聚类上色)
    _plot_scatter(
        umap_emb, labels,
        os.path.join(epoch_dir, "umap_cluster.png"),
        title=f"Epoch {epoch} | UMAP Latent",
        point_size=2.0, is_discrete=True
    )
