import os
import argparse
import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

# 导入模块
from dataset import SpatialSequenceDataset, KEY_PCA
from model import SpatialTransformer
from utils import set_global_seed, reconstruction_loss, kl_rgb_loss, save_visualization

# ================= 配置 =================
DEFAULT_PATH = "/home/yangqx/YYY/LLM/dataset/xenium_he_clustered.h5ad"
OUT_DIR = "./output_transformer_pca"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default=DEFAULT_PATH)
    parser.add_argument("--out_dir", type=str, default=OUT_DIR)

    # 数据参数
    parser.add_argument("--k_spatial", type=int, default=6)
    parser.add_argument("--k_feature", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=512)

    # 模型参数
    parser.add_argument("--dim_pca", type=int, default=50)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)

    # 训练参数
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--mask_ratio", type=float, default=0.5, help="Mask中心点的概率")

    # Loss 权重
    parser.add_argument("--w_recon", type=float, default=10.0)
    parser.add_argument("--w_vis", type=float, default=1.0)

    return parser.parse_args()


def run_inference(model, dataloader, device):
    """
    全量推理：返回 RGB(z) 和 高维特征(h_fuse)
    """
    model.eval()
    all_z = []
    all_h = []  # [新增] 用于收集隐变量

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference", leave=False):
            # 搬运数据
            seq_pca = batch['seq_pca'].to(device)
            rel_coords = batch['rel_coords'].to(device)

            # 前向传播 (无 Mask)
            out = model(seq_pca, rel_coords)

            # 收集 z (视觉头输出, 3维)
            z = out['visual']['z'].cpu().numpy()
            all_z.append(z)

            # [新增] 收集 h_fuse (隐变量, 128维)
            h = out['h_fuse'].cpu().numpy()
            all_h.append(h)

    # 拼装
    full_z = np.concatenate(all_z, axis=0)
    full_h = np.concatenate(all_h, axis=0)  # [新增]

    return full_z, full_h


def main():
    args = parse_args()
    set_global_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[Init] Output Dir: {args.out_dir}")

    # 1. Dataset & Dataloader
    # -----------------------
    dataset = SpatialSequenceDataset(
        h5ad_path=args.path,
        k_spatial=args.k_spatial,
        k_feature=args.k_feature
    )

    # 训练加载器 (打乱)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)

    # 推理加载器 (不打乱，用于可视化)
    # num_workers=0 避免多进程顺序混乱风险 (虽然理论上没事)
    infer_loader = DataLoader(dataset, batch_size=args.batch_size * 2, shuffle=False, num_workers=4)

    # 获取全局坐标用于画图
    full_coords = dataset.coords  # numpy array

    # 2. Model
    # --------
    model = SpatialTransformer(
        dim_pca=args.dim_pca,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.layers,
        k_spatial=args.k_spatial
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 3. Training Loop
    # ----------------
    print(f"[Train] Start training for {args.epochs} epochs...")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_recon_loss = 0
        total_vis_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)

        for batch in pbar:
            # 数据搬运
            seq_pca = batch['seq_pca'].to(device)  # (B, 17, 50)
            rel_coords = batch['rel_coords'].to(device)  # (B, 17, 2)
            target_pca = batch['target_pca'].to(device)  # (B, 50)
            target_mu = batch['target_rgb_mu'].to(device)
            target_var = batch['target_rgb_var'].to(device)

            # --- Masking Logic ---
            # 以 mask_ratio 的概率将中心点 (Index 0) 置零
            # 生成一个 (B, 1, 1) 的 mask
            B, L, D = seq_pca.shape
            mask_indices = torch.rand(B, device=device) < args.mask_ratio

            # 克隆一下避免修改原数据 (虽然这里也没关系)
            masked_input = seq_pca.clone()
            # 将 mask 为 True 的样本的第 0 个位置置零
            masked_input[mask_indices, 0, :] = 0.0

            # --- Forward ---
            out = model(masked_input, rel_coords)

            # --- Loss ---
            # 1. 重构损失 (MSE)
            loss_r = reconstruction_loss(out['pca_recon'], target_pca)

            # 2. 视觉损失 (KL)
            loss_v = kl_rgb_loss(
                out['visual']['mu'], out['visual']['logvar'],
                target_mu, target_var
            )

            loss = (args.w_recon * loss_r) + (args.w_vis * loss_v)

            # --- Backward ---
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪防爆炸
            optimizer.step()

            total_recon_loss += loss_r.item()
            total_vis_loss += loss_v.item()

            pbar.set_postfix({'Recon': loss_r.item(), 'Vis': loss_v.item()})

        scheduler.step()

        # --- Visualization & Logging ---
        if epoch % 10 == 0 or epoch == 1:
            avg_r = total_recon_loss / len(train_loader)
            avg_v = total_vis_loss / len(train_loader)
            print(f"Epoch {epoch} | Recon: {avg_r:.4f} | Vis: {avg_v:.4f}")

            # [修改] 运行推理，接收两个返回值 (z 和 h)
            full_z, full_h = run_inference(model, infer_loader, device)

            # [修改] 调用可视化，传入 full_h
            save_visualization(
                full_coords=full_coords,
                full_z_rgb=full_z,
                full_h_fuse=full_h,  # <--- 这里是补上的参数
                epoch=epoch,
                base_dir=args.out_dir,
                cluster_k=17  # 你可以根据需要修改聚类类别数
            )

            # 保存模型
            torch.save(model.state_dict(), os.path.join(args.out_dir, "last_model.pth"))


if __name__ == "__main__":
    main()
