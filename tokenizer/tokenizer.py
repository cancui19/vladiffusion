import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors
import torch
import math
from torch.nn import functional as F
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os
import wandb

import tqdm

from nuscenes_dataset import NuScenesDataset

torch.manual_seed(42)


def get_xy_points(dataset: NuScenesDataset):
    """
    Get all the things. Same as "fut_waypoints_ego" from gen_data.py

    Returns:
        (N, 2) array of all points in the ego vehicle frame
    """
    xy_points = []

    # Iterate through every sample in the dataset and collect future waypoints
    # in the ego vehicle frame. Equivalent conceptually to the fut_waypoints_ego
    # list in gen_data.py (future ego-frame waypoints), but aggregated across
    # the entire dataset for clustering / codebook creation.
    for idx in tqdm.tqdm(range(len(dataset))):
        item = dataset[idx]
        traj = item.get("trajectory", None)
        if traj is None:
            continue
        if traj.shape[0] == 0:
            continue
        # Take only x,y (ignore z) -> (T,2)
        xy = traj[:, :2]
        xy_points.append(xy)

    if not xy_points:
        return np.empty((0, 2), dtype=np.float32)

    all_xy = np.vstack(xy_points).astype(np.float32)
    return all_xy

def save_data():
    load_dotenv()

    VERSION = 'v1.0-trainval'
    DATAROOT = os.getenv("NUSCENES_ROOT")
    dataset = NuScenesDataset(nuscenes_path=DATAROOT, version=VERSION, split='train', future_seconds=10, future_hz=2, get_img_data=False)
    xy_points = get_xy_points(dataset)
    np.save("points_xy.npy", xy_points)

    print(f"Total number of points: {xy_points.shape[0]}")
    plt.figure(figsize=(8, 8))
    plt.scatter(xy_points[:, 0], xy_points[:, 1], s=0.1, alpha=0.5)
    plt.title("Future Waypoints in Ego Vehicle Frame")
    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    plt.axis('equal')
    plt.grid(True)
    plt.savefig("future_waypoints_ego_frame.png", dpi=300)

def get_clusters(points: np.ndarray, num_clusters=2048):
    nbrs = NearestNeighbors(n_neighbors=16).fit(points)
    dists, _ = nbrs.kneighbors(points)
    # larger mean distance => lower density => higher weight
    w = (dists.mean(axis=1) + 1e-6)**2
    w /= w.mean()

    kmeans = MiniBatchKMeans(n_clusters=num_clusters, batch_size=16384, random_state=42)
    kmeans.fit(points, sample_weight=w)
    return kmeans

def viz(kmeans, points: np.ndarray):
    plt.figure(figsize=(8, 8))
    plt.scatter(points[:, 0], points[:, 1], s=0.1, alpha=0.5, label='Data Points')
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='red', s=20, marker='x', label='Cluster Centers')
    plt.title("K-Means Clustering of Future Waypoints")
    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.savefig("kmeans_clusters.png", dpi=450)

def preprocess(points: np.ndarray, kmeans: MiniBatchKMeans):
    """
    Given a set of points/their clusters, normalize this data
    s.t. all points are in the range [-1, 1].
    """
    minx, maxx = points[:, 0].min(), points[:, 0].max()
    miny, maxy = points[:, 1].min(), points[:, 1].max()

    print(f"x range: {minx} to {maxx}")
    print(f"y range: {miny} to {maxy}")

    # Scale and translate points to [-1, 1]
    scale = 2 / (max(maxx - minx, maxy - miny) + 1e-6)
    translate_x = -(maxx + minx) / 2.0
    translate_y = -(maxy + miny) / 2.0
    points_normalized = points.copy()
    points_normalized[:, 0] = (points[:, 0] + translate_x)
    points_normalized[:, 1] = (points[:, 1] + translate_y)
    points_normalized[:, 0] *= scale
    points_normalized[:, 1] *= scale

    # Apply same transformation to cluster centers
    centers = kmeans.cluster_centers_.copy()
    centers[:, 0] = (centers[:, 0] + translate_x) * scale
    centers[:, 1] = (centers[:, 1] + translate_y) * scale
    kmeans.cluster_centers_ = centers
    return points_normalized, kmeans, (scale, translate_x, translate_y)
    

class PointEmbedding(nn.Module):
    def __init__(self, kmeans: MiniBatchKMeans, D=128, K=16, dropout=0.00):
        super().__init__()
        self.register_buffer("C", torch.as_tensor(kmeans.cluster_centers_, dtype=torch.float32))  # (num_clusters, 2)
        self.dropout = nn.Dropout(dropout)

        self.D = D
        self.E = nn.Parameter(nn.init.orthogonal_(torch.empty(self.C.shape[0], self.D)))  # (num_clusters, D)
        self.K = K
        self.tau = 1.0


    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        X: (B, 2) tensor of points
        Returns:
            (B, D) tensor of embeddings
        """
        distances = torch.cdist(X.unsqueeze(1), self.C.unsqueeze(0), p=2).squeeze(1)  # (B, num_clusters)
        knn_dists, knn_indices = torch.topk(distances, self.K, largest=False, dim=1)  # (B, K)

        weights = F.softmax(-knn_dists / self.tau, dim=1)  # (B, K)
        embeddings = self.E[knn_indices]  # (B, K, D)
        weights = weights.unsqueeze(2)  # (B, K, 1)
        weighted_embeddings = (embeddings * weights).sum(dim=1)  # (B, D)

        return self.dropout(weighted_embeddings)
    
class EmbeddingDecoder(nn.Module):
    def __init__(self, kmeans: MiniBatchKMeans, D=128, hidden_dim=96):
        super().__init__()
        self.register_buffer("C", torch.as_tensor(kmeans.cluster_centers_, dtype=torch.float32))  # (num_clusters, 2)
        self.D = D
        self.hidden_dim = hidden_dim

        self.linear = nn.Linear(D, 2)

        self.mlp = nn.Sequential(
            nn.Linear(D, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        embeddings: (B, D) tensor of embeddings
        Returns:
            (B, 2) tensor of reconstructed points
        """
        recon_points = self.linear(embeddings) + self.mlp(embeddings)  # (B, 2)
        return recon_points

@torch.no_grad()
def geometry_loss(embeddings: torch.Tensor, points: torch.Tensor,
                  n_pairs=1024, detach_scales=True):
    """
    embeddings: (B, D)
    points: (B, 2)
    """
    B = embeddings.size(0)
    if B < 2:
        return embeddings.new_tensor(0.0)

    i = torch.randint(0, B, (n_pairs,), device=embeddings.device)
    j = torch.randint(0, B, (n_pairs,), device=embeddings.device)

    dE  = torch.norm(embeddings[i] - embeddings[j], dim=1)
    dUV = torch.norm(points[i] - points[j], dim=1)

    # robust scaling (median)
    scaleE  = dE.median() + 1e-9
    scaleUV = dUV.median() + 1e-9
    if detach_scales:
        scaleE = scaleE.detach()
        scaleUV = scaleUV.detach()

    return ((dE / scaleE - dUV / scaleUV) ** 2).mean()

def assign_batch_labels(x: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    """
    x: (B, 2) points (already normalized)
    centers: (K, 2)
    Returns: (B,) cluster index per point
    """
    # (B, K)
    dists = torch.cdist(x, centers, p=2)
    return dists.argmin(dim=1)

def supervised_contrastive_loss(z, labels, temperature=0.07, eps=1e-12):
    z = F.normalize(z, dim=1)
    logits = torch.matmul(z, z.T) / temperature

    logits_mask = torch.ones_like(logits, dtype=torch.bool)
    logits_mask.fill_diagonal_(False)

    labels = labels.unsqueeze(1)
    positive_mask = (labels == labels.T) & logits_mask
    neg_mask = logits_mask & ~positive_mask

    # log-softmax over all non-self pairs
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + eps)

    pos_counts = positive_mask.sum(dim=1)
    valid = pos_counts > 0
    mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1) / pos_counts.clamp_min(1)
    return -(mean_log_prob_pos[valid]).mean()
    
def train_loop(points, kmeans, num_epochs=100, batch_size=512, lr=1e-3, device='cuda'):
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(points).float())
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model = nn.Module()
    model.embedding = PointEmbedding(kmeans).to(device)
    model.decoder = EmbeddingDecoder(kmeans).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse_loss_fn = nn.MSELoss()

    for epoch in range(num_epochs):
        total_loss = 0.0
        total_recon = 0.0
        total_geom = 0.0
        # total_triplet = 0.0
        grad_norms = []

        # freeze decoder for first 10 epochs
        if epoch < 10:
            for p in model.decoder.parameters(): p.requires_grad = False
        else:
            for p in model.decoder.parameters(): p.requires_grad = True

        for batch in dataloader:        
            x = batch[0].to(device)  # (B, 2)

            optimizer.zero_grad()
            embeddings = model.embedding(x)  # (B, D)
            recon_points = model.decoder(embeddings)  # (B, 2)

            mse_loss = mse_loss_fn(recon_points, x)
            geom_loss = geometry_loss(embeddings, x, n_pairs=2048)
            
            # with torch.no_grad():
            #     labels = assign_batch_labels(x, model.embedding.C)  # (B,)
            # triplet_loss = supervised_contrastive_loss(embeddings, labels)
            
            loss = mse_loss + 0.1 * geom_loss # + 0.1 * triplet_loss
            loss.backward()

            grad_norms.append(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0))
            optimizer.step()

            bs = x.size(0)
            total_loss += loss.item() * bs
            total_recon += mse_loss.item() * bs
            total_geom += geom_loss.item() * bs
            # total_triplet += triplet_loss.item() * bs

        denom = len(dataset)
        avg_loss = total_loss / denom
        avg_recon = total_recon / denom
        avg_geom = total_geom / denom
        # avg_triplet = total_triplet / denom
        avg_grad_norm = sum(grad_norms) / len(grad_norms) if grad_norms else 0.0
        print(f"{epoch+1}/{num_epochs} | total {avg_loss:.6f} | recon {avg_recon:.6f} | geom {avg_geom:.6f} | grad {avg_grad_norm:.4f}")
        wandb.log({
            "epoch": epoch+1,
            "loss": avg_loss,
            "loss_recon": avg_recon,
            "loss_geom": avg_geom,
            # "loss_triplet": avg_triplet,
            "grad_norm": avg_grad_norm,
        })

    return model

if __name__ == "__main__":
    # save_data() # Uncomment to re-save data from NuScenes
    points = np.load("points_xy.npy")
    kmeans = get_clusters(points, num_clusters=2048)
    # viz(kmeans, points)

    points_normalized, kmeans, transform = preprocess(points, kmeans)
    viz(kmeans, points_normalized)

    # train
    wandb.init(project="vladiffusion", name="tokenizer_training")
    model = train_loop(points_normalized, kmeans, num_epochs=100, batch_size=512, lr=1e-5, device='cuda')
    torch.save(model.state_dict(), "tokenizer_model.pth")
    wandb.finish()
