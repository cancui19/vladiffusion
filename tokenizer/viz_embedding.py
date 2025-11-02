import numpy as np
import torch
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tokenizer import PointEmbedding, EmbeddingDecoder, PointTokenizer  # import PointTokenizer
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# --- helper: embed a big grid in chunks to avoid OOM ---
@torch.no_grad()
def embed_points_in_batches(pts_xy, model, device='cuda', batch=16):
    out = []
    T = torch.from_numpy(pts_xy).float().to(device)
    for i in tqdm(range(0, T.shape[0], batch)):
        out.append(model.embedding(T[i:i+batch], K=1).cpu())
    return torch.cat(out, dim=0).numpy()

def visualize_embedding_field(model, transform, grid_res=300, device='cuda',
                              pca_fit_on='centers'):  # or 'train'
    """
    model: with .embedding (your PointEmbedding) already loaded + eval()
    transform: (scale, tx, ty) from preprocess()  [only needed if you want meter ticks]
    grid_res: number of samples per axis
    pca_fit_on: 'centers' = fit PCA on codebook, 'train' = you pass some training embeddings instead
    """
    model.eval()

    # 1) Build normalized grid (your embedding expects normalized coords)
    xs = np.linspace(-1.05, 1.05, grid_res)
    ys = np.linspace(-1.05, 1.05, grid_res)
    X, Y = np.meshgrid(xs, ys)
    grid_xy = np.stack([X.ravel(), Y.ravel()], axis=1).astype(np.float32)

    # 2) Embed
    E = embed_points_in_batches(grid_xy, model, device=device)   # (N, D)
    D = E.shape[1]

    # 3) Pick a fixed 3D linear view for coloring (so colors are comparable anywhere)
    if pca_fit_on == 'centers':
        with torch.no_grad():
            C = model.embedding.C.cpu().numpy()                  # (K,2)
            # embed centers themselves to set a stable basis for coloring
            EC = embed_points_in_batches(C, model, device=device)
        pca = PCA(n_components=3).fit(EC)
    else:
        pca = PCA(n_components=3).fit(E)  # fallback

    E3 = pca.transform(E)                 # (N, 3)
    # normalize to [0,1] for display
    Emin, Emax = E3.min(axis=0), E3.max(axis=0)
    E3n = (E3 - Emin) / (Emax - Emin + 1e-9)
    rgb_img = E3n.reshape(grid_res, grid_res, 3)

    # 4) Finite-difference smoothness maps (approx. |∂E/∂x|, |∂E/∂y|, and total)
    Er = E.reshape(grid_res, grid_res, D)
    # forward differences (pad with zeros to keep same shape)
    dEx = np.zeros((grid_res, grid_res), dtype=np.float32)
    dEy = np.zeros((grid_res, grid_res), dtype=np.float32)

    # Δx
    diff_x = Er[:, 1:, :] - Er[:, :-1, :]
    dEx[:, 1:] = np.linalg.norm(diff_x, axis=2) / (xs[1] - xs[0])
    # Δy
    diff_y = Er[1:, :, :] - Er[:-1, :, :]
    dEy[1:, :] = np.linalg.norm(diff_y, axis=2) / (ys[1] - ys[0])

    dEmag = np.sqrt(dEx**2 + dEy**2)
    L_est = float(dEmag.max())   # Lipschitz upper-bound estimate on this grid

    # 5) Plots (separate figures as requested)
    plt.figure(figsize=(7, 7))
    plt.imshow(rgb_img, extent=[xs.min(), xs.max(), ys.min(), ys.max()], origin='lower', interpolation='nearest')
    plt.title("Embedding field (RGB = PCA(emb))")
    plt.xlabel("x (normalized)")
    plt.ylabel("y (normalized)")
    plt.savefig("imgs/embedding_rgb.png", dpi=300)

    plt.figure(figsize=(7, 7))
    plt.imshow(dEx, extent=[xs.min(), xs.max(), ys.min(), ys.max()], origin='lower')
    plt.title("|∂E/∂x| (finite difference)")
    plt.xlabel("x (normalized)"); plt.ylabel("y (normalized)")
    plt.colorbar(); plt.savefig("imgs/embedding_grad_x.png", dpi=300)

    plt.figure(figsize=(7, 7))
    plt.imshow(dEy, extent=[xs.min(), xs.max(), ys.min(), ys.max()], origin='lower')
    plt.title("|∂E/∂y| (finite difference)")
    plt.xlabel("x (normalized)"); plt.ylabel("y (normalized)")
    plt.colorbar(); plt.savefig("imgs/embedding_grad_y.png", dpi=300)

    plt.figure(figsize=(7, 7))
    plt.imshow(dEmag, extent=[xs.min(), xs.max(), ys.min(), ys.max()], origin='lower')
    plt.title("||∇E|| (Frobenius, via FD)")
    plt.xlabel("x (normalized)"); plt.ylabel("y (normalized)")
    plt.colorbar(); plt.savefig("imgs/embedding_grad_mag.png", dpi=300)

    print(f"Grid resolution: {grid_res}×{grid_res}")
    print(f"Estimated Lipschitz constant on grid (||ΔE||/||Δx||): {L_est:.4f}")
    return L_est

if __name__ == "__main__":
    """"
    Visualizer
    Expects a checkpoint saved from `tokenizer.py` training
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    sd = torch.load("tokenizer_model.pth", map_location=device)

    # Infer number of centers & embedding dim from checkpoint tensors
    num_centers = sd['embedding.C'].shape[0]
    emb_dim = sd['embedding.E'].shape[1]

    # Reconstruct centers tensor for initialization (will be overwritten)
    centers = sd['embedding.C'].clone().cpu().numpy()

    # Build tokenizer with dummy transform
    tokenizer = PointTokenizer(
        kmeans_or_centers=centers,
        D=emb_dim,
        dropout=0.0,
        hidden_dim=256,
        transform=(1.0, 0.0, 0.0),  # placeholder
        tau=1.0,
    ).to(device)

    missing, unexpected = tokenizer.load_state_dict(sd, strict=False)
    if missing:
        print(f"Warning: missing keys when loading tokenizer: {missing}")
    if unexpected:
        print(f"Warning: unexpected keys when loading tokenizer: {unexpected}")

    tokenizer.eval()

    # Extract transform from loaded buffers for meter visualization
    scale = float(tokenizer.scale.item())
    tx, ty = tokenizer.translate.cpu().tolist()
    transform = (scale, tx, ty)

    L = visualize_embedding_field(tokenizer, transform, grid_res=480, device=device, pca_fit_on='centers')
    print(f"Estimated Lipschitz constant L={L:.4f} for tokenizer with {num_centers} centers.")
