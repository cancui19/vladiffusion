import numpy as np
from sklearn.cluster import MiniBatchKMeans
import torch
import math
from torch.nn import functional as F
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os

from nuscenes_dataset import NuScenesDataset

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
    for idx in range(len(dataset)):
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

if __name__ == "__main__":
    load_dotenv()

    VERSION = 'v1.0-trainval'
    DATAROOT = os.getenv("NUSCENES_ROOT")
    dataset = NuScenesDataset(nuscenes_path=DATAROOT, version=VERSION, split='train', future_seconds=10, future_hz=2)
    xy_points = get_xy_points(dataset)

    print(f"Total number of points: {xy_points.shape[0]}")
    plt.figure(figsize=(8, 8))
    plt.scatter(xy_points[:, 0], xy_points[:, 1], s=0.1, alpha=0.5)
    plt.title("Future Waypoints in Ego Vehicle Frame")
    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    plt.axis('equal')
    plt.grid(True)
    plt.savefig("future_waypoints_ego_frame.png", dpi=300)

