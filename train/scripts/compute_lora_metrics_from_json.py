import json
import re
import torch
import sys
import os

# Ensure we can import from train module if running from project root
sys.path.append(os.getcwd())

from train.nuscene_inference_lora_metrics import compute_displacement_errors

path = "/depot/ziran/apps/jiaru/projects/vladiffusion/results/iterative_refinement/iter_5.json"

def extract_trajectory(text):
    """
    Extracts trajectory points from text.
    Matches patterns like [1.23, 4.56] or [1.23, 4.56], ...
    Returns a torch tensor of shape (N, 2).
    """
    if not isinstance(text, str):
        return torch.zeros((0, 2))
        
    # Pattern to match [x, y]
    # Handles optional spaces, negative signs, scientific notation
    pattern = r"\[\s*(-?\d+(?:\.\d+)?(?:e-?\d+)?)\s*,\s*(-?\d+(?:\.\d+)?(?:e-?\d+)?)\s*\]"
    matches = re.findall(pattern, text)
    
    traj = []
    for x, y in matches:
        traj.append([float(x), float(y)])
    
    if not traj:
        return torch.zeros((0, 2))
        
    return torch.tensor(traj, dtype=torch.float32)

with open(path) as f:
    entries = json.load(f)

# 5s
total_ade = 0.0
total_fde = 0.0
total_steps = 0
num_samples = 0

for item in entries:
    # item structure: [pred_traj_str, pred_text_str, gt_traj_text_str, ...]
    if len(item) < 3:
        continue
        
    pred_str = item[0]
    gt_str = item[2]
    
    # Extract trajectories
    pred = extract_trajectory(pred_str)
    gt = extract_trajectory(gt_str)
    
    if pred.size(0) == 0 or gt.size(0) == 0:
        continue

    try:
        ade, fde, steps_used, _ = compute_displacement_errors(pred, gt)
        total_ade += ade * steps_used
        total_fde += fde
        total_steps += steps_used
        num_samples += 1
    except ValueError:
        continue

if total_steps > 0:
    dataset_ade = total_ade / total_steps
    dataset_fde = total_fde / num_samples
    print(f"ADE={dataset_ade:.4f}, FDE={dataset_fde:.4f}")
else:
    print("No valid samples found for 5s metrics.")

### 3s

total_ade = 0.0
total_fde = 0.0
total_steps = 0
num_samples = 0

for item in entries:
    if len(item) < 3:
        continue

    pred_str = item[0]
    gt_str = item[2]
    
    pred = extract_trajectory(pred_str)
    gt = extract_trajectory(gt_str)

    if pred.size(0) == 0 or gt.size(0) == 0:
        continue

    # Calculate for first 6 points (assuming 2Hz, 3s = 6 steps)
    # Clamp to available length if shorter than 6
    horizon = min(6, pred.size(0), gt.size(0))
    if horizon == 0:
        continue
        
    pred_slice = pred[:horizon]
    gt_slice = gt[:horizon]

    try:
        ade, fde, steps_used, _ = compute_displacement_errors(pred_slice, gt_slice)
        total_ade += ade * steps_used
        total_fde += fde
        total_steps += steps_used
        num_samples += 1
    except ValueError:
        continue

if total_steps > 0:
    dataset_ade = total_ade / total_steps
    dataset_fde = total_fde / num_samples
    print(f"ADE={dataset_ade:.4f}, FDE={dataset_fde:.4f}")
else:
    print("No valid samples found for 3s metrics.")
