import json, ast, torch

from train.nuscene_inference_lora_metrics import compute_displacement_errors

path = "results/vla_20251222_1818.json"

with open(path) as f:
    entries = json.load(f)

# 5s

total_ade = 0.0
total_fde = 0.0
total_steps = 0
num_samples = 0

for pred_str, gt_str in entries:
    # 字符串转成真正的 [N, 2] 数组
    pred = torch.tensor(ast.literal_eval("[" + pred_str.strip().strip(",") + "]"), dtype=torch.float32)
    gt = torch.tensor(ast.literal_eval("[" + gt_str.strip().strip(",") + "]"), dtype=torch.float32)

    ade, fde, steps_used, _ = compute_displacement_errors(pred, gt)
    total_ade += ade * steps_used
    total_fde += fde
    total_steps += steps_used
    num_samples += 1

dataset_ade = total_ade / total_steps
dataset_fde = total_fde / num_samples
print(f"ADE@5s={dataset_ade:.4f}, FDE={dataset_fde:.4f}")

### 3s

total_ade = 0.0
total_fde = 0.0
total_steps = 0
num_samples = 0

for pred_str, gt_str in entries:
    # 字符串转成真正的 [N, 2] 数组
    pred = torch.tensor(ast.literal_eval("[" + pred_str.strip().strip(",") + "]"), dtype=torch.float32)
    gt = torch.tensor(ast.literal_eval("[" + gt_str.strip().strip(",") + "]"), dtype=torch.float32)

    ade, fde, steps_used, _ = compute_displacement_errors(pred[:6], gt[:6])
    total_ade += ade * steps_used
    total_fde += fde
    total_steps += steps_used
    num_samples += 1

dataset_ade = total_ade / total_steps
dataset_fde = total_fde / num_samples
print(f"ADE@3s={dataset_ade:.4f}, FDE={dataset_fde:.4f}")
