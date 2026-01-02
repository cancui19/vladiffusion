import json, ast, torch

from train.nuscene_inference_lora_metrics import compute_displacement_errors

path = "results/vla_20260101_1621.json"
DELTA = True


with open(path) as f:
    entries = json.load(f)

def compute_metric(entries, steps=10, specific=None):
    # 5s

    total_ade = 0.0
    total_fde = 0.0
    total_steps = 0
    num_samples = 0

    if not specific:
        min_idx = 0
        max_idx = steps

    else:
        min_idx = steps - 1
        max_idx = steps

    for pred_str, gt_str in entries:
        # 字符串转成真正的 [N, 2] 数组
        pred = torch.tensor(ast.literal_eval("[" + pred_str.strip().strip(",") + "]"), dtype=torch.float32)
        gt = torch.tensor(ast.literal_eval("[" + gt_str.strip().strip(",") + "]"), dtype=torch.float32)

        if DELTA:
            pred = torch.cumsum(pred, dim=0)
            gt = torch.cumsum(gt, dim=0)

        ade, fde, steps_used, _ = compute_displacement_errors(pred[min_idx:max_idx], gt[min_idx:max_idx])
        total_ade += ade * steps_used
        total_fde += fde
        total_steps += steps_used
        num_samples += 1

    dataset_ade = total_ade / total_steps
    dataset_fde = total_fde / num_samples

    num_seconds = steps * 0.5
    delim = "->" if not specific else "@"
    if not specific:
        print(f"ADE{delim}{num_seconds}s={dataset_ade:.4f}, FDE={dataset_fde:.4f}")
    else:
        print(f"ADE{delim}{num_seconds}s={dataset_ade:.4f}")

if __name__ == "__main__":
    # compute @1s, @2s, @3s.
    # compute up to 3s, 5s

    for steps in [2, 4, 6]:
        compute_metric(entries, steps=steps, specific=True)
    
    for steps in [6, 10]:
        compute_metric(entries, steps=steps)

