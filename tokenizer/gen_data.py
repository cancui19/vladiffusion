import os
import json
from math import atan2
from dotenv import load_dotenv
import numpy as np
from typing import Dict, List
from collections import defaultdict

OBS_LEN = 10
FUT_LEN = 10
TTL_LEN = OBS_LEN + FUT_LEN

from nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes

load_dotenv()

VERSION = 'v1.0-trainval'
DATAROOT = os.getenv("NUSCENES_ROOT")
# output dir（ train / val）
PROMPT_TYPE = ["long", "short"][0]
OUTPUT_TEXT = False # True
SCAN_EXISTING_IMAGES = True  # Set True to enable reverse lookup mode (glob CAM_FRONT folder)
if OUTPUT_TEXT:
    OUTPUT_JSON_TRAIN = f"/depot/ziran/apps/jiaru/projects/vladiffusion/data/nuscenes_waypoint_text_{PROMPT_TYPE}_prompt_train.json"
    OUTPUT_JSON_VAL   = f"/depot/ziran/apps/jiaru/projects/vladiffusion/data/nuscenes_waypoint_text_{PROMPT_TYPE}_prompt_val.json"
else:
    OUTPUT_JSON_TRAIN = f"/{DATAROOT}/nuscenes_waypoint_{PROMPT_TYPE}_prompt_train.json"
    OUTPUT_JSON_VAL   = f"/{DATAROOT}/nuscenes_waypoint_{PROMPT_TYPE}_prompt_val.json"

nusc = NuScenes(version=VERSION, dataroot=DATAROOT, verbose=True)
scenes = nusc.scene

splits = create_splits_scenes()
train_scene_names = set(splits['train'])
val_scene_names   = set(splits['val'])

def convert_with_format(num):
    return "{:012d}".format(num)

def yaw_from_quat(q):
    """
    q: [w, x, y, z]
    """
    w, x, y, z = q
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return atan2(siny_cosp, cosy_cosp)

def world_to_ego0_xy(p_world_xy, p0_world_xy, yaw0):
    dx = p_world_xy[0] - p0_world_xy[0]
    dy = p_world_xy[1] - p0_world_xy[1]
    cy, sy = np.cos(-yaw0), np.sin(-yaw0)
    x_ego = cy * dx - sy * dy
    y_ego = sy * dx + cy * dy
    return float(x_ego), float(y_ego)

def format_waypoints_str(pts_xy, prec=2):
    def fmt(v): 
        return f"{v:.{prec}f}"
    return ", ".join([f"[{fmt(x)}, {fmt(y)}]" for x, y in pts_xy])

def history_waypoints_ego0(obs_traj_world, yaw0, p0_xy, prec=2):
    hist = []
    for p in obs_traj_world:
        x, y = world_to_ego0_xy((p[0], p[1]), p0_xy, yaw0)
        hist.append([x, y])
    return hist

# --------- （train / val） ----------
data_train = []
data_val = []
idx_train = 1
idx_val = 1

if SCAN_EXISTING_IMAGES:
    cam_dir = os.path.join(DATAROOT, 'samples', 'CAM_FRONT')
    existing = set(os.listdir(cam_dir)) if os.path.isdir(cam_dir) else set()
    print(f"[scan] Found {len(existing)} existing CAM_FRONT images in {cam_dir}")

    # Build mapping from scene name to list of entries (only existing files)
    scene_entries: Dict[str, List[Dict]] = defaultdict(list)
    # Iterate over all sample_data records for CAM_FRONT
    # Access underlying tables via nusc.sample_data, but NuScenes API wrapper doesn't expose list directly; replicate via json
    # We'll traverse via nusc.sample_data if available, else fallback to nusc.sample
    # NuScenes has internal attribute nusc.sample_data (dict token->record)
    sample_data_store = getattr(nusc, 'sample_data', None)
    # Fallback: load JSON directly if attribute not structured
    if sample_data_store is None:
        json_path = os.path.join(DATAROOT, VERSION, 'sample_data.json')
        with open(json_path, 'r') as f:
            sample_data_store = json.load(f)
    total_cam_records = 0
    kept_records = 0
    # NuScenes internal may be list of dicts; normalize iteration
    if isinstance(sample_data_store, dict):
        iterable = sample_data_store.values()
    else:
        iterable = sample_data_store
    for sd in iterable:
        fn = sd.get('filename', '')
        if 'samples/CAM_FRONT/' not in fn:
            continue
        total_cam_records += 1
        basename = fn.split('samples/CAM_FRONT/')[1]
        if basename not in existing:
            continue
        # Retrieve sample to get scene token
        sample = nusc.get('sample', sd['sample_token'])
        scene_token = sample['scene_token']
        scene_rec = nusc.get('scene', scene_token)
        scene_name = scene_rec['name']
        if scene_name not in train_scene_names and scene_name not in val_scene_names:
            continue
        entry = {
            'filename': os.path.join(DATAROOT, fn),
            'ego_pose': nusc.get('ego_pose', sd['ego_pose_token']),
            'calib': nusc.get('calibrated_sensor', sd['calibrated_sensor_token']),
            'timestamp': sd['timestamp']
        }
        scene_entries[scene_name].append(entry)
        kept_records += 1
    print(f"[scan] Camera records total: {total_cam_records}, kept (existing & split): {kept_records}")

    # Process each scene entries sorted by timestamp
    for scene_name, entries in scene_entries.items():
        target_split = 'train' if scene_name in train_scene_names else 'val'
        entries.sort(key=lambda e: e['timestamp'])
        # Build image list & ego poses list
        front_camera_images = [e['filename'] for e in entries]
        ego_poses = [e['ego_pose'] for e in entries]
        scene_length = len(front_camera_images)
        print(f"[scan][{target_split}] Scene {scene_name} has {scene_length} existing frames")
        if scene_length < TTL_LEN:
            print(f"[scan][{target_split}] Scene {scene_name} < {TTL_LEN} frames, skip")
            continue
        ego_traj_world = [pose['translation'][:3] for pose in ego_poses]
        for i in range(scene_length - TTL_LEN + 1):
            obs_images = front_camera_images[i:i+OBS_LEN]
            obs_ego_traj_world = ego_traj_world[i:i+OBS_LEN]
            fut_ego_traj_world = ego_traj_world[i+OBS_LEN:i+TTL_LEN]
            t0_pose = ego_poses[i + OBS_LEN - 1]
            p0 = t0_pose['translation']
            yaw0 = yaw_from_quat(t0_pose['rotation'])
            p0_xy = (float(p0[0]), float(p0[1]))
            hist_waypoints = history_waypoints_ego0(obs_ego_traj_world, yaw0, p0_xy, prec=2)
            hist_waypoints_str = format_waypoints_str(hist_waypoints)
            fut_waypoints_ego = []
            for p in fut_ego_traj_world:
                xk, yk = world_to_ego0_xy((p[0], p[1]), p0_xy, yaw0)
                fut_waypoints_ego.append([xk, yk])
            fut_waypoints_str = format_waypoints_str(fut_waypoints_ego)
            if PROMPT_TYPE == "short":
                system_message = (
                    f"Predict the next {FUT_LEN} ego-frame waypoints (x-forward, y-left) at 0.5s intervals based on front-view image."
                )
                user_prompt = ""
            elif PROMPT_TYPE == "long":
                system_message = (
                    "You are an autonomous driving labeller. You are given a sequence of front-view images, "
                    "the 5-second historical ego waypoints in the *current ego frame* (x-forward, y-left), "
                    "and a driving rationale. Your task is to predict the future ego-frame waypoints for the next "
                    f"{FUT_LEN} timesteps (≈0.5 s interval). Provide raw coordinates only.\n"
                )
                user_prompt = (
                    "This is a frame from a front camera video captured.\n\n"
                    f"The 5-second historical ego-frame waypoints (relative to the last observed frame) are {hist_waypoints_str}.\n\n"
                    f"Generate the predicted future waypoints in the format [x_1, y_1], [x_2, y_2], ..., [x_{FUT_LEN}, y_{FUT_LEN}]. "
                    "Write the raw text, not markdown or LaTeX. Future waypoints:\n    "
                )
            cur_data = {
                'id': convert_with_format(idx_train if target_split == 'train' else idx_val),
                'image': obs_images[-1],
                'action_targets': fut_waypoints_ego,
                'action_mask': [1] * len(fut_waypoints_ego),
                'scene': scene_name,
                'conversations': [
                    {
                        'from': 'human',
                        'value': system_message + '<image>' + user_prompt
                    },
                    {
                        'from': 'gpt',
                        # Always emit a parseable Python literal (list of [x,y]) even when OUTPUT_TEXT is False
                        'value': (fut_waypoints_str if OUTPUT_TEXT else str(fut_waypoints_ego))
                    }
                ]
            }
            if target_split == 'train':
                data_train.append(cur_data)
                idx_train += 1
            else:
                data_val.append(cur_data)
                idx_val += 1
else:
    # Original forward traversal relying on full dataset completeness
    for scene in scenes:
        name = scene['name']
        target_split = None
        if name in train_scene_names:
            target_split = 'train'
        elif name in val_scene_names:
            target_split = 'val'
        else:
            continue
        first_sample_token = scene['first_sample_token']
        last_sample_token  = scene['last_sample_token']
        front_camera_images = []
        ego_poses = []
        curr_sample_token = first_sample_token
        while True:
            sample = nusc.get('sample', curr_sample_token)
            cam_front_data = nusc.get('sample_data', sample['data']['CAM_FRONT'])
            # Removed blocking input()
            img_path = os.path.join(nusc.dataroot, cam_front_data['filename'])
            if not os.path.exists(img_path):
                # Skip missing image frames (partial dataset scenario)
                print(f"[warn] missing image {img_path}, skipping frame")
                if curr_sample_token == last_sample_token:
                    break
                curr_sample_token = sample['next']
                continue
            front_camera_images.append(img_path)
            pose = nusc.get('ego_pose', cam_front_data['ego_pose_token'])
            ego_poses.append(pose)
            if curr_sample_token == last_sample_token:
                break
            curr_sample_token = sample['next']
        scene_length = len(front_camera_images)
        print(f"[{target_split}] Scene {name} has {scene_length} existing frames (after filtering)")
        if scene_length < TTL_LEN:
            print(f"[{target_split}] Scene {name} < {TTL_LEN} frames, skip")
            continue
        ego_traj_world = [ego_poses[t]['translation'][:3] for t in range(scene_length)]
        for i in range(scene_length - TTL_LEN + 1):
            obs_images = front_camera_images[i:i+OBS_LEN]
            obs_ego_traj_world = ego_traj_world[i:i+OBS_LEN]
            fut_ego_traj_world = ego_traj_world[i+OBS_LEN:i+TTL_LEN]
            t0_pose = ego_poses[i + OBS_LEN - 1]
            p0 = t0_pose['translation']
            yaw0 = yaw_from_quat(t0_pose['rotation'])
            p0_xy = (float(p0[0]), float(p0[1]))
            hist_waypoints = history_waypoints_ego0(obs_ego_traj_world, yaw0, p0_xy, prec=2)
            hist_waypoints_str = format_waypoints_str(hist_waypoints)
            fut_waypoints_ego = []
            for p in fut_ego_traj_world:
                xk, yk = world_to_ego0_xy((p[0], p[1]), p0_xy, yaw0)
                fut_waypoints_ego.append([xk, yk])
            fut_waypoints_str = format_waypoints_str(fut_waypoints_ego)
            if PROMPT_TYPE == "short":
                system_message = (
                    f"Predict the next {FUT_LEN} ego-frame waypoints (x-forward, y-left) at 0.5s intervals based on front-view image."
                )
                user_prompt = ""
            elif PROMPT_TYPE == "long":
                system_message = (
                    "You are an autonomous driving labeller. You are given a sequence of front-view images, "
                    "the 5-second historical ego waypoints in the *current ego frame* (x-forward, y-left), "
                    "and a driving rationale. Your task is to predict the future ego-frame waypoints for the next "
                    f"{FUT_LEN} timesteps (≈0.5 s interval). Provide raw coordinates only.\n"
                )
                user_prompt = (
                    "This is a frame from a front camera video captured.\n\n"
                    f"The 5-second historical ego-frame waypoints (relative to the last observed frame) are {hist_waypoints_str}.\n\n"
                    f"Generate the predicted future waypoints in the format [x_1, y_1], [x_2, y_2], ..., [x_{FUT_LEN}, y_{FUT_LEN}]. "
                    "Write the raw text, not markdown or LaTeX. Future waypoints:\n    "
                )
            cur_data = {
                'id': convert_with_format(idx_train if target_split == 'train' else idx_val),
                'image': obs_images[-1],
                'action_targets': fut_waypoints_ego,
                'action_mask': [1] * len(fut_waypoints_ego),
                'scene': name,
                'conversations': [
                    {'from': 'human', 'value': system_message + '<image>' + user_prompt},
                    # Provide parseable list literal string for assistant reply when OUTPUT_TEXT False
                    {'from': 'gpt', 'value': (fut_waypoints_str if OUTPUT_TEXT else str(fut_waypoints_ego))}
                ]
            }
            if target_split == 'train':
                data_train.append(cur_data)
                idx_train += 1
            else:
                data_val.append(cur_data)
                idx_val += 1

os.makedirs(os.path.dirname(OUTPUT_JSON_TRAIN), exist_ok=True)
with open(OUTPUT_JSON_TRAIN, 'w', encoding='utf-8') as f:
    json.dump(data_train, f, indent=2, ensure_ascii=False)
with open(OUTPUT_JSON_VAL, 'w', encoding='utf-8') as f:
    json.dump(data_val, f, indent=2, ensure_ascii=False)

print(f"[ok] wrote train samples: {len(data_train)} -> {OUTPUT_JSON_TRAIN}")
print(f"[ok] wrote val   samples: {len(data_val)}   -> {OUTPUT_JSON_VAL}")
