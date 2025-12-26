import json
import re
import numpy as np

PATHS = {
    "data/nuscenes_waypoint_text_long_prompt_train_Nu_X.json": "data/nuscenes_delta_text_long_prompt_train_Nu_X.json",
    "data/nuscenes_waypoint_text_long_prompt_val_Nu_X.json": "data/nuscenes_delta_text_long_prompt_val_Nu_X.json",
    "data/nuscenes_waypoint_text_long_prompt_train.json": "data/nuscenes_delta_text_long_prompt_train.json"
}

PROMPT = "Generate the predicted future waypoints in the format [x_1, y_1], [x_2, y_2], ..., [x_10, y_10]. Write the raw text, not markdown or LaTeX. Future waypoints"
REPLACE = "Generate the predicted future waypoints in the action token format action_1, action_2, ... action_10. Write action tokens, not raw text, markdown or LaTeX. Future waypoints"

for src, dest in PATHS.items():
    with open(src, "r") as f:
        in_data = json.load(f)

    delta_data = in_data.copy()

    for i, item in enumerate(delta_data):
        action_targets = item["action_targets"]
        gpt_msg = item["conversations"][1]["value"]

        # form new gpt_msg
        # originally, looks like this:
        # 'value': '[2.98, -0.03], [5.91, -0.06], [8.75, -0.08], [11.42, -0.10], [13.78, -0.12], [16.26, -0.16], [18.54, -0.23], [20.90, -0.38], [23.33, -0.58], [25.34, -0.82]'

        action_targets_array = np.array(action_targets)
        delta_array = np.diff(
            action_targets_array,
            axis=0,
            prepend=np.zeros((1, action_targets_array.shape[1])),
        )

        # this can later be reversed during inference/eval:
        # l = "[" + delta_str + "]"
        # np.cumsum(l, axis=0)

        delta_list = delta_array.tolist()
        delta_str = ", ".join([f"[{x[0]}, {x[1]}]" for x in delta_list])
        delta_data[i]["conversations"][1]["value"] = delta_str

        # for non-NuX, replace
        if "Nu_X" not in src:
            delta_data[i]["conversations"][0]["value"] = delta_data[i]["conversations"][0]["value"].replace(PROMPT, REPLACE)

    with open(dest, "w") as f:
        json.dump(delta_data, f, indent=2)
