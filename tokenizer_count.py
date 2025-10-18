from collections import Counter
import json
from tqdm import tqdm
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("/scratch/gilbreth/cancui/models/LLaDA-V")
counter = Counter()
# reserved tokens: 126084 - 126339, 255 tokens, except 126336, which is |mdm_mask|
# we need to add 2048 tokens, so we need to find additional 2048-255 = 1793 tokens

counter.update(range(0, 126080)) # make sure all tokens are counted

def update_counts(text):
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    counter.update(ids)


with open("data/nuscenes_waypoint_short_prompt_train.json", "r") as f:
    for sample in tqdm(json.load(f)):
        for turn in sample["conversations"]:
            update_counts(turn["value"])


least_used = [tok_id for tok_id, _ in counter.most_common()[::-1][:1793]]
assert max(least_used) <= 126080
print("IDs:", least_used[:10], "…", len(least_used))
