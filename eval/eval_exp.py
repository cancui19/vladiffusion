import json
import re
from typing import Dict, Optional
import evaluate
bleu_metric = evaluate.load("bleu")
rouge_metric = evaluate.load("rouge")
meteor_metric = evaluate.load("meteor")

total_bleu_narration_score = 0
total_rouge_narration_score = 0
total_meteor_narration_score = 0
total_bleu_reasoning_score = 0
total_rouge_reasoning_score = 0
total_meteor_reasoning_score = 0
COUNT_MISSING = False
missing_narration_count = 0
missing_reasoning_count = 0
narration_count = 0
reasoning_count = 0


def _extract_segments(text: str) -> Dict[str, str]:
    """解析单条文本，提取 Action/Narration/Reasoning/Description 等段落。"""
    label_pattern = re.compile(r"(Action|Narration|Reasoning|Description)\s*[:：]+", re.IGNORECASE)
    matches = list(label_pattern.finditer(text))
    if not matches:
        return {}

    segments: Dict[str, str] = {}
    for idx, match in enumerate(matches):
        label = match.group(1).lower()
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        if content and label not in segments:
            segments[label] = content
    return segments


def _normalize(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    cleaned = text.strip().strip("\n")
    return cleaned or None


with open('/depot/ziran/apps/jiaru/projects/vladiffusion/results/vla_explanation_init.json', 'r') as f:
    exp = json.load(f)

for item in exp:
    text_outputs = item[1] if len(item) > 1 else []

    narration: Optional[str] = None
    reasoning: Optional[str] = None

    # 先尝试逐条解析标签段落
    for text in text_outputs:
        segments = _extract_segments(text)
        if not segments:
            continue
        if narration is None:
            narration = _normalize(segments.get('narration') or segments.get('description'))
        if reasoning is None:
            reasoning = _normalize(segments.get('reasoning'))
        if narration and reasoning:
            break

    # 如果仍缺失，尝试正则兜底
    if narration is None or reasoning is None:
        for text in text_outputs:
            if narration is None:
                match = re.search(r'(Narration|Description)\s*[:：]+\s*(.+?)(?=Reasoning\s*[:：]+|$)', text, re.IGNORECASE | re.DOTALL)
                if match:
                    narration = _normalize(match.group(2))
            if reasoning is None:
                match = re.search(r'Reasoning\s*[:：]+\s*(.+)', text, re.IGNORECASE | re.DOTALL)
                if match:
                    reasoning = _normalize(match.group(1))
            if narration and reasoning:
                break

    label_raw = item[2] if len(item) > 2 else ""
    label_segments = _extract_segments(label_raw)
    narration_gt = _normalize(label_segments.get('narration') or label_segments.get('description'))
    reasoning_gt = _normalize(label_segments.get('reasoning'))

    print("Pred Narration:", narration or "(missing)")
    print("Pred Reasoning:", reasoning or "(missing)")
    print("GT Narration:", narration_gt or "(missing)")
    print("GT Reasoning:", reasoning_gt or "(missing)")
    
    if COUNT_MISSING:
        if not narration: 
            narration = ""
        if not reasoning:
            reasoning = ""
            
    if narration is not None:
        narration_bleu = bleu_metric.compute(predictions=[narration], references=[narration_gt])['bleu']
        narration_rouge = rouge_metric.compute(predictions=[narration], references=[narration_gt])['rougeL']
        narration_meteor = meteor_metric.compute(predictions=[narration], references=[narration_gt])['meteor']
        print(f"Narration BLEU: {narration_bleu}")
        print(f"Narration ROUGE: {narration_rouge}")
        print(f"Narration METEOR: {narration_meteor}")
        total_bleu_narration_score += narration_bleu
        total_rouge_narration_score += narration_rouge
        total_meteor_narration_score += narration_meteor
        narration_count += 1
    else:
        missing_narration_count += 1
    if reasoning is not None:
        reasoning_bleu = bleu_metric.compute(predictions=[reasoning], references=[reasoning_gt])['bleu']
        reasoning_rouge = rouge_metric.compute(predictions=[reasoning], references=[reasoning_gt])['rougeL']
        reasoning_meteor = meteor_metric.compute(predictions=[reasoning], references=[reasoning_gt])['meteor']
        print(f"Reasoning BLEU: {reasoning_bleu}")
        print(f"Reasoning ROUGE: {reasoning_rouge}")
        print(f"Reasoning METEOR: {reasoning_meteor}")
        total_bleu_reasoning_score += reasoning_bleu
        total_rouge_reasoning_score += reasoning_rouge
        total_meteor_reasoning_score += reasoning_meteor
        reasoning_count += 1
    else:
        missing_reasoning_count += 1
        
print("missing narration count:", missing_narration_count)
print("missing reasoning count:", missing_reasoning_count)
print("narration count:", narration_count)
print("reasoning count:", reasoning_count)
print("average bleu narration score:", total_bleu_narration_score / narration_count)
print("average rouge narration score:", total_rouge_narration_score / narration_count)
print("average meteor narration score:", total_meteor_narration_score / narration_count)
print("average bleu reasoning score:", total_bleu_reasoning_score / reasoning_count)
print("average rouge reasoning score:", total_rouge_reasoning_score / reasoning_count)
print("average meteor reasoning score:", total_meteor_reasoning_score / reasoning_count)