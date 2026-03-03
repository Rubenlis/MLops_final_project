import re
import string
from collections import Counter
from typing import Dict, List

import torch
import torch.nn as nn


def normalize_answer(s: str) -> str:
    def lower(text): return text.lower()
    def remove_punc(text): return "".join(ch for ch in text if ch not in set(string.punctuation))
    def remove_articles(text): return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text): return " ".join(text.split())
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction: str, ground_truth: str) -> float:
    return 1.0 if normalize_answer(prediction) == normalize_answer(ground_truth) else 0.0


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()

    if not pred_tokens and not gt_tokens:
        return 1.0
    if not pred_tokens or not gt_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def decode_span(tokens: List[str], start: int, end: int) -> str:
    if start < 0 or end < 0 or start >= len(tokens) or end >= len(tokens) or end < start:
        return ""
    return " ".join(tokens[start:end + 1])


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    ce_loss: nn.Module,
    max_answer_len: int = 30,
    sep_token: str = "<SEP>",
) -> Dict[str, float]:
    model.eval()

    total_loss = 0.0
    n_batches = 0

    total_em = 0.0
    total_f1 = 0.0
    n_eval = 0

    dataset = dataloader.dataset  # expects SquadDataset with .data

    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        start_pos = batch["start_position"].to(device)
        end_pos = batch["end_position"].to(device)

        start_logits, end_logits = model(input_ids)

        loss_start = ce_loss(start_logits, start_pos)
        loss_end = ce_loss(end_logits, end_pos)
        loss = (loss_start + loss_end) / 2

        total_loss += float(loss.item())
        n_batches += 1

        # predictions
        start_pred = torch.argmax(start_logits, dim=1)
        end_pred = torch.argmax(end_logits, dim=1)

        # enforce end >= start and cap length
        end_pred = torch.maximum(end_pred, start_pred)
        end_pred = torch.minimum(end_pred, start_pred + max_answer_len)

        bs = input_ids.size(0)
        for i in range(bs):
            # skip ignored labels
            if int(start_pos[i].item()) == -100 or int(end_pos[i].item()) == -100:
                continue

            # because val_loader shuffle=False, this mapping is OK
            global_idx = batch_idx * dataloader.batch_size + i
            if global_idx >= len(dataset):
                continue

            item = dataset.data[global_idx]
            tokens = item["question_tokens"] + [sep_token] + item["context_tokens"]

            gold_start = int(item["start_token_idx"] + len(item["question_tokens"]) + 1)
            gold_end = int(item["end_token_idx"] + len(item["question_tokens"]) + 1)
            gold = decode_span(tokens, gold_start, gold_end)

            pred = decode_span(tokens, int(start_pred[i].item()), int(end_pred[i].item()))

            total_em += exact_match_score(pred, gold)
            total_f1 += f1_score(pred, gold)
            n_eval += 1

    return {
        "val_loss": total_loss / max(1, n_batches),
        "em": total_em / max(1, n_eval),
        "f1": total_f1 / max(1, n_eval),
        "n_eval": float(n_eval),
    }