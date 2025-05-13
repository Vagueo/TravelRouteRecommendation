import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# ------------------ 评估指标 ------------------
def compute_topk(logits, targets, k_list):
    metrics = {k: {"hit": 0, "count": 0} for k in k_list}
    for logit, target in zip(logits, targets):
        if target == -100: continue
        probs = torch.softmax(logit, dim=-1)
        for k in k_list:
            topk = probs.topk(k).indices
            hit = int(target in topk)
            metrics[k]["hit"] += hit
            metrics[k]["count"] += 1
    return metrics

def compute_topk_metrics(mdd_logits, poi_logits, mdd_targets, poi_targets, k_list=[5, 10]):
    mdd_metrics = compute_topk(mdd_logits, mdd_targets, k_list) if mdd_logits.numel() > 0 else {
        k: {"hit": 0, "count": 0} for k in k_list}
    poi_metrics = compute_topk(poi_logits, poi_targets, k_list) if poi_logits.numel() > 0 else {
        k: {"hit": 0, "count": 0} for k in k_list}

    merged = {}
    for k in k_list:
        total_hit = mdd_metrics[k]["hit"] + poi_metrics[k]["hit"]
        total_count = mdd_metrics[k]["count"] + poi_metrics[k]["count"]
        merged[k] = {
            "hit_rate": total_hit / total_count if total_count else 0,
            "precision_at_k": total_hit / k / total_count if total_count else 0,
            "recall": total_hit / total_count if total_count else 0,
        }
    return merged

def merge_metrics(metric_list):
    final = {}
    if not metric_list: return final
    keys = metric_list[0].keys()
    for k in keys:
        total_hit, total_count = 0, 0
        for m in metric_list:
            total_hit += m[k]["hit_rate"] * m[k]["count"]
            total_count += m[k]["count"]
        final[k] = {
            "hit_rate": total_hit / total_count if total_count else 0,
            "precision_at_k": 0,
            "recall": total_hit / total_count if total_count else 0,
            "count": total_count
        }
    return final
def print_metrics(metrics, prefix=""):
    for k, v in metrics.items():
        print(f"{prefix}Top-{k}: HitRate={v['hit_rate']:.4f}, Precision@K={v['precision_at_k']:.4f}, Recall={v['recall']:.4f}")


# ------------------ 训练一个轮次 ------------------
def train_one_epoch(model, dataloader, optimizer, device, criterion=None):
    model.train()
    total_loss = 0
    all_metrics = []

    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    for batch in tqdm(dataloader):
        src = batch['input_ids'].to(device)
        types = batch['input_types'].to(device)
        target = batch['target_id'].to(device)
        target_type = batch['target_type'].to(device)
        details_input_ids = batch['details_tokens']['input_ids'].to(device)
        details_attention_mask = batch['details_tokens']['attention_mask'].to(device)

        optimizer.zero_grad()
        mdd_logits, poi_logits, type_logits = model(src, types, details_input_ids, details_attention_mask)

        mdd_last_logits = mdd_logits[:, -1, :]
        poi_last_logits = poi_logits[:, -1, :]

        loss = 0
        loss_count = 0

        if (target_type == 0).any():
            mdd_indices = (target_type == 0)
            mdd_pred = mdd_last_logits[mdd_indices]
            mdd_target = target[mdd_indices]
            loss += criterion(mdd_pred, mdd_target)
            loss_count += 1

        if (target_type == 1).any():
            poi_indices = (target_type == 1)
            poi_pred = poi_last_logits[poi_indices]
            poi_target = target[poi_indices]
            loss += criterion(poi_pred, poi_target)
            loss_count += 1

        if loss_count > 0:
            loss = loss / loss_count

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        mdd_logits_list = mdd_last_logits.detach().cpu()
        poi_logits_list = poi_last_logits.detach().cpu()
        target_list = target.detach().cpu()
        target_type_list = target_type.detach().cpu()

        mdd_targets = torch.where(target_type_list == 0, target_list, torch.full_like(target_list, -100))
        poi_targets = torch.where(target_type_list == 1, target_list, torch.full_like(target_list, -100))

        metrics = compute_topk_metrics(
            mdd_logits_list, poi_logits_list,
            mdd_targets, poi_targets,
            k_list=[5, 10]
        )
        all_metrics.append(metrics)

    avg_metrics = merge_metrics(all_metrics)
    print_metrics(avg_metrics, prefix="Train | ")
    return total_loss / len(dataloader)

# ------------------ 验证和测试 ------------------
@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    all_metrics = []

    for batch in tqdm(dataloader, desc="Evaluating"):
        src = batch['input_ids'].to(device)
        types = batch['input_types'].to(device)
        targets = batch['target_id'].to(device)
        target_types = batch['target_type'].to(device)
        details_input_ids = batch['details_tokens']['input_ids'].to(device)
        details_attention_mask = batch['details_tokens']['attention_mask'].to(device)

        mdd_logits, poi_logits, type_logits = model(src, types, details_input_ids, details_attention_mask)

        mdd_logits = mdd_logits[:, -1, :]
        poi_logits = poi_logits[:, -1, :]

        mdd_mask = (target_types == 0)
        poi_mask = (target_types == 1)

        loss_mdd = F.cross_entropy(mdd_logits[mdd_mask], targets[mdd_mask], ignore_index=-100) if mdd_mask.any() else torch.tensor(0.0, device=device)
        loss_poi = F.cross_entropy(poi_logits[poi_mask], targets[poi_mask], ignore_index=-100) if poi_mask.any() else torch.tensor(0.0, device=device)
        loss_type = F.cross_entropy(type_logits, target_types, ignore_index=-100)

        loss = loss_mdd + loss_poi + 0.5 * loss_type
        total_loss += loss.item()

        mdd_logits_selected = mdd_logits[mdd_mask] if mdd_mask.any() else torch.tensor([], device=device)
        poi_logits_selected = poi_logits[poi_mask] if poi_mask.any() else torch.tensor([], device=device)
        mdd_targets = targets[mdd_mask] if mdd_mask.any() else torch.tensor([], device=device)
        poi_targets = targets[poi_mask] if poi_mask.any() else torch.tensor([], device=device)

        metrics = compute_topk_metrics(mdd_logits_selected, poi_logits_selected, mdd_targets, poi_targets)
        all_metrics.append(metrics)

    return total_loss / len(dataloader), merge_metrics(all_metrics)

# ------------------ 自回归推理 + BLEU 评估 ------------------
@torch.no_grad()
def evaluate_autoregressive_bleu(model, dataloader, device, max_len=10):
    model.eval()
    bleu_scores_per_step = [[] for _ in range(max_len)]
    smoother = SmoothingFunction().method1

    for batch in tqdm(dataloader, desc='AutoRegressive Eval'):
        batch_size = batch['input_ids'].size(0)

        for b in range(batch_size):
            input_ids = batch['input_ids'][b].tolist()[:2]
            input_types = batch['input_types'][b].tolist()[:2]
            target_ids = batch['target_id'][b].tolist()
            target_types = batch['target_type'][b].tolist()
            details_input_ids = batch['details_input_ids'][b].unsqueeze(0).to(device)
            details_attention_mask = batch['details_attention_mask'][b].unsqueeze(0).to(device)

            gen_ids = input_ids.copy()
            gen_types = input_types.copy()

            for step in range(max_len):
                curr_input = torch.tensor([gen_ids], dtype=torch.long, device=device)
                curr_types = torch.tensor([gen_types], dtype=torch.long, device=device)

                mdd_logits, poi_logits, type_logits = model(curr_input, curr_types, details_input_ids, details_attention_mask)

                pred_type = torch.argmax(torch.softmax(type_logits[:, -1, :], dim=-1), dim=-1).item()
                gen_types.append(pred_type)

                if pred_type == 0:
                    logits = mdd_logits[:, -1, :]
                else:
                    logits = poi_logits[:, -1, :]

                pred_token = torch.argmax(torch.softmax(logits, dim=-1), dim=-1).item()
                gen_ids.append(pred_token)

                ref = target_ids[2:2 + step + 1]
                hyp = gen_ids[2:2 + step + 1]
                if len(ref) >= len(hyp):
                    bleu = sentence_bleu([ref], hyp, weights=(1.0,), smoothing_function=smoother)
                    bleu_scores_per_step[step].append(bleu)

    avg_bleus = [sum(step) / len(step) for step in bleu_scores_per_step]
    return avg_bleus

# ------------------ 可视化 ------------------

def plot_metrics(train_losses, val_losses, bleu_scores):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(16, 5))

    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Over Epochs')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(range(1, len(bleu_scores) + 1), bleu_scores)
    plt.xlabel('Step')
    plt.ylabel('Avg BLEU')
    plt.title('Auto-Regressive BLEU per Step')

    plt.subplot(1, 3, 3)
    top5_hit = [x["hit_rate"] for x in bleu_scores] if isinstance(bleu_scores[0], dict) else None
    if top5_hit:
        plt.plot(epochs, top5_hit, label="Top-5 Hit Rate")
        plt.title("Top-5 HitRate Over Epochs")
        plt.legend()

    plt.tight_layout()
    plt.show()

