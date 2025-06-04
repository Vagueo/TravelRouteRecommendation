import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from config import k_lists
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# ------------------ 评估指标 ------------------
def compute_topk(logits, targets, k_list):
    metrics = {
        k: {"hit": 0, "count": 0, "mrr": 0.0, "ndcg": 0.0} for k in k_list
    }
    for logit, target in zip(logits, targets):
        if target == -100: continue
        probs = torch.softmax(logit, dim=-1)
        sorted_indices = torch.argsort(probs, descending=True)

        for k in k_list:
            topk = sorted_indices[:k]
            if target in topk:
                rank = (topk == target).nonzero(as_tuple=True)[0].item() + 1  # 1-based
                metrics[k]["hit"] += 1
                metrics[k]["mrr"] += 1.0 / rank
                metrics[k]["ndcg"] += 1.0 / torch.log2(torch.tensor(rank + 1, dtype=torch.float)).item()
            metrics[k]["count"] += 1
    return metrics

def compute_topk_metrics(mdd_logits, poi_logits, mdd_targets, poi_targets, k_list=[5, 10]):
    mdd_metrics = compute_topk(mdd_logits, mdd_targets, k_list) if mdd_logits.numel() > 0 else {
        k: {"hit": 0, "count": 0, "mrr": 0.0, "ndcg": 0.0} for k in k_list}
    poi_metrics = compute_topk(poi_logits, poi_targets, k_list) if poi_logits.numel() > 0 else {
        k: {"hit": 0, "count": 0, "mrr": 0.0, "ndcg": 0.0} for k in k_list}

    merged = {}
    for k in k_list:
        total_hit = mdd_metrics[k]["hit"] + poi_metrics[k]["hit"]
        total_count = mdd_metrics[k]["count"] + poi_metrics[k]["count"]
        total_mrr = mdd_metrics[k]["mrr"] + poi_metrics[k]["mrr"]
        total_ndcg = mdd_metrics[k]["ndcg"] + poi_metrics[k]["ndcg"]
        merged[k] = {
            "hit_rate": total_hit / total_count if total_count else 0,
            "mrr": total_mrr / total_count if total_count else 0,
            "ndcg": total_ndcg / total_count if total_count else 0,
            "count": total_count
        }
    return merged

def merge_metrics(metric_list):
    final = {}
    if not metric_list: return final
    keys = metric_list[0].keys()
    for k in keys:
        total_hit, total_count, total_mrr, total_ndcg = 0, 0, 0.0, 0.0
        for m in metric_list:
            total_hit += m[k]["hit_rate"] * m[k]["count"]
            total_mrr += m[k]["mrr"] * m[k]["count"]
            total_ndcg += m[k]["ndcg"] * m[k]["count"]
            total_count += m[k]["count"]
        final[k] = {
            "hit_rate": total_hit / total_count if total_count else 0,
            "mrr": total_mrr / total_count if total_count else 0,
            "ndcg": total_ndcg / total_count if total_count else 0,
            "count": total_count
        }
    return final

def print_metrics(metrics, prefix=""):
    for k, v in metrics.items():
        print(f"{prefix}Top-{k}: HitRate={v['hit_rate']:.4f}, MRR={v['mrr']:.4f}, NDCG={v['ndcg']:.4f}")

# ------------------ 训练一个轮次 ------------------
def train_one_epoch(model, dataloader, optimizer, device, alpha=0.5):
    model.train()
    total_loss = 0
    all_metrics = []

    for batch in tqdm(dataloader, desc="Training"):
        src = batch['input_ids'].to(device)
        types = batch['input_types'].to(device)
        targets = batch['target_id'].to(device)
        target_types = batch['target_type'].to(device)
        cls_features = batch['cls_features'].to(device)
        domain_labels = batch['domain_label'].to(device)

        optimizer.zero_grad()
        mdd_logits, poi_logits, type_logits = model(src, types, cls_features, domain_labels)

        mdd_logits = mdd_logits[:, -1, :]
        poi_logits = poi_logits[:, -1, :]

        mdd_mask = (target_types == 0)
        poi_mask = (target_types == 1)

        loss_mdd = F.cross_entropy(mdd_logits[mdd_mask], targets[mdd_mask], ignore_index=-100) if mdd_mask.any() else torch.tensor(0.0, device=device)
        loss_poi = F.cross_entropy(poi_logits[poi_mask], targets[poi_mask], ignore_index=-100) if poi_mask.any() else torch.tensor(0.0, device=device)
        loss_type = F.cross_entropy(type_logits[:, -1, :], target_types, ignore_index=-100)

        loss = loss_mdd + loss_poi + alpha * loss_type
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        metrics = compute_topk_metrics(
            mdd_logits[mdd_mask] if mdd_mask.any() else torch.tensor([], device=device),
            poi_logits[poi_mask] if poi_mask.any() else torch.tensor([], device=device),
            targets[mdd_mask] if mdd_mask.any() else torch.tensor([], device=device),
            targets[poi_mask] if poi_mask.any() else torch.tensor([], device=device),
            k_list=k_lists
        )
        all_metrics.append(metrics)

    return total_loss / len(dataloader), merge_metrics(all_metrics)

# ------------------ 验证和测试 ------------------
@torch.no_grad()
def evaluate(model, dataloader, device, alpha=0.5):
    model.eval()
    total_loss = 0
    all_metrics = []

    for batch in tqdm(dataloader, desc="Evaluating"):
        src = batch['input_ids'].to(device)
        types = batch['input_types'].to(device)
        targets = batch['target_id'].to(device)
        target_types = batch['target_type'].to(device)
        cls_features = batch['cls_features'].to(device)
        domain_labels = batch['domain_label'].to(device)

        mdd_logits, poi_logits, type_logits = model(src, types, cls_features, domain_labels)

        mdd_logits = mdd_logits[:, -1, :]
        poi_logits = poi_logits[:, -1, :]

        mdd_mask = (target_types == 0)
        poi_mask = (target_types == 1)

        loss_mdd = F.cross_entropy(mdd_logits[mdd_mask], targets[mdd_mask], ignore_index=-100) if mdd_mask.any() else torch.tensor(0.0, device=device)
        loss_poi = F.cross_entropy(poi_logits[poi_mask], targets[poi_mask], ignore_index=-100) if poi_mask.any() else torch.tensor(0.0, device=device)
        loss_type = F.cross_entropy(type_logits[:, -1, :], target_types, ignore_index=-100)

        loss = loss_mdd + loss_poi + alpha * loss_type
        total_loss += loss.item()

        mdd_logits_selected = mdd_logits[mdd_mask] if mdd_mask.any() else torch.tensor([], device=device)
        poi_logits_selected = poi_logits[poi_mask] if poi_mask.any() else torch.tensor([], device=device)
        mdd_targets = targets[mdd_mask] if mdd_mask.any() else torch.tensor([], device=device)
        poi_targets = targets[poi_mask] if poi_mask.any() else torch.tensor([], device=device)

        metrics = compute_topk_metrics(mdd_logits_selected, poi_logits_selected, mdd_targets, poi_targets, k_list=k_lists)
        all_metrics.append(metrics)

    return total_loss / len(dataloader), merge_metrics(all_metrics)

# ------------------ 自回归推理 + BLEU 评估 ------------------
# @torch.no_grad()
# def evaluate_autoregressive_bleu_beam(model, ar_eval_routes, bert_cls_dict, device, beam_width=5, max_len=10):
#     model.eval()
#     smoother = SmoothingFunction().method1
#     bleu_scores_per_step = [[] for _ in range(max_len)]
#     gt_all_titles, pred_all_titles = [], []
#
#     for full_traj in tqdm(ar_eval_routes, desc='Beam Search Eval'):
#         if len(full_traj) < 3:
#             continue
#
#         init_points = full_traj[:2]
#         gt = full_traj[2:]
#         init_ids = [p[1] for p in init_points]
#         init_types = [0 if p[0] == 'mdd' else 1 for p in init_points]
#
#         init_cls = [bert_cls_dict.get((p[0], p[1]), torch.zeros(768)) for p in init_points]
#         init_cls = torch.stack(init_cls).unsqueeze(0).to(device)
#
#         beams = [{
#             'ids': init_ids,
#             'types': init_types,
#             'cls': init_cls,
#             'score': 0.0
#         }]
#
#         max_gen_len = min(max_len, len(full_traj) - 2)
#
#         for step in range(max_gen_len):
#             new_beams = []
#             for beam in beams:
#                 curr_input = torch.tensor([beam['ids']], dtype=torch.long, device=device)
#                 curr_types = torch.tensor([beam['types']], dtype=torch.long, device=device)
#                 domain_labels = torch.tensor([0], dtype=torch.long, device=device)
#
#                 mdd_logits, poi_logits, type_logits = model(curr_input, curr_types, beam['cls'], domain_labels)
#
#                 last_type_logits = type_logits[:, -1, :]
#                 last_type_probs = F.log_softmax(last_type_logits, dim=-1).squeeze()
#
#                 for pred_type in [0, 1]:
#                     if pred_type == 0:
#                         logits = mdd_logits[:, -1, :]
#                     else:
#                         logits = poi_logits[:, -1, :]
#                     probs = F.log_softmax(logits, dim=-1).squeeze()
#                     topk_probs, topk_ids = torch.topk(probs, beam_width)
#
#                     for i in range(beam_width):
#                         new_id = topk_ids[i].item()
#                         new_score = beam['score'] + last_type_probs[pred_type].item() + topk_probs[i].item()
#                         new_ids = beam['ids'] + [new_id]
#                         new_types = beam['types'] + [pred_type]
#
#                         key = (pred_type, new_id)
#                         cls_feat = bert_cls_dict.get(key, torch.zeros(768)).to(device)
#                         new_cls = torch.cat([beam['cls'], cls_feat.unsqueeze(0).unsqueeze(0)], dim=1)
#
#                         new_beams.append({
#                             'ids': new_ids,
#                             'types': new_types,
#                             'cls': new_cls,
#                             'score': new_score
#                         })
#
#             beams = sorted(new_beams, key=lambda x: x['score'], reverse=True)[:beam_width]
#
#             for beam in beams:
#                 ref = [p[1] for p in gt[:step + 1]]
#                 hyp = beam['ids'][2:2 + step + 1]
#                 if len(ref) >= len(hyp):
#                     bleu = sentence_bleu([ref], hyp, weights=(1.0,), smoothing_function=smoother)
#                     bleu_scores_per_step[step].append(bleu)
#
#         # 选出得分最高的路径
#         best_beam = beams[0]
#         def get_title(ptype, pid):
#             return id2mdd.get(pid, f"mdd_{pid}") if ptype in ['mdd', 0] else id2poi.get(pid, f"poi_{pid}")
#
#         gt_titles = " -> ".join([get_title(p[0], p[1]) for p in full_traj])
#         pred_titles = " -> ".join([get_title(pt, pid) for pt, pid in zip(best_beam['types'], best_beam['ids'])])
#         gt_all_titles.append(gt_titles)
#         pred_all_titles.append(pred_titles)
#
#     for i, (gt_titles, pred_titles) in enumerate(zip(gt_all_titles, pred_all_titles), 1):
#         print(f"\n========= Truth vs Beam Prediction {i} =========")
#         print(f"Ground Truth: {gt_titles}")
#         print(f"Predicted   : {pred_titles}")
#
#     avg_bleus = [
#         sum(step_scores) / len(step_scores) if step_scores else 0.0
#         for step_scores in bleu_scores_per_step
#     ]
#     return avg_bleus

# ------------------ 可视化 ------------------
def plot_metrics(train_losses, val_losses, train_metrics, val_metrics,k_list):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(20, 12))

    # ---- 子图1：Loss曲线 ----
    plt.subplot(3, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Over Epochs')
    plt.legend()

    # # ---- 子图2：BLEU曲线 ----
    # plt.subplot(3, 2, 2)
    # plt.plot(range(1, len(bleu_scores) + 1), bleu_scores, label='BLEU')
    # plt.xlabel('Step')
    # plt.ylabel('Avg BLEU')
    # plt.title('Auto-Regressive BLEU per Step')
    # plt.legend()

    # ---- 子图3：Top-K HitRate ----
    plt.subplot(3, 2, 2)
    for k in k_list:
        train_values = [epoch_metrics[k]["hit_rate"] for epoch_metrics in train_metrics]
        val_values = [epoch_metrics[k]["hit_rate"] for epoch_metrics in val_metrics]
        plt.plot(epochs, train_values, label=f'Train@{k}')
        plt.plot(epochs, val_values, linestyle='--', label=f'Val@{k}')
    plt.xlabel('Epoch')
    plt.ylabel('HitRate')
    plt.title('Top-K HitRate')
    plt.legend()

    # ---- 子图4：Top-K MRR ----
    plt.subplot(3, 2, 3)
    for k in k_list:
        train_values = [epoch_metrics[k]["mrr"] for epoch_metrics in train_metrics]
        val_values = [epoch_metrics[k]["mrr"] for epoch_metrics in val_metrics]
        plt.plot(epochs, train_values, label=f'Train@{k}')
        plt.plot(epochs, val_values, linestyle='--', label=f'Val@{k}')
    plt.xlabel('Epoch')
    plt.ylabel('MRR')
    plt.title('Top-K MRR')
    plt.legend()

    # ---- 子图5：Top-K NDCG ----
    plt.subplot(3, 2, 4)
    for k in k_list:
        train_values = [epoch_metrics[k]["ndcg"] for epoch_metrics in train_metrics]
        val_values = [epoch_metrics[k]["ndcg"] for epoch_metrics in val_metrics]
        plt.plot(epochs, train_values, label=f'Train@{k}')
        plt.plot(epochs, val_values, linestyle='--', label=f'Val@{k}')
    plt.xlabel('Epoch')
    plt.ylabel('NDCG')
    plt.title('Top-K NDCG')
    plt.legend()

    plt.tight_layout()
    plt.show()


