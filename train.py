import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from config import id2mdd, id2poi
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

        optimizer.zero_grad()
        mdd_logits, poi_logits, type_logits = model(src, types, cls_features)

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
            k_list=[5, 10, 15, 20]
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
        # details_input_ids = batch['details_tokens']['input_ids'].to(device)
        # details_attention_mask = batch['details_tokens']['attention_mask'].to(device)

        mdd_logits, poi_logits, type_logits = model(src, types, cls_features)

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

        metrics = compute_topk_metrics(mdd_logits_selected, poi_logits_selected, mdd_targets, poi_targets, k_list=[5, 10, 15, 20])
        all_metrics.append(metrics)

    return total_loss / len(dataloader), merge_metrics(all_metrics)

# ------------------ 自回归推理 + BLEU 评估 ------------------
@torch.no_grad()
def evaluate_autoregressive_bleu(model, traj_list, bert_cls_dict, device, max_len=10):
    model.eval()
    bleu_scores_per_step = [[] for _ in range(max_len)]
    smoother = SmoothingFunction().method1

    print("\n===== Autoregressive Trajectory Predictions =====")
    gt_all_titles = []
    pred_all_titles = []
    for full_traj in tqdm(traj_list, desc='AutoRegressive Eval'):
        if len(full_traj) < 3:
            continue

        init_points = full_traj[:2]
        gen_ids = [p[1] for p in init_points]
        gen_types = []
        for p in init_points:
            if p[0] == 'mdd':
                gen_types.append(0)
            else: gen_types.append(1)

        # 初始化 CLS 特征
        cls_features_list = []
        for p in init_points:
            key = (p[0], p[1])
            cls_feat = bert_cls_dict.get(key, torch.zeros(768))
            cls_features_list.append(cls_feat)
        cls_features = torch.stack(cls_features_list).unsqueeze(0).to(device)

        max_gen_len = min(max_len, len(full_traj) - 2)

        for step in range(max_gen_len):
            curr_input = torch.tensor([gen_ids], dtype=torch.long, device=device)
            curr_types = torch.tensor([gen_types], dtype=torch.long, device=device)

            mdd_logits, poi_logits, type_logits = model(curr_input, curr_types, cls_features)

            pred_type = torch.argmax(type_logits[:, -1, :], dim=-1).item()
            gen_types.append(pred_type)

            if pred_type == 0:
                logits = mdd_logits[:, -1, :]
            else:
                logits = poi_logits[:, -1, :]

            pred_id = torch.argmax(logits, dim=-1).item()
            gen_ids.append(pred_id)

            # 新的 CLS 特征
            key = (pred_type, pred_id)
            cls_feat = bert_cls_dict.get(key, torch.zeros(768)).to(device)
            cls_features = torch.cat([cls_features, cls_feat.unsqueeze(0).unsqueeze(0)], dim=1)

            # BLEU 分数
            ref = [p[1] for p in full_traj[2:2 + step + 1]]
            hyp = gen_ids[2:2 + step + 1]

            if len(ref) >= len(hyp):
                bleu = sentence_bleu([ref], hyp, weights=(1.0,), smoothing_function=smoother)
                bleu_scores_per_step[step].append(bleu)

        # 打印预测与真实轨迹对比
        def get_title(ptype, pid):
            if ptype == 'mdd' or ptype == 0:
                return id2mdd.get(pid, f"mdd_{pid}")
            else:
                return id2poi.get(pid, f"poi_{pid}")

        gt_titles = " -> ".join([get_title(p[0], p[1]) for p in full_traj])
        pred_titles = " -> ".join([get_title(pt, pid) for pt, pid in zip(gen_types, gen_ids)])

        gt_all_titles.append(gt_titles)
        pred_all_titles.append(pred_titles)

    route_count = 1
    for gt_titles, pred_titles in zip(gt_all_titles, pred_all_titles):
        print(f"============================================================ Truth Route & Predicted Route {route_count} ============================================================")
        print(f"Ground Truth: {gt_titles}")
        print(f"Predicted   : {pred_titles}")
        route_count += 1

    # 平均 BLEU 分数
    avg_bleus = [
        sum(step_scores) / len(step_scores) if step_scores else 0.0
        for step_scores in bleu_scores_per_step
    ]
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

