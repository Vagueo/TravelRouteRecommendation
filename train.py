import torch
import torch.nn.functional as F
from tqdm import tqdm
import json
from torch.utils.data import Dataset

# ------------------ 数据处理 ------------------
# 加载路线
def load_routes(file_paths):
    routes = []
    for path in tqdm(file_paths, desc="Loading files"):
        with open(path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc=f" Reading {path.name}", leave=False):
                route = json.loads(line)
                routes.append(route)
    return routes

# 创建 ID 映射字典
mdd2id = {}
id2mdd = {}
poi2id = {}
id2poi = {}
mdd_counter = 0
poi_counter = 0

# 给路线中的mdd和poi的id重新编码（因为路线中的mdd和poi的id并非连续的）
def encode_trajectory(trajectory):
    global mdd_counter, poi_counter
    encoded = []
    for step in trajectory:
        details = step.get("details", "")  # 如果没有 details，使用空字符串
        if step['type'] == 'mdd':
            mdd = step['mddTitle']
            if mdd not in mdd2id:
                mdd2id[mdd] = mdd_counter
                id2mdd[mdd_counter] = mdd
                mdd_counter += 1
            encoded.append(('mdd', mdd2id[mdd], details))
        elif step['type'] == 'poi':
            poi = step['poi_title']
            if poi not in poi2id:
                poi2id[poi] = poi_counter
                id2poi[poi_counter] = poi
                poi_counter += 1
            encoded.append(('poi', poi2id[poi], details))
    return encoded


# ------------------ 数据集 ------------------
class POITrajectoryDataset(Dataset):
    def __init__(self, data, max_len=20, tokenizer=None):
        self.data = data
        self.max_len = max_len
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        traj = self.data[idx]
        input_ids = [x[1] for x in traj[:-1]]
        input_types = [0 if x[0] == 'mdd' else 1 for x in traj[:-1]]
        target_ids = [x[1] for x in traj[1:]]
        target_types = [0 if x[0] == 'mdd' else 1 for x in traj[1:]]
        details = [x[2] for x in traj[:-1]]  # 获取每个地点的 details

        # BERT tokenization
        detail_tokens = [self.tokenizer(detail, truncation=True, padding='max_length', max_length=30) for detail in details]
        input_ids_details = [tokens['input_ids'] for tokens in detail_tokens]
        attention_mask_details = [tokens['attention_mask'] for tokens in detail_tokens]

        pad_len = self.max_len - len(input_ids)
        if pad_len > 0:
            input_ids += [0] * pad_len
            input_types += [0] * pad_len
            target_ids += [-100] * pad_len
            target_types += [0] * pad_len
            input_ids_details += [[0] * 30] * pad_len  # 对应 padding 30长度的 details tokens
            attention_mask_details += [[0] * 30] * pad_len  # 对应 padding 30长度的 attention_mask

        # Convert to tensors and return the batch
        return {
            'input_ids': torch.tensor(input_ids[:self.max_len]),
            'input_types': torch.tensor(input_types[:self.max_len]),
            'target_ids': torch.tensor(target_ids[:self.max_len]),
            'target_types': torch.tensor(target_types[:self.max_len]),
            'details_tokens': {
                'input_ids': torch.tensor(input_ids_details[:self.max_len]),  # [max_len, 30]
                'attention_mask': torch.tensor(attention_mask_details[:self.max_len]),  # [max_len, 30]
            },
        }

# class POITrajectoryDataset(Dataset):
#     def __init__(self, data, max_len=20, tokenizer=None):
#         self.data = data
#         self.max_len = max_len
#         self.tokenizer = tokenizer
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         traj = self.data[idx]
#         input_ids = [x[1] for x in traj[:-1]]
#         input_types = [0 if x[0] == 'mdd' else 1 for x in traj[:-1]]
#         target_ids = [x[1] for x in traj[1:]]
#         target_types = [0 if x[0] == 'mdd' else 1 for x in traj[1:]]
#         details = [x[2] for x in traj[:-1]]  # 获取每个地点的 details
#
#         # BERT tokenization
#         detail_tokens = [self.tokenizer(detail, truncation=True, padding='max_length', max_length=30)['input_ids'] for
#                          detail in details]
#
#         pad_len = self.max_len - len(input_ids)
#         if pad_len > 0:
#             input_ids += [0] * pad_len
#             input_types += [0] * pad_len
#             target_ids += [-100] * pad_len
#             target_types += [0] * pad_len
#             detail_tokens += [[0] * 30] * pad_len  # 对应 padding 30长度的 details tokens
#
#         return {
#             'input_ids': torch.tensor(input_ids[:self.max_len]),
#             'input_types': torch.tensor(input_types[:self.max_len]),
#             'target_ids': torch.tensor(target_ids[:self.max_len]),
#             'target_types': torch.tensor(target_types[:self.max_len]),
#             'details_tokens': torch.tensor(detail_tokens[:self.max_len]),  # 添加 details 的 tokenized 输入
#         }


# ------------------ 训练和评估 ------------------
def compute_topk_metrics(mdd_logits, poi_logits, mdd_targets, poi_targets, k_list=[5, 10]):
    metrics = {k: {"hit": 0, "precision": 0, "recall": 0, "count": 0} for k in k_list}
    for logit, target in zip(mdd_logits, mdd_targets):
        if target == -100: continue
        probs = torch.softmax(logit, dim=-1)
        for k in k_list:
            topk = probs.topk(k).indices
            hit = int(target in topk)
            metrics[k]["hit"] += hit
            metrics[k]["precision"] += hit / k
            metrics[k]["recall"] += hit
            metrics[k]["count"] += 1
    for logit, target in zip(poi_logits, poi_targets):
        if target == -100: continue
        probs = torch.softmax(logit, dim=-1)
        for k in k_list:
            topk = probs.topk(k).indices
            hit = int(target in topk)
            metrics[k]["hit"] += hit
            metrics[k]["precision"] += hit / k
            metrics[k]["recall"] += hit
            metrics[k]["count"] += 1
    return metrics

def merge_metrics(metrics_list):
    merged = {}
    for k in metrics_list[0].keys():
        merged[k] = {
            "hit": sum(m[k]["hit"] for m in metrics_list),
            "precision": sum(m[k]["precision"] for m in metrics_list),
            "recall": sum(m[k]["recall"] for m in metrics_list),
            "count": sum(m[k]["count"] for m in metrics_list),
        }
    return merged

def print_metrics(metrics, prefix=""):
    for k, v in metrics.items():
        hit_rate = v["hit"] / v["count"] if v["count"] else 0
        precision = v["precision"] / v["count"] if v["count"] else 0
        recall = v["recall"] / v["count"] if v["count"] else 0
        print(f"{prefix}Top-{k}: HitRate={hit_rate:.4f}, Precision={precision:.4f}, Recall={recall:.4f}")

def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    all_metrics = []

    for batch in tqdm(dataloader):
        # 不再使用 .T，直接使用 batch 中的原始维度
        src = batch['input_ids'].to(device)              # [batch_size, seq_len]
        types = batch['input_types'].to(device)          # [batch_size, seq_len]
        targets = batch['target_ids'].to(device)         # [batch_size, seq_len]
        target_types = batch['target_types'].to(device)  # [batch_size, seq_len]

        # 提取 BERT 编码用的 input_ids 和 attention_mask
        details_input_ids = batch['details_tokens']['input_ids'].to(device)        # [batch_size, seq_len]
        details_attention_mask = batch['details_tokens']['attention_mask'].to(device)  # [batch_size, seq_len]

        optimizer.zero_grad()

        # 传入模型，注意传入 attention_mask
        mdd_logits, poi_logits = model(src, types, details_input_ids, details_attention_mask)

        mdd_mask = (target_types == 0)
        poi_mask = (target_types == 1)

        loss_mdd = F.cross_entropy(mdd_logits[mdd_mask], targets[mdd_mask], ignore_index=-100) if mdd_mask.any() else torch.tensor(0.0, device=device)
        loss_poi = F.cross_entropy(poi_logits[poi_mask], targets[poi_mask], ignore_index=-100) if poi_mask.any() else torch.tensor(0.0, device=device)

        loss = loss_mdd + loss_poi
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        metrics = compute_topk_metrics(mdd_logits[mdd_mask], poi_logits[poi_mask], targets[mdd_mask], targets[poi_mask])
        all_metrics.append(metrics)

    return total_loss / len(dataloader), merge_metrics(all_metrics)

@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    all_metrics = []
    for batch in dataloader:
        src = batch['input_ids'].T.to(device)
        types = batch['input_types'].T.to(device)
        targets = batch['target_ids'].T.to(device)
        target_types = batch['target_types'].T.to(device)

        mdd_logits, poi_logits = model(src, types)

        mdd_mask = (target_types == 0)
        poi_mask = (target_types == 1)

        loss_mdd = F.cross_entropy(mdd_logits[mdd_mask], targets[mdd_mask], ignore_index=-100) if mdd_mask.any() else 0
        loss_poi = F.cross_entropy(poi_logits[poi_mask], targets[poi_mask], ignore_index=-100) if poi_mask.any() else 0
        loss = loss_mdd + loss_poi
        total_loss += loss.item()

        metrics = compute_topk_metrics(mdd_logits[mdd_mask], poi_logits[poi_mask], targets[mdd_mask], targets[poi_mask])
        all_metrics.append(metrics)

    return total_loss / len(dataloader), merge_metrics(all_metrics)

@torch.no_grad()
def autoregressive_inference(model, start_tokens, start_types, max_steps=10, top_k=1, device='cpu'):
    model.eval()
    generated_ids = start_tokens[:]
    generated_types = start_types[:]

    for _ in range(max_steps):
        src = torch.tensor(generated_ids, dtype=torch.long).unsqueeze(1).to(device)
        types = torch.tensor(generated_types, dtype=torch.long).unsqueeze(1).to(device)

        mdd_logits, poi_logits = model(src, types)

        last_type = generated_types[-1]
        next_token = None
        if last_type == 0:
            probs = torch.softmax(poi_logits[-1, 0], dim=-1)
            topk = probs.topk(top_k).indices.tolist()
            next_token = topk[0]
            next_type = 1
        else:
            probs = torch.softmax(mdd_logits[-1, 0], dim=-1)
            topk = probs.topk(top_k).indices.tolist()
            next_token = topk[0]
            next_type = 0

        generated_ids.append(next_token)
        generated_types.append(next_type)

    return list(zip(generated_types, generated_ids))

