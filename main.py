import torch
import random
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from model import DualLayerRecModel
from train import (
    train_one_epoch,
    evaluate,
    print_metrics,
    evaluate_autoregressive_bleu,
    plot_metrics
)
from config import mdd2id, id2mdd, poi2id, id2poi
from transformers import AutoModel,AutoTokenizer
from tqdm import tqdm
import json

# ------------------ 数据处理 ------------------
# 加载路线
def load_routes(file_paths):
    routes = []
    for path in tqdm(file_paths, desc="Loading files"):
        count = 0
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                route = json.loads(line)
                routes.append(route)
                count += 1
        print(f"✓ Loaded {count} routes from {Path(path).name}")
    return routes

# 创建 ID 映射字典
# mdd2id = {}
# id2mdd = {}
# poi2id = {}
# id2poi = {}
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

def extract_cls_vectors(model,tokenizer,all_data):
    """
    all_data: List of (type, id, details)
    返回：dict[(type, id)] = Tensor[768]
    """
    bert_cls_dict = {}
    seen_keys = set()
    for traj in tqdm(all_data, desc="Extracting BERT [CLS] vectors from trajectories"):  # 例如 data 是所有轨迹的合集
        for typ, pid, details in traj:
            key = (typ, pid)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            inputs = tokenizer(details, return_tensors='pt', truncation=True, padding='max_length', max_length=50)
            inputs = {k: v.cuda() for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
                cls_vec = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu()  # 取 [CLS]
                bert_cls_dict[key] = cls_vec
    return bert_cls_dict
def generate_sub_trajectories(trajectory, max_subs, min_len=2):
    """
    trajectory: List[Tuple[type, id, details]]
    max_subs: 限制最多生成多少个子轨迹（None 表示不限制）
    """
    sub_trajectories = []
    for i in range(min_len, len(trajectory)):
        input_seq = trajectory[:i]
        target_point = trajectory[i]
        sub_trajectories.append((input_seq, target_point))
        if max_subs and len(sub_trajectories) >= max_subs:
            break
    return sub_trajectories

class POITrajectoryDataset(Dataset):
    '''
        data：二维列表，每一个元素是路线（list of (type, id, details)）
        bert_cls_dict：dict[(type, id)] -> Tensor[768]，预先提取好的 BERT CLS 向量
        max_subs_per_traj: 限制每条轨迹最多生成多少个子轨迹
    '''
    def __init__(self, data, bert_out_dim, bert_cls_dict, max_len=10,max_subs_per_traj=10):
        self.samples = []
        self.max_len = min(max_len,max_subs_per_traj+1)
        self.bert_cls_dict = bert_cls_dict  # {('poi', '123'): torch.tensor([...])}
        # self.return_full_traj = return_full_traj
        self.bert_out_dim = bert_out_dim

        for input_seq, target in data:
            self.samples.append((input_seq, target))
        # for traj in data:
        #     subs = generate_sub_trajectories(traj, max_subs=max_subs_per_traj)
        #     for input_seq, target in subs:
        #         if return_full_traj:
        #             self.samples.append((input_seq, target, traj))
        #         else:
        #             self.samples.append((input_seq, target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # if self.return_full_traj:
        #     input_seq, target, full_traj = self.samples[idx]
        # else:
        #     input_seq, target = self.samples[idx]
        input_seq, target = self.samples[idx]
        input_ids = [x[1] for x in input_seq]
        input_types = [0 if x[0] == 'mdd' else 1 for x in input_seq]

        # 获取已预提取的BERT CLS特征向量
        cls_features = []
        for x in input_seq:
            key = (x[0], x[1])
            if key in self.bert_cls_dict:
                cls_features.append(self.bert_cls_dict[key])
            else:
                # 如果缺失则用0向量填充
                cls_features.append(torch.zeros(self.bert_out_dim))

        # Padding 到 max_len
        pad_len = self.max_len - len(input_ids)
        if pad_len > 0:
            input_ids += [0] * pad_len
            input_types += [0] * pad_len
            cls_features += [torch.zeros(self.bert_out_dim)] * pad_len
        else:
            input_ids = input_ids[:self.max_len]
            input_types = input_types[:self.max_len]
            cls_features = cls_features[:self.max_len]

        result = {
            'input_ids': torch.tensor(input_ids),
            'input_types': torch.tensor(input_types),
            'cls_features': torch.stack(cls_features),  # shape: (max_len, 768)
            'target_id': torch.tensor(target[1]),
            'target_type': torch.tensor(0 if target[0] == 'mdd' else 1),
        }

        # if self.return_full_traj:
        #     result['full_traj'] = full_traj
        return result

def main():
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")
    model = AutoModel.from_pretrained('distilbert-base-multilingual-cased')
    bert_out_dim = model.config.hidden_size
    model.eval().cuda()

    # 自动收集文件路径
    # mfw_path = Path('./datasets/precleaning/MFW/route_step4.jsonl')
    fs_dir = Path('./datasets/precleaning/FourSquare')
    # foursquare_paths = sorted(fs_dir.glob('trajectories_batch*.jsonl'))
    foursquare_paths = [
         fs_dir / f'trajectories_batch{i}.jsonl' for i in range(1)
    ]

    # print(f"Loading MFW routes from {mfw_path}")
    # print(f"Loading FourSquare routes from {len(foursquare_paths)} files")

    # 加载数据
    # mfw_routes = load_routes([mfw_path])
    fs_routes = load_routes(foursquare_paths)

    # 拆分自回归评估数据（前10条MFW路线），剩余的用于训练
    # ar_eval_routes_raw = mfw_routes[:10] + fs_routes[:10]
    ar_eval_routes_raw = fs_routes[:10]
    # mfw_routes_remain = mfw_routes[10:]
    fs_routes_remain = fs_routes[10:]

    # 合并训练用数据（FourSquare + MFW剩余）
    # train_eval_routes_raw = mfw_routes_remain + fs_routes_remain
    train_eval_routes_raw = fs_routes_remain

    # 编码轨迹
    ar_eval_routes = []
    for route in ar_eval_routes_raw:
        if 'trajectory' in route:
            encoded = encode_trajectory(route['trajectory'])
            if len(encoded) >= 3:
                ar_eval_routes.append(encoded)

    encoded_routes = []
    for route in train_eval_routes_raw:
        if 'trajectory' in route:
            encoded = encode_trajectory(route['trajectory'])
            if len(encoded) >= 3:
                encoded_routes.append(encoded)

    # 提取 BERT CLS 向量（包含全部数据：训练+自回归）
    bert_cls_dict = extract_cls_vectors(model, tokenizer, encoded_routes + ar_eval_routes)
    torch.save(bert_cls_dict, 'bert_cls_dict.pt')

    # 生成子轨迹样本（仅用于训练的轨迹）
    max_subs_per_traj = 9
    all_sub_routes = []
    for route in encoded_routes:
        subs = generate_sub_trajectories(route, max_subs=max_subs_per_traj)
        all_sub_routes.extend(subs)

    # 划分数据集
    random.seed(42)
    random.shuffle(all_sub_routes)
    num_total = len(all_sub_routes)
    train_split = int(0.8 * num_total)
    val_split = int(0.9 * num_total)
    train_data = all_sub_routes[:train_split]
    val_data = all_sub_routes[train_split:val_split]
    test_data = all_sub_routes[val_split:]

    print(f"子轨迹划分：Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    print(f"自回归评估轨迹数量: {len(ar_eval_routes)}")

    # Dataloader 构建（不包括自回归轨迹）
    bert_cls_dict = torch.load('bert_cls_dict.pt')
    train_loader = DataLoader(
        POITrajectoryDataset(train_data, bert_out_dim, bert_cls_dict=bert_cls_dict, max_len=10),
        batch_size=512, shuffle=True)

    val_loader = DataLoader(
        POITrajectoryDataset(val_data, bert_out_dim, bert_cls_dict=bert_cls_dict, max_len=10),
        batch_size=512)

    test_loader = DataLoader(
        POITrajectoryDataset(test_data, bert_out_dim, bert_cls_dict=bert_cls_dict, max_len=10),
        batch_size=512)

    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DualLayerRecModel(
        mdd_vocab_size=len(mdd2id),
        poi_vocab_size=len(poi2id),
        embedding_dim=128,
        num_heads=4,
        num_layers=2,
        bert_out_dim=bert_out_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    best_val_loss = float('inf')
    train_losses, val_losses = [], []

    # 训练阶段
    for epoch in range(20):
        print(f"\nEpoch {epoch + 1}")

        train_loss, train_metrics = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_metrics = evaluate(model, val_loader, device)

        print(f"\nTrain Loss: {train_loss:.4f}")
        print_metrics(train_metrics, prefix="Train ")
        print(f"Val Loss: {val_loss:.4f}")
        print_metrics(val_metrics, prefix="Val ")

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pt')
            print("Best model saved.")

    # 测试阶段
    test_loss, test_metrics = evaluate(model, test_loader, device)
    print(f"Test Loss: {test_loss:.4f}")
    print_metrics(test_metrics, prefix="Test ")

    # 自回归 BLEU 评估（仅使用前10条 MFW 轨迹）
    print("Running autoregressive BLEU evaluation...")
    bleu_scores = evaluate_autoregressive_bleu(model, ar_eval_routes, bert_cls_dict, device)

    # 画出损失函数与 BLEU 曲线
    plot_metrics(train_losses, val_losses, bleu_scores)

if __name__ == '__main__':
    main()

# class POITrajectoryDataset(Dataset):
#     '''
#         data： 二维列表，每一个元素是路线的列表，路线中的每一项是[type, id, details]
#         return_full_traj: 是否保留每条子序列对应的完整轨迹，仅在测试集为 True
#     '''
#     def __init__(self, data, max_len=10, tokenizer=None, return_full_traj=False):
#         self.samples = []
#         self.max_len = max_len
#         self.tokenizer = tokenizer
#         self.return_full_traj = return_full_traj
#
#         for traj in data:
#             subs = generate_sub_trajectories(traj)
#             for input_seq, target in subs:
#                 if return_full_traj:
#                     self.samples.append((input_seq, target, traj))  # 追加完整轨迹
#                 else:
#                     self.samples.append((input_seq, target))
#
#     def __len__(self):
#         return len(self.samples)
#
#     def __getitem__(self, idx):
#         if self.return_full_traj:
#             input_seq, target, full_traj = self.samples[idx]
#         else:
#             input_seq, target = self.samples[idx]
#
#         input_ids = [x[1] for x in input_seq]
#         input_types = [0 if x[0] == 'mdd' else 1 for x in input_seq]
#         details = [x[2] for x in input_seq]
#
#         # 对每个details进行BERT分词
#         detail_tokens = [self.tokenizer(d, truncation=True, padding='max_length', max_length=50) for d in details]
#         input_ids_details = [t['input_ids'] for t in detail_tokens]
#         attention_mask_details = [t['attention_mask'] for t in detail_tokens]
#
#         pad_len = self.max_len - len(input_ids)
#         if pad_len > 0:
#             input_ids += [0] * pad_len
#             input_types += [0] * pad_len
#             input_ids_details += [[0] * 50] * pad_len
#             attention_mask_details += [[0] * 50] * pad_len
#         else:
#             input_ids = input_ids[:self.max_len]
#             input_types = input_types[:self.max_len]
#             input_ids_details = input_ids_details[:self.max_len]
#             attention_mask_details = attention_mask_details[:self.max_len]
#
#         result = {
#             'input_ids': torch.tensor(input_ids),
#             'input_types': torch.tensor(input_types),
#             'details_tokens': {
#                 'input_ids': torch.tensor(input_ids_details),
#                 'attention_mask': torch.tensor(attention_mask_details)
#             },
#             'target_id': torch.tensor(target[1]),
#             'target_type': torch.tensor(0 if target[0] == 'mdd' else 1),
#         }
#
#         if self.return_full_traj:
#             result['full_traj'] = full_traj  # 返回完整路线（list of [type, id, details]）
#         return result