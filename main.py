import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from model import DualLayerRecModel
from train import (
    train_one_epoch,
    evaluate,
    print_metrics,
    evaluate_autoregressive_bleu
)
from transformers import BertTokenizer,AutoTokenizer
from tqdm import tqdm
import json

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

def generate_sub_trajectories(trajectory, min_len=2):
    """
    trajectory: List[Tuple[type, id, details]]
    返回多个子轨迹，每个子轨迹为一个完整的训练样本：[输入轨迹], 目标点
    """
    sub_trajectories = []
    for i in range(min_len, len(trajectory)):
        input_seq = trajectory[:i]  # 前i个
        target_point = trajectory[i]  # 第i+1个作为预测目标
        sub_trajectories.append((input_seq, target_point))
    return sub_trajectories

# ------------------ 构造数据集 ------------------
class POITrajectoryDataset(Dataset):
    '''
        data： 二维列表，每一个元素是路线的列表，然后路线的列表中的数据格式是[type, id, details]
        其中，type为poi或是mdd
            id为poi_id或mddId
            details为poi或mdd的details
    '''
    def __init__(self, data, max_len=10, tokenizer=None):
        self.samples = []
        self.max_len = max_len
        self.tokenizer = tokenizer
        for traj in data:
            subs = generate_sub_trajectories(traj)
            for input_seq, target in subs:
                self.samples.append((input_seq, target))
                # self.samples.append((input_seq, target, traj))  # 加上原始轨迹

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # input_seq, target, full_traj = self.samples[idx]
        input_seq, target = self.samples[idx]
        input_ids = [x[1] for x in input_seq]
        input_types = [0 if x[0] == 'mdd' else 1 for x in input_seq]
        details = [x[2] for x in input_seq]

        detail_tokens = [self.tokenizer(d, truncation=True, padding='max_length', max_length=50) for d in details]
        input_ids_details = [t['input_ids'] for t in detail_tokens]
        attention_mask_details = [t['attention_mask'] for t in detail_tokens]

        pad_len = self.max_len - len(input_ids)
        '''
            补齐
            例如: 一条线路：[A,B,C,D,E,F]被分解成了下面几个子序列(前面是输入的地点，后面是实际的地点，然后要预测下一个地点是不是C)
                1. [A,B]->C     
                2. [A,B,C]->D
                ...
            但是这样一来就有问题了，每一个数据的长度不统一，这样没法进行训练，因此得在后面补上PAD，但是PAD不参与loss计算，只是为了让每一条数据集长度统一。
        '''
        if pad_len > 0:
            input_ids += [0] * pad_len
            input_types += [0] * pad_len
            input_ids_details += [[0] * 50] * pad_len
            attention_mask_details += [[0] * 50] * pad_len
        else:
            input_ids = input_ids[:self.max_len]
            input_types = input_types[:self.max_len]
            input_ids_details = input_ids_details[:self.max_len]
            attention_mask_details = attention_mask_details[:self.max_len]

        return {
            'input_ids': torch.tensor(input_ids),
            'input_types': torch.tensor(input_types),
            'details_tokens': {
                'input_ids': torch.tensor(input_ids_details),
                'attention_mask': torch.tensor(attention_mask_details)
            },
            'target_id': torch.tensor(target[1]),
            'target_type': torch.tensor(0 if target[0] == 'mdd' else 1),
        }

def main():
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
    # all_routes = mfw_routes + fs_routes
    all_routes = fs_routes

    # 编码轨迹
    # encoded_routes = [encode_trajectory(route['trajectory']) for route in all_routes if 'trajectory' in route]
    # 编码轨迹并生成子轨迹样本
    encoded_routes = []
    for route in all_routes:
        if 'trajectory' in route:
            encoded = encode_trajectory(route['trajectory'])
            if len(encoded) >= 3:  # 至少要有2个输入 + 1个目标
                encoded_routes.append(encoded)
    # 划分数据集
    random.seed(42)
    random.shuffle(encoded_routes)
    num_total = len(encoded_routes)
    train_split = int(0.8 * num_total)
    val_split = int(0.9 * num_total)
    train_data = encoded_routes[:train_split]
    val_data = encoded_routes[train_split:val_split]
    test_data = encoded_routes[val_split:]

    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    # tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")

    # 构建 Dataloader
    train_loader = DataLoader(
        POITrajectoryDataset(train_data, tokenizer=tokenizer),
        batch_size=2, shuffle=True)

    val_loader = DataLoader(
        POITrajectoryDataset(val_data, tokenizer=tokenizer),
        batch_size=2)

    test_loader = DataLoader(
        POITrajectoryDataset(test_data, tokenizer=tokenizer),
        batch_size=2)

    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DualLayerRecModel(
        mdd_vocab_size=len(mdd2id),
        poi_vocab_size=len(poi2id),
        embedding_dim=128,
        num_heads=4,
        num_layers=2,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    best_val_loss = float('inf')
    train_losses, val_losses = [], []

    # 训练模型
    for epoch in range(20):
        print(f"\nEpoch {epoch+1}")

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

    # 测试集评估
    test_loss, test_metrics = evaluate(model, test_loader, device)
    print(f"Test Loss: {test_loss:.4f}")
    print_metrics(test_metrics, prefix="Test ")

    # 可视化 Loss 曲线
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses)+1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curve')
    plt.grid(True)
    plt.show()

    print("Running autoregressive BLEU evaluation...")
    bleu_scores = evaluate_autoregressive_bleu(model, test_loader, device, max_len=10)

    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(bleu_scores) + 1), bleu_scores, marker='o')
    plt.xlabel("Generated Step")
    plt.ylabel("Average BLEU Score")
    plt.title("BLEU Score vs Generation Step")
    plt.grid(True)
    plt.savefig("bleu_vs_step.png")
    plt.show()

if __name__ == '__main__':
    main()
