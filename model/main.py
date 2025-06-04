import torch
import random
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from model import DualLayerRecModel,MultiHeadAttentionPooling
from train import (
    train_one_epoch,
    evaluate,
    print_metrics,
    plot_metrics
)
from config import (
    device, batch_size,
    tokenizer, bert, bert_out_dim,
    max_len, embedding_dim, num_heads, num_layers, lr, epochs, weight_decay,
    k_lists
)
from split import mdd2id, poi2id, split_all_datasets
from tqdm import tqdm
import json
from sklearn.model_selection import train_test_split
from collections import Counter
random.seed(42)

def load_sub_routes(file_paths):
    sub_routes = []
    count = 0
    with open(file_paths, 'r', encoding='utf-8') as f:
        for line in f:
            route = json.loads(line)
            sub_routes.append(route)
            count += 1
    print(f"\n✓ Loaded {count} sub routes from {file_paths}")
    return sub_routes

def extract_cls_vectors(bert_model, tokenizer, all_data, device='cuda'):
    bert_model.eval()
    text_extractor = MultiHeadAttentionPooling(hidden_size=bert_model.config.hidden_size).to(device)
    text_extractor.eval()

    bert_cls_dict = {}
    seen_keys = set()

    with torch.no_grad():
        for traj in tqdm(all_data, desc="Extracting BERT vectors from trajectories"):
            for loc_type, loc_id, details, source in traj:
                key = (source, loc_type, loc_id)
                if key in seen_keys:
                    continue
                seen_keys.add(key)

                inputs = tokenizer(details, truncation=True, max_length=128, padding='max_length', return_tensors='pt')
                input_ids = inputs['input_ids'].to(device)
                attention_mask = inputs['attention_mask'].to(device)

                outputs = bert_model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
                hidden_states = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]

                # 使用多头注意力池化提取文本特征
                text_feat = text_extractor(hidden_states, attention_mask)  # [batch_size, hidden_size]
                bert_cls_dict[key] = text_feat.squeeze(0).cpu()

    return bert_cls_dict
class POITrajectoryDataset(Dataset):
    '''
        data：二维列表，每一个元素是路线（list of (type, id, details)）
        bert_cls_dict：dict[(type, id)] -> Tensor[768]，预先提取好的 BERT CLS 向量
        max_len: 限制每条子轨迹的最大地点数
        max_subs_per_traj: 限制每条轨迹最多生成多少个子轨迹
    '''
    def __init__(self, data, bert_out_dim, bert_cls_dict, domain_labels=None, max_len=10):
        self.samples = []
        self.max_len = max_len
        self.bert_cls_dict = bert_cls_dict
        self.bert_out_dim = bert_out_dim

        # domain_labels 对应每条样本的标签，长度应和 data 相同
        if domain_labels is None:
            domain_labels = [0] * len(data)  # 默认为0
        assert len(domain_labels) == len(data)

        for (input_seq, target), dom_label in zip(data, domain_labels):
            self.samples.append((input_seq, target, dom_label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_seq, target, domain_label = self.samples[idx]
        input_ids = [x[1] for x in input_seq]
        input_types = [0 if x[0] == 'mdd' else 1 for x in input_seq]

        cls_features = []
        for x in input_seq:
            key = (x[3], x[0], x[1])
            if key in self.bert_cls_dict:
                cls_features.append(self.bert_cls_dict[key])
            else:
                cls_features.append(torch.zeros(self.bert_out_dim))

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
            'cls_features': torch.stack(cls_features),
            'target_id': torch.tensor(target[1]),
            'target_type': torch.tensor(0 if target[0] == 'mdd' else 1),
            'domain_label': torch.tensor(domain_label)  # 每条样本独立domain_label
        }

        return result

def run_training_validation_loop(model, train_loader, val_loader, stage_name, save_path, patience=5):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val_loss = float('inf')
    early_stop_counter = 0
    train_losses, val_losses = [], []
    all_train_metrics = []
    all_valid_metrics = []

    for epoch in range(epochs):
        print(f"\n[{stage_name}] Epoch {epoch + 1}")
        train_loss, train_metrics = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_metrics = evaluate(model, val_loader, device)

        print(f"\n[{stage_name}] Train Loss: {train_loss:.4f}")
        print_metrics(train_metrics, prefix="Train ")
        print(f"[{stage_name}] Val Loss: {val_loss:.4f}")
        print_metrics(val_metrics, prefix="Val ")

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        all_train_metrics.append(train_metrics)
        all_valid_metrics.append(val_metrics)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            early_stop_counter = 0  # 重置
            print(f'Best model has been saved{save_path}')
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print("Early stopping triggered.")
                break
    return train_losses, val_losses, all_train_metrics, all_valid_metrics

def extract_contiguous_subseqs(seq, min_len=3):
    """
    输入：seq——一个轨迹序列（list），里面每个元素本身是一个四元组列表。
    输出：所有长度从 min_len 到 len(seq) 的连续子序列（用 tuple(tuple()) 表示，可哈希）。
    """
    L = len(seq)
    subseqs = set()
    for l in range(min_len, L + 1):
        for i in range(0, L - l + 1):
            # 将每个点都转成 tuple，这样整个子序列就可以 hash 存到 set 里
            window = tuple(tuple(pt) for pt in seq[i : i + l])
            subseqs.add(window)
    return subseqs
# def split_trajectories(encoded_routes, train_ratio=0.8, val_ratio=0.1):
#     random.shuffle(encoded_routes)
#     n = len(encoded_routes)
#     train_end = int(n * train_ratio)
#     val_end = int(n * (train_ratio + val_ratio))
#     return encoded_routes[:train_end], encoded_routes[train_end:val_end], encoded_routes[val_end:]
#
# def get_kfold_subsets(trajectories, k=5, input_len=3, seed=42):
#     """
#     输入原始轨迹列表，输出K折划分后的 子轨迹对 (input_seq, target)
#     """
#     kf = KFold(n_splits=k, shuffle=True, random_state=seed)
#     all_subsets = []
#
#     traj_indices = list(range(len(trajectories)))
#     for fold, (train_idx, val_idx) in enumerate(kf.split(traj_indices)):
#         train_trajs = [trajectories[i] for i in train_idx]
#         val_trajs = [trajectories[i] for i in val_idx]
#
#         train_subs = [sub for traj in train_trajs for sub in generate_sub_trajectories_fixed_window(traj, input_len=input_len)]
#         val_subs = [sub for traj in val_trajs for sub in generate_sub_trajectories_fixed_window(traj, input_len=input_len)]
#
#         all_subsets.append((train_subs, val_subs))
#
#     return all_subsets

def main():
    bert.eval().cuda()
    mfw_encoded, mfw_virtual_encoded = split_all_datasets()
    # 提取 BERT CLS 向量
    bert_cls_dict_real = extract_cls_vectors(bert, tokenizer, mfw_encoded)
    bert_cls_dict_virtual = extract_cls_vectors(bert, tokenizer, mfw_virtual_encoded)
    bert_cls_dict = {**bert_cls_dict_virtual, **bert_cls_dict_real}  # real优先覆盖virtual
    torch.save(bert_cls_dict, 'bert_cls_dict.pt')

    # Stage 1: MFW虚拟生成的路线进行预训练
    # 预训练阶段未处理数据泄露的部分
    mfw_virtual_train_subs = load_sub_routes("D:/Project/pythonProject/mfwscrapy/datasets/pretraining/mfw_virtual_train_subs.jsonl")
    mfw_virtual_val_subs = load_sub_routes("D:/Project/pythonProject/mfwscrapy/datasets/pretraining/mfw_virtual_val_subs.jsonl")

    train_dataset_mfw_virtual = POITrajectoryDataset(mfw_virtual_train_subs, bert_out_dim, bert_cls_dict, max_len=max_len,
                                            domain_labels=[1] * len(mfw_virtual_train_subs))
    val_dataset_mfw_virtual = POITrajectoryDataset(mfw_virtual_val_subs, bert_out_dim, bert_cls_dict, max_len=max_len,
                                          domain_labels=[1] * len(mfw_virtual_val_subs))

    print(f"\n阶段1 子轨迹数: Train={len(train_dataset_mfw_virtual)}, Val={len(val_dataset_mfw_virtual)}"), # Test={len(test_dataset_mfw_virtual)}")

    # 预训练阶段处理了数据泄露的部分
    mfw_virtual_train_subs_filtered = load_sub_routes(
        "D:/Project/pythonProject/mfwscrapy/datasets/pretraining/mfw_virtual_train_subs_filtered.jsonl")
    mfw_virtual_val_subs_filtered = load_sub_routes(
        "D:/Project/pythonProject/mfwscrapy/datasets/pretraining/mfw_virtual_val_subs_filtered.jsonl")

    train_dataset_mfw_virtual_filtered = POITrajectoryDataset(mfw_virtual_train_subs_filtered, bert_out_dim, bert_cls_dict,
                                                     max_len=max_len,
                                                     domain_labels=[1] * len(mfw_virtual_train_subs_filtered))
    val_dataset_mfw_virtual_filtered = POITrajectoryDataset(mfw_virtual_val_subs_filtered, bert_out_dim, bert_cls_dict, max_len=max_len,
                                                   domain_labels=[1] * len(mfw_virtual_val_subs_filtered))

    print(f"\n阶段1 子轨迹数: Train={len(train_dataset_mfw_virtual_filtered)}, Val={len(val_dataset_mfw_virtual_filtered)}")

    model = DualLayerRecModel(
        mdd_vocab_size=len(mdd2id) + 1,
        poi_vocab_size=len(poi2id) + 1,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        bert_out_dim=bert_out_dim,
    ).to(device)

    # run_training_validation_loop(
    #     model,
    #     DataLoader(train_dataset_mfw_virtual, batch_size=batch_size, shuffle=True),
    #     DataLoader(val_dataset_mfw_virtual, batch_size=batch_size),
    #     stage_name="阶段1(MFW虚拟路线预训练)",
    #     save_path="./best_model_stage1_MFW.pt"
    # )

    run_training_validation_loop(
        model,
        DataLoader(train_dataset_mfw_virtual_filtered, batch_size=batch_size, shuffle=True),
        DataLoader(val_dataset_mfw_virtual_filtered, batch_size=batch_size),
        stage_name="阶段1(MFW虚拟路线预训练)",
        save_path="./best_model_stage1_MFW_filtered.pt"
    )

    # Stage 2: MFW微调
    all_subs = load_sub_routes("D:/Project/pythonProject/mfwscrapy/datasets/finetuning/mfw_real_train_val_subs.jsonl")
    mfw_test_subs = load_sub_routes("D:/Project/pythonProject/mfwscrapy/datasets/finetuning/mfw_real_test_subs.jsonl")
    # ———————— 1. 准备 “全轨迹” 列表 ————————
    # 训练／验证集的全轨迹
    train_full_seqs = [
        item[0] + [item[1]]
        for item in all_subs
    ]
    # 测试集的全轨迹
    test_full_seqs = [
        item[0] + [item[1]]
        for item in mfw_test_subs
    ]

    # ———————— 2. 拆出所有连续子序列 ————————
    # 2.1 训练集中所有子序列，存为一个 set 方便快速查重
    train_subseqs = set()
    for seq in train_full_seqs:
        train_subseqs |= extract_contiguous_subseqs(seq, min_len=2)

    # 2.2 测试集中所有子序列，存为 list（允许重复，用于计数）
    test_subseqs_list = []
    for seq in test_full_seqs:
        test_subseqs_list.extend(extract_contiguous_subseqs(seq, min_len=2))

    # 训练集内部子序列重复情况
    train_subseqs_list = []
    for seq in train_full_seqs:
        train_subseqs_list.extend(extract_contiguous_subseqs(seq, min_len=2))
    train_counter = Counter(train_subseqs_list)
    train_total = len(train_subseqs_list)
    train_dup = sum(1 for c in train_counter.values() if c > 1)
    print(f"训练集中重复子序列数: {train_dup} / {train_total}，重合率 {train_dup / train_total:.2%}")

    # 测试集内部子序列重复情况
    test_counter = Counter(test_subseqs_list)
    test_total = len(test_subseqs_list)
    test_dup = sum(1 for c in test_counter.values() if c > 1)
    print(f"测试集中重复子序列数: {test_dup} / {test_total}，重合率 {test_dup / test_total:.2%}")

    filtered_train_subs = []
    for item in all_subs:
        full_seq = item[0] + [item[1]]
        sub_seqs = extract_contiguous_subseqs(full_seq, min_len=2)
        # 如果子序列中没有任何一段出现在测试集中，则保留
        if not any(sub in test_subseqs_list for sub in sub_seqs):
            filtered_train_subs.append(item)

    print(f"过滤后的训练集的子序列数为：{len(filtered_train_subs)}")

    test_overlap = sum(1 for subseq in test_subseqs_list if subseq in train_subseqs)
    print(f"测试集中在训练集中出现的子序列数: {test_overlap} / {test_total}，泄露率 {test_overlap / test_total:.2%}")

    test_dataset = POITrajectoryDataset(mfw_test_subs, bert_out_dim, bert_cls_dict, max_len=max_len,
                                        domain_labels=[0] * len(mfw_test_subs))

    train_subs, val_subs = train_test_split(filtered_train_subs, test_size=0.2, random_state=42)

    train_dataset = POITrajectoryDataset(train_subs, bert_out_dim, bert_cls_dict, max_len=max_len,
                                         domain_labels=[0] * len(train_subs))
    val_dataset = POITrajectoryDataset(val_subs, bert_out_dim, bert_cls_dict, max_len=max_len,
                                       domain_labels=[0] * len(val_subs))

    model.load_state_dict(torch.load("best_model_stage1_MFW_filtered.pt"))  # 加载预训练模型
    print("\nLoaded pretrained model...")

    fine_tuning_model_path = './best_model_stage2_MFW.pt'
    train_losses, val_losses, train_metrics, val_metrics = run_training_validation_loop(model,DataLoader(train_dataset,batch_size=batch_size,shuffle=True),
                                                                                        DataLoader(val_dataset,batch_size=batch_size),stage_name=f"阶段2(微调)",
                                                                                        save_path=fine_tuning_model_path)

    print(f"\n========== 使用验证集表现最好的模型 {fine_tuning_model_path} 在测试集上评估 ==========")
    model.load_state_dict(torch.load("best_model_stage2_MFW.pt"))
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    test_loss, test_metrics = evaluate(model, test_loader, device)
    print(f"阶段2 最终 Test Loss: {test_loss:.4f}")
    print_metrics(test_metrics, prefix="Test ")

    plot_metrics(train_losses, val_losses, train_metrics, val_metrics, k_lists)

if __name__ == '__main__':
    main()

