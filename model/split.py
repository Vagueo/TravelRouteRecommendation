from pathlib import Path
import json
from config import mfw_path, mfw_virtual_path, mfw_virtual_path_filtered
from sklearn.model_selection import train_test_split
from collections import defaultdict
from tqdm import tqdm
from generate_virtual_routes import generate
import os

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
        print(f"\n✓ Loaded {count} routes from {Path(path).name}")
    return routes

# 创建 ID 映射字典
mdd_counter = 1
poi_counter = 1
mdd2id = {}
id2mdd = {}
poi2id = {}
id2poi = {}

# 给路线中的mdd和poi的id重新编码（因为路线中的mdd和poi的id并非连续的）
def encode_trajectory(trajectory, source):
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
            encoded.append(('mdd', mdd2id[mdd], details, source))
        elif step['type'] == 'poi':
            poi = step['poi_title']
            if poi not in poi2id:
                poi2id[poi] = poi_counter
                id2poi[poi_counter] = poi
                poi_counter += 1
            encoded.append(('poi', poi2id[poi], details, source))
    return encoded

def generate_sub_trajectories_fixed_window(trajectory, input_len=2, stride=1, max_subs=None):
    """
    生成固定输入长度的子轨迹对，目标是下一个地点。
    输入长度固定为 input_len，滑动步长为 stride
    """
    sub_trajectories = []
    for i in range(0, len(trajectory) - input_len, stride):
        input_seq = trajectory[i:i + input_len]
        target_point = trajectory[i + input_len]
        sub_trajectories.append((input_seq, target_point))
        if max_subs and len(sub_trajectories) >= max_subs:
            break
    return sub_trajectories

def save_sub_trajectories(subs, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        for sub in subs:
            json.dump(sub, f, ensure_ascii=False)
            f.write('\n')
    print(f"successfully saved the file {filepath}")

def extract_subsequences(sequence, min_len=2, max_len=3):
    subseqs = set()
    for l in range(min_len, max_len + 1):
        for i in range(len(sequence) - l + 1):
            subseqs.add(tuple(sequence[i:i+l]))
    return subseqs

def get_sequence_overlap_ratio(train_data, test_data, min_len=2, max_len=3):
    train_subseqs = set()
    for traj in train_data:
        train_subseqs.update(extract_subsequences(traj, min_len, max_len))

    test_subseqs = defaultdict(int)
    total = 0
    for traj in test_data:
        subseqs = extract_subsequences(traj, min_len, max_len)
        for s in subseqs:
            test_subseqs[s] += 1
            total += 1

    overlap_count = sum(1 for s in test_subseqs if s in train_subseqs)
    overlap_ratio = overlap_count / len(test_subseqs) if test_subseqs else 0
    return overlap_ratio

# 划分数据集
def split_all_datasets():
    global mfw_encoded, mfw_virtual_encoded
    # Stage 1: 加载MFW真实数据
    print(f"Loading MFW routes from {mfw_path}")
    mfw_routes = load_routes([mfw_path])
    train_val_trajs, test_trajs = train_test_split(mfw_routes, test_size=0.1, random_state=42)

    mfw_train_val_encoded = []
    for route in train_val_trajs:
        if 'trajectory' in route:
            encoded = encode_trajectory(route['trajectory'], source="mfw")
            if len(encoded) >= 3:
                mfw_train_val_encoded.append(encoded)

    mfw_test_encoded = []
    for route in test_trajs:
        if 'trajectory' in route:
            encoded = encode_trajectory(route['trajectory'], source="mfw")
            if len(encoded) >= 3:
                mfw_test_encoded.append(encoded)

    mfw_encoded = mfw_train_val_encoded + mfw_test_encoded

    all_subs = [sub for traj in mfw_train_val_encoded for sub in generate_sub_trajectories_fixed_window(traj)]
    mfw_test_subs = [sub for traj in mfw_test_encoded for sub in generate_sub_trajectories_fixed_window(traj)]

    ## 保存真实子轨迹
    if not os.path.exists("D:/Project/pythonProject/mfwscrapy/datasets/finetuning/mfw_real_train_val_routes.jsonl") \
        and not os.path.exists("D:/Project/pythonProject/mfwscrapy/datasets/finetuning/mfw_real_test_routes.jsonl") \
        and not os.path.exists("D:/Project/pythonProject/mfwscrapy/datasets/finetuning/mfw_real_train_val_subs.jsonl")\
        and not os.path.exists("D:/Project/pythonProject/mfwscrapy/datasets/finetuning/mfw_real_test_subs.jsonl"):
        # 编码前完整路径
        save_sub_trajectories(train_val_trajs,"D:/Project/pythonProject/mfwscrapy/datasets/finetuning/mfw_real_train_val_routes.jsonl")
        save_sub_trajectories(test_trajs,"D:/Project/pythonProject/mfwscrapy/datasets/finetuning/mfw_real_test_routes.jsonl")
        # 编码后子序列
        save_sub_trajectories(all_subs, "D:/Project/pythonProject/mfwscrapy/datasets/finetuning/mfw_real_train_val_subs.jsonl")
        save_sub_trajectories(mfw_test_subs, "D:/Project/pythonProject/mfwscrapy/datasets/finetuning/mfw_real_test_subs.jsonl")

    # Stage 2: 加载MFW虚拟生成的路线
    if not os.path.exists("D:/Project/pythonProject/mfwscrapy/datasets/pretraining/mfw_virtual_train_subs_filtered.jsonl") \
        and not os.path.exists("D:/Project/pythonProject/mfwscrapy/datasets/pretraining/mfw_virtual_val_subs_filtered.jsonl") \
        and not os.path.exists("D:/Project/pythonProject/mfwscrapy/datasets/pretraining/mfw_virtual_train_subs.jsonl") \
        and not os.path.exists("D:/Project/pythonProject/mfwscrapy/datasets/pretraining/mfw_virtual_val_subs.jsonl"):
        generate()
    else:
        print(f"These files already exists. Skipping generation.")
    print(f"Loading MFW virtual routes from {mfw_virtual_path}")
    mfw_virtual_routes_filtered = load_routes([mfw_virtual_path_filtered])
    # MFW虚拟生成的路线的编码（全部用作预训练训练集）,处理了数据泄露的
    mfw_virtual_encoded_filtered = []
    for route in mfw_virtual_routes_filtered:
        if 'trajectory' in route:
            encoded = encode_trajectory(route['trajectory'], source="mfw_virtual")
            if len(encoded) >= 3:
                mfw_virtual_encoded_filtered.append(encoded)
    mfw_virtual_train_trajs_filtered, mfw_virtual_val_trajs_filtered = train_test_split(mfw_virtual_encoded_filtered, test_size=0.2, random_state=42)

    mfw_virtual_train_subs_filtered = [sub for traj in mfw_virtual_train_trajs_filtered for sub in generate_sub_trajectories_fixed_window(traj)]
    mfw_virtual_val_subs_filtered = [sub for traj in mfw_virtual_val_trajs_filtered for sub in generate_sub_trajectories_fixed_window(traj)]
    # mfw_virtual_test_subs = [sub for traj in mfw_virtual_test_trajs for sub in generate_sub_trajectories_fixed_window(traj, input_len=3)]

    if not os.path.exists("D:/Project/pythonProject/mfwscrapy/datasets/pretraining/mfw_virtual_train_subs_filtered.jsonl") and \
        not os.path.exists("D:/Project/pythonProject/mfwscrapy/datasets/pretraining/mfw_virtual_val_subs_filtered.jsonl"):
        # 保存子轨迹
        save_sub_trajectories(mfw_virtual_train_subs_filtered, "D:/Project/pythonProject/mfwscrapy/datasets/pretraining/mfw_virtual_train_subs_filtered.jsonl")
        save_sub_trajectories(mfw_virtual_val_subs_filtered, "D:/Project/pythonProject/mfwscrapy/datasets/pretraining/mfw_virtual_val_subs_filtered.jsonl")
    # MFW虚拟生成的路线的编码（全部用作预训练训练集）,未处理数据泄露的
    print(f"Loading MFW virtual routes from {mfw_virtual_path}")
    mfw_virtual_routes = load_routes([mfw_virtual_path])
    mfw_virtual_encoded = []
    for route in mfw_virtual_routes:
        if 'trajectory' in route:
            encoded = encode_trajectory(route['trajectory'], source="mfw_virtual")
            if len(encoded) >= 3:
                mfw_virtual_encoded.append(encoded)
    mfw_virtual_train_trajs, mfw_virtual_val_trajs = train_test_split(mfw_virtual_encoded, test_size=0.2, random_state=42)

    mfw_virtual_train_subs = [sub for traj in mfw_virtual_train_trajs for sub in generate_sub_trajectories_fixed_window(traj)]
    mfw_virtual_val_subs = [sub for traj in mfw_virtual_val_trajs for sub in generate_sub_trajectories_fixed_window(traj)]

    if not os.path.exists("D:/Project/pythonProject/mfwscrapy/datasets/pretraining/mfw_virtual_train_subs.jsonl") and \
        not os.path.exists("D:/Project/pythonProject/mfwscrapy/datasets/pretraining/mfw_virtual_val_subs.jsonl"):
        # 保存子轨迹
        save_sub_trajectories(mfw_virtual_train_subs, "D:/Project/pythonProject/mfwscrapy/datasets/pretraining/mfw_virtual_train_subs.jsonl")
        save_sub_trajectories(mfw_virtual_val_subs, "D:/Project/pythonProject/mfwscrapy/datasets/pretraining/mfw_virtual_val_subs.jsonl")
    return mfw_encoded, mfw_virtual_encoded


