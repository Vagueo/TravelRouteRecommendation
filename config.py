from transformers import AutoModel,AutoTokenizer
from pathlib import Path
import torch

# 设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载数据集的文件路径
mfw_path = Path('D:/Project/pythonProject/mfwscrapy/datasets/precleaning/MFW/route_step4.jsonl')
mfw_virtual_path = Path('D:/Project/pythonProject/mfwscrapy/datasets/precleaning/MFW/virtual_routes.jsonl')
mfw_virtual_path_filtered = Path('D:/Project/pythonProject/mfwscrapy/datasets/precleaning/MFW/virtual_routes_filtered.jsonl')
fs_dir = Path('D:/Project/pythonProject/mfwscrapy/datasets/precleaning/FourSquare')
foursquare_paths = [
    fs_dir / f'trajectories_batch{i}.jsonl' for i in range(1)
]

# 预训练模型的参数部分
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")
bert = AutoModel.from_pretrained('distilbert-base-multilingual-cased')
bert_out_dim = bert.config.hidden_size

max_len = 10    # 最大子轨迹地点数
# max_subs_per_traj = 9   # 最大子轨迹数

# 模型超参数
batch_size = 512    # 一个批次加载的数据量
embedding_dim = 128 # 模型生成的向量长度
num_heads = 4   # 注意力机制头数
num_layers = 2  # 层数
epochs = 30     # 训练轮次
lr = 1e-3   # 学习率
weight_decay=1e-5   # 正则化系数
drop_prob = 0.1
# 评估的top-k中的k的列表值
k_lists = [5, 10, 15, 20]
