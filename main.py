import torch
import random
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
from model import DualLayerRecModel
from train import (
    load_routes,
    encode_trajectory,
    POITrajectoryDataset,
    train_one_epoch,
    evaluate,
    print_metrics
)
from train import mdd2id, poi2id
from transformers import BertTokenizer,AutoTokenizer

def main():
    # 自动收集文件路径
    mfw_path = Path('./datasets/precleaning/MFW/route_step4.jsonl')
    fs_dir = Path('./datasets/precleaning/FourSquare')
    # foursquare_paths = sorted(fs_dir.glob('trajectories_batch*.jsonl'))
    foursquare_paths = [
        fs_dir / f'trajectories_batch{i}.jsonl' for i in range(1)
    ]

    print(f"Loading MFW routes from {mfw_path}")
    print(f"Loading FourSquare routes from {len(foursquare_paths)} files")

    # 加载数据
    mfw_routes = load_routes([mfw_path])
    fs_routes = load_routes(foursquare_paths)
    all_routes = mfw_routes + fs_routes

    # 编码轨迹
    encoded_routes = [encode_trajectory(route['trajectory']) for route in all_routes if 'trajectory' in route]

    # 获取所有的details
    # all_details = []
    # for traj in encoded_routes:
    #     for step in traj:
    #         step_type, step_id, detail = step
    #         all_details.append(detail)

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
        batch_size=3, shuffle=True)

    val_loader = DataLoader(
        POITrajectoryDataset(val_data, tokenizer=tokenizer),
        batch_size=3)

    test_loader = DataLoader(
        POITrajectoryDataset(test_data, tokenizer=tokenizer),
        batch_size=3)

    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DualLayerRecModel(
        mdd_vocab_size=len(mdd2id),
        poi_vocab_size=len(poi2id),
        embedding_dim=128,
        num_heads=4,
        num_layers=2,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val_loss = float('inf')
    train_losses, val_losses = [], []

    # 训练模型
    for epoch in range(20):
        print(f"\nEpoch {epoch+1}")

        train_loss, train_metrics = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_metrics = evaluate(model, val_loader, device)

        print(f"Train Loss: {train_loss:.4f}")
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

if __name__ == '__main__':
    main()
