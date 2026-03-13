import os
import sys
import torch
import numpy as np
import logging
from torch_geometric.loader import DataLoader
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# --- 路径定位 ---
current_file_path = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src', 'utils'))
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))

from logging_config import setup_preset_logging

logger = setup_preset_logging(log_level=logging.INFO)

from models.session_graphmae.graphmae_model import SessionGraphMAE


def extract_graph_embeddings(loader, model, device):
    """通过冻结的 GraphMAE 提取全图 128 维表征"""
    embeddings, y_bin, y_multi = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            # 模型 forward 已经修改为返回三个参数: logits_bin, logits_multi, graph_emb
            _, _, g_emb = model(batch.x, batch.edge_index, batch.batch)
            embeddings.append(g_emb.cpu().numpy())
            y_bin.append(batch.y_bin.squeeze(-1).cpu().numpy())
            y_multi.append(batch.y_multi.cpu().numpy())

    return np.concatenate(embeddings, axis=0), np.concatenate(y_bin, axis=0), np.concatenate(y_multi, axis=0)


def evaluate_classifier(clf, name, X_train, y_train, X_test, y_test):
    """训练分类器并输出论文表格格式的指标"""
    logger.info(f"⏳ 正在训练 {name}...")
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    acc = accuracy_score(y_test, preds) * 100
    p = precision_score(y_test, preds, zero_division=0) * 100
    r = recall_score(y_test, preds, zero_division=0) * 100
    f1 = f1_score(y_test, preds, zero_division=0) * 100

    return f"{name:25} {acc:.2f} \t {p:.2f} \t {r:.2f} \t {f1:.2f}"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_dir = "/tmp/pycharm_project_982/processed_data/CIC-IDS-2017-session-srcIP_srcPort_dstIP_dstPort_proto"

    # 1. 加载冻结的 GraphMAE 模型
    ckpt_path = os.path.join(dataset_dir, "best_graphmae.ckpt")
    if not os.path.exists(ckpt_path):
        logger.error("❌ 找不到 best_graphmae.ckpt，请先运行流水线预训练模型！")
        return

    logger.info("📦 加载 Best GraphMAE 提取器...")
    model = SessionGraphMAE.load_from_checkpoint(ckpt_path)
    model.to(device)
    model.eval()

    # 2. 加载包含 768 维初始特征的 PyG 图数据集
    pt_path = os.path.join(dataset_dir, "embedded_graphs.pt")
    dataset = torch.load(pt_path, weights_only=False)

    train_dataset = [g for g in dataset if 'train' in getattr(g, 'split', 'train')]
    test_dataset = [g for g in dataset if 'test' in getattr(g, 'split', '')]

    # 兼容性处理：如果 split 没有严格分出 test，则切分
    if not test_dataset:
        logger.warning("数据集中没有标记为 test 的图，将自动按 8:2 切分作为测试评估...")
        split_idx = int(len(dataset) * 0.8)
        train_dataset = dataset[:split_idx]
        test_dataset = dataset[split_idx:]

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # 3. 提取 128 维图级别表征
    logger.info("🧠 正在使用 GraphMAE 提取训练集表征...")
    X_train, y_train_bin, _ = extract_graph_embeddings(train_loader, model, device)

    logger.info("🧠 正在使用 GraphMAE 提取测试集表征...")
    X_test, y_test_bin, _ = extract_graph_embeddings(test_loader, model, device)

    # 4. 初始化四个下游机器学习分类器 (对标你的论文描述)
    classifiers = {
        "MultiViewBert-GraphMAE+SVM": SVC(kernel='rbf', probability=True, random_state=42),
        "MultiViewBert-GraphMAE+RF": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "MultiViewBert-GraphMAE+MLP": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42),
        "MultiViewBert-GraphMAE+DT": DecisionTreeClassifier(random_state=42)
    }

    # 5. 训练并打印表 3.15 风格的输出
    logger.info("\n" + "=" * 70)
    logger.info("📊 表 3.15 二分类场景下各模型的性能指标结果 (下半部分)")
    logger.info("=" * 70)
    logger.info("模型                      A(%) \t P(%) \t R(%) \t F1(%)")
    logger.info("-" * 70)

    for name, clf in classifiers.items():
        result_str = evaluate_classifier(clf, name, X_train, y_train_bin, X_test, y_test_bin)
        logger.info(result_str)

    logger.info("=" * 70)


if __name__ == "__main__":
    main()