from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stroke_bci_mvp.config import load_config
from stroke_bci_mvp.datasets import load_dataset
from stroke_bci_mvp.evaluation import make_train_test_split
from stroke_bci_mvp.models import build_model
from stroke_bci_mvp.signal import filter_valid_epochs, notch_epochs


def train(config_path: str) -> dict:
    """
    训练BCI基线解码器模型并评估性能。

    该函数加载配置文件和数据集，执行信号预处理（陷波滤波和质量过滤），
    划分训练测试集，训练模型，计算评估指标，并保存模型和结果。

    Args:
        config_path (str): 配置文件的路径，包含数据集、预处理、模型和输出等配置信息。

    Returns:
        dict: 包含以下键的评估指标字典：
            - dataset (str): 数据集名称
            - config_path (str): 配置文件路径
            - n_epochs_total (int): 总epoch数量
            - n_epochs_valid (int): 有效epoch数量
            - rejected_epochs (int): 被拒绝的epoch数量
            - quality_reject_rate (float): 质量拒绝率
            - train_epochs (int): 训练集epoch数量
            - test_epochs (int): 测试集epoch数量
            - balanced_accuracy (float): 平衡准确率
            - auc (float): ROC曲线下面积
            - f1 (float): F1分数
            - confusion_matrix (list): 混淆矩阵（嵌套列表形式）
            - quality_score_mean (float): 平均质量分数
    """
    config = load_config(config_path)
    outputs = config["outputs"]
    output_dir = Path(outputs["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载数据集并执行信号预处理：陷波滤波去除工频干扰，质量过滤去除低质量epoch
    dataset = load_dataset(config)
    X = notch_epochs(dataset.X, dataset.sfreq, config["preprocessing"].get("notch_hz"))
    X_valid, y_valid, quality_results = filter_valid_epochs(
        X,
        dataset.y,
        dataset.sfreq,
        dataset.ch_names,
        config["quality"],
    )
    valid_subject_ids = dataset.subject_ids[np.asarray([result.valid for result in quality_results], dtype=bool)]

    # 按配置划分训练集和测试集；真实数据默认按 subject 划分，避免同一受试者泄漏到测试集。
    split = make_train_test_split(y_valid, valid_subject_ids, config)
    X_train, X_test = X_valid[split.train_idx], X_valid[split.test_idx]
    y_train, y_test = y_valid[split.train_idx], y_valid[split.test_idx]

    # 构建模型并训练
    model = build_model(config, dataset.sfreq)
    model.fit(X_train, y_train)

    # 在测试集上进行预测并计算多维度评估指标
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]
    metrics = {
        "dataset": config["dataset"]["name"],
        "config_path": str(Path(config_path)),
        "n_epochs_total": int(len(dataset.y)),
        "n_epochs_valid": int(len(y_valid)),
        "rejected_epochs": int(len(dataset.y) - len(y_valid)),
        "quality_reject_rate": float(1.0 - len(y_valid) / max(1, len(dataset.y))),
        "train_epochs": int(len(y_train)),
        "test_epochs": int(len(y_test)),
        "split_strategy": split.strategy,
        "test_subject_ids": split.test_subject_ids,
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "auc": float(roc_auc_score(y_test, y_score)),
        "f1": float(f1_score(y_test, y_pred)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "quality_score_mean": float(np.mean([result.score for result in quality_results])),
    }

    # 打包模型及相关元数据并保存到磁盘
    model_bundle = {
        "model": model,
        "sfreq": dataset.sfreq,
        "ch_names": dataset.ch_names,
        "label_names": dataset.label_names,
        "config": config,
        "split": {
            "strategy": split.strategy,
            "train_idx": split.train_idx.tolist(),
            "test_idx": split.test_idx.tolist(),
            "test_subject_ids": split.test_subject_ids,
        },
    }
    joblib.dump(model_bundle, outputs["model_path"])
    Path(outputs["offline_metrics_path"]).write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return metrics


def main() -> None:
    """
    主函数：解析命令行参数并启动训练流程。

    通过命令行参数指定配置文件路径（默认为configs/default.yaml），
    调用train函数进行模型训练，并将评估指标以JSON格式打印到标准输出。
    """
    parser = argparse.ArgumentParser(description="Train the BCI MVP baseline decoder.")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    metrics = train(args.config)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
