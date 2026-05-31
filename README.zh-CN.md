# 中风上肢康复 BCI MVP

这个 MVP 演示了一套面向产品落地的中风上肢康复 BCI 算法管线：

```text
EEG epoch 数据
  -> 信号质量门控
  -> 实时可用的预处理 / 特征提取
  -> 运动意图解码
  -> 触发决策
  -> 模拟 BU100 腕部 / 手部辅助动作
  -> 训练 session 报告
```

默认 demo 使用合成的 EEG-like 数据，因此不下载公开数据集也可以跑通完整链路。后续可以在 `src/stroke_bci_mvp/datasets/` 下继续添加公开数据集或真实设备数据的 loader。

## 快速开始

```powershell
pip install -r requirements.txt
python scripts/train_baseline.py --config configs/default.yaml
python scripts/simulate_online.py --config configs/default.yaml
```

如果要使用公开的 PhysioNet EEG Motor Movement/Imagery 数据集：

```powershell
python scripts/train_baseline.py --config configs/physionet.yaml
python scripts/simulate_online.py --config configs/physionet.yaml
```

如果要接入 Figshare 中风患者 MI 数据集，先下载元数据或 EDF 压缩包：

```powershell
python scripts/download_figshare_stroke.py --metadata-only
python scripts/download_figshare_stroke.py --include-edf-zip
```

解压 `edffile.zip` 后，确认 `configs/figshare_stroke.yaml` 里的 `dataset.path` 指向 EDF 文件所在目录，再运行：

```powershell
python scripts/train_baseline.py --config configs/figshare_stroke.yaml
python scripts/simulate_online.py --config configs/figshare_stroke.yaml
```

如果要做更接近真实新患者泛化的评估，可以运行 leave-one-subject-out：

```powershell
python scripts/evaluate_subject_generalization.py --config configs/figshare_stroke.yaml
```

运行结果会按数据集写入各自目录。默认合成数据会写入 `outputs/synthetic/`，PhysioNet 会写入 `outputs/physionet/`，Figshare 中风数据会写入 `outputs/figshare_stroke/`：

- `model.joblib`：训练好的 baseline 解码模型
- `offline_metrics.json`：balanced accuracy、AUC、F1、混淆矩阵等离线指标
- `session_report.json`：伪在线触发指标
- `session_report.md`：面向治疗师/临床使用场景的训练摘要草稿

训练脚本默认使用 `online_windows` 模式：训练样本会按在线控制相同的窗口长度、步长和任务时间窗生成，避免“完整 epoch 训练、1 秒窗口推理”的分布不一致。真实公开数据配置默认使用 subject-aware train/test split，避免同一受试者的数据同时进入训练集和测试集。默认合成数据仍使用随机分层划分，方便快速 smoke test。

运行最小测试：

```powershell
pytest
```

## MVP 范围

第一版实现只聚焦一个非常窄的闭环动作：

```text
静息 vs 患侧手运动意图 -> 模拟触发开手 / 伸腕辅助
```

这版实现刻意保持保守：

- 使用 FBCSP + LDA baseline，而不是一开始上深度模型
- 在解码前加入信号质量门控
- 使用连续窗口确认触发，降低瞬时误判
- 每次触发后设置 refractory period，避免连续误触发
- 明确记录拒识和未触发原因，方便后续复盘

## 当前定位

这个项目不是完整医疗器械软件，而是一个算法 MVP：

```text
公开/合成 EEG 数据
  -> 运动意图识别
  -> 信号质量拒识
  -> 伪在线触发控制
  -> 康复训练报告
```

后续接入真实产品时，可以逐步替换两部分：

- 将 synthetic / PhysioNet 数据入口替换为 BCI100 实时 EEG 数据流
- 将模拟触发事件替换为 BU100 机器人控制接口

## 常用脚本

训练 baseline 模型：

```powershell
python scripts/train_baseline.py --config configs/default.yaml
```

运行伪在线 session 模拟：

```powershell
python scripts/simulate_online.py --config configs/default.yaml
```

扫描触发阈值，观察有效触发率和误触发率的权衡：

```powershell
python scripts/tune_trigger_threshold.py --config configs/default.yaml --thresholds 0.8 0.85 0.9
```

下载 PhysioNet EEGMMI 数据，用于后续公开数据集实验：

```powershell
python scripts/download_physionet.py --subjects 1 2 3 --runs 4 8 12
```
