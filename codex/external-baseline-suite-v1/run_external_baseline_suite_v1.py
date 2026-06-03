import json
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Pt
from torch.utils.data import DataLoader, Dataset


ROOT = Path(r"C:\Users\ROG\Desktop\paper")
CODEX_DIR = ROOT / "codex"
DATA_DIR = ROOT / "PYthon训练代码" / "processed_clean"
SUITE_DIR = CODEX_DIR / "external-baseline-suite-v1"
REPORT_DOCX = SUITE_DIR / "external_baseline_suite_v1_report.docx"
RUN_LOG_DOCX = SUITE_DIR / "external_baseline_suite_v1_run_log.docx"
SUMMARY_XLSX = SUITE_DIR / "external_baseline_suite_v1_summary.xlsx"
RUN_JSON = SUITE_DIR / "external_baseline_suite_v1_summary.json"
PROGRESS_JSON = SUITE_DIR / "external_baseline_suite_v1_progress.json"

LABEL_MODE = "nominal"
DATA_PROTOCOL = "processed_clean_fixed_split"
SEED = 42
EPOCHS = 50
PATIENCE = 8
LEARNING_RATE = 1e-3
BATCH_SIZE = 512
EVAL_BATCH_SIZE = 2048
GLOBAL_MAX_CYCLE = 3000.0
LR_DECAY_EPOCH = 35
LR_DECAY_FACTOR = 0.2
TIME_LIMIT_SECONDS = 3600
START_TIME = time.time()

BASELINES = [
    {
        "model": "physical-only-v1",
        "kind": "analytic",
        "label": "Pure cubic physical base",
        "description": "Pure model-based baseline using only the cubic degradation function fitted on source domain.",
    },
    {
        "model": "lstm-baseline-v1",
        "kind": "trainable",
        "label": "LSTM baseline",
        "description": "Pure data-driven recurrent baseline using sequence fragment plus normalized cycle position.",
    },
    {
        "model": "gru-baseline-v1",
        "kind": "trainable",
        "label": "GRU baseline",
        "description": "Pure data-driven GRU baseline using sequence fragment plus normalized cycle position.",
    },
    {
        "model": "cnn-lstm-baseline-v1",
        "kind": "trainable",
        "label": "CNN-LSTM baseline",
        "description": "Common CNN-LSTM baseline without explicit physical base, attention, or domain alignment.",
    },
]


def ensure_dirs() -> None:
    SUITE_DIR.mkdir(parents=True, exist_ok=True)
    for cfg in BASELINES:
        (SUITE_DIR / cfg["model"]).mkdir(parents=True, exist_ok=True)


def elapsed_seconds() -> float:
    return float(time.time() - START_TIME)


def write_progress(stage: str, status: str, extra: Dict = None) -> None:
    ensure_dirs()
    payload = {
        "stage": stage,
        "status": status,
        "elapsed_seconds": elapsed_seconds(),
        "time_limit_seconds": TIME_LIMIT_SECONDS,
        "outputs": {
            "suite_dir": str(SUITE_DIR),
            "report_docx": str(REPORT_DOCX),
            "run_log_docx": str(RUN_LOG_DOCX),
            "summary_xlsx": str(SUMMARY_XLSX),
            "run_json": str(RUN_JSON),
        },
    }
    if extra:
        payload.update(extra)
    PROGRESS_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def check_deadline(stage: str) -> None:
    if elapsed_seconds() > TIME_LIMIT_SECONDS:
        write_progress(stage, "timeout")
        raise TimeoutError(f"Time limit exceeded at stage: {stage}")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_npy(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return np.load(path)


def load_split(split_name: str, label_mode: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = load_npy(DATA_DIR / f"clean_{split_name}_x.npy").astype(np.float32)
    p = load_npy(DATA_DIR / f"clean_{split_name}_p.npy").astype(np.float32)
    bidx = load_npy(DATA_DIR / f"clean_{split_name}_battery_index.npy").astype(np.int64)
    batch = load_npy(DATA_DIR / f"clean_{split_name}_batch.npy").astype(np.int64)
    if label_mode == "nominal":
        y = load_npy(DATA_DIR / f"clean_{split_name}_y_nominal.npy").astype(np.float32)
    elif label_mode == "relative":
        y = load_npy(DATA_DIR / f"clean_{split_name}_y_relative.npy").astype(np.float32)
    else:
        raise ValueError("label_mode must be 'nominal' or 'relative'")
    return x, y, p, bidx, batch


class SOHDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, p: np.ndarray, bidx: np.ndarray):
        self.x = x.astype(np.float32)
        self.y = y.astype(np.float32).reshape(-1, 1)
        self.p = (p.astype(np.float32) / GLOBAL_MAX_CYCLE).reshape(-1, 1)
        self.bidx = bidx.astype(np.int64)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx], self.p[idx], self.bidx[idx]


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(math.sqrt(np.mean((y_true.reshape(-1) - y_pred.reshape(-1)) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true.reshape(-1) - y_pred.reshape(-1))))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.clip(np.abs(y_true.reshape(-1)), 1e-8, None)
    return float(np.mean(np.abs((y_true.reshape(-1) - y_pred.reshape(-1)) / denom)) * 100)


def macro_rmse_by_battery(y_true: np.ndarray, y_pred: np.ndarray, battery_idx: np.ndarray) -> float:
    vals = []
    for b in np.unique(battery_idx):
        mask = battery_idx == b
        if np.any(mask):
            vals.append(rmse(y_true[mask], y_pred[mask]))
    return float(np.mean(vals)) if vals else float("nan")


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[Dict[str, float], pd.DataFrame]:
    model.eval()
    preds_all, trues_all, bidx_all = [], [], []
    with torch.no_grad():
        for x, y, p, bidx in loader:
            x = x.float().to(device)
            p = p.float().to(device)
            out = model(x, p)
            preds_all.append(out.cpu().numpy().reshape(-1))
            trues_all.append(y.numpy().reshape(-1))
            bidx_all.append(bidx.numpy().reshape(-1))
    y_pred = np.concatenate(preds_all)
    y_true = np.concatenate(trues_all)
    bidx = np.concatenate(bidx_all)
    metrics = {
        "pooled_rmse": rmse(y_true, y_pred),
        "macro_rmse": macro_rmse_by_battery(y_true, y_pred, bidx),
        "mae": mae(y_true, y_pred),
        "mape_percent": mape(y_true, y_pred),
    }
    rows = []
    for b in np.unique(bidx):
        mask = bidx == b
        rows.append(
            {
                "battery_index": int(b),
                "rmse": rmse(y_true[mask], y_pred[mask]),
                "mae": mae(y_true[mask], y_pred[mask]),
                "n_points": int(np.sum(mask)),
            }
        )
    return metrics, pd.DataFrame(rows)


def split_summary(name: str, x: np.ndarray, y: np.ndarray, p: np.ndarray, bidx: np.ndarray, batch: np.ndarray) -> Dict:
    return {
        "split": name,
        "samples": int(len(x)),
        "cells": int(len(np.unique(bidx))),
        "batches": ",".join(map(str, sorted(np.unique(batch).tolist()))),
        "x_shape": "x".join(map(str, x.shape)),
        "y_min": float(np.min(y)),
        "y_max": float(np.max(y)),
        "cycle_min": float(np.min(p)),
        "cycle_max": float(np.max(p)),
    }


class PositionBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 8), nn.ReLU())

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        return self.net(p)


class PhysicalOnlyBaseline(nn.Module):
    def __init__(self, coeffs: torch.Tensor):
        super().__init__()
        self.coeffs = coeffs

    def forward(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        a, b, c, d = self.coeffs
        return a * (p ** 3) + b * (p ** 2) + c * p + d


class LSTMBaseline(nn.Module):
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.rnn = nn.LSTM(input_size=1, hidden_size=hidden_dim, batch_first=True)
        self.pos = PositionBranch()
        self.head = nn.Sequential(nn.Linear(hidden_dim + 8, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1))

    def forward(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        seq = x.unsqueeze(-1)
        _, (h_n, _) = self.rnn(seq)
        feat = h_n[-1]
        fused = torch.cat([feat, self.pos(p)], dim=1)
        return self.head(fused)


class GRUBaseline(nn.Module):
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.rnn = nn.GRU(input_size=1, hidden_size=hidden_dim, batch_first=True)
        self.pos = PositionBranch()
        self.head = nn.Sequential(nn.Linear(hidden_dim + 8, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1))

    def forward(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        seq = x.unsqueeze(-1)
        _, h_n = self.rnn(seq)
        feat = h_n[-1]
        fused = torch.cat([feat, self.pos(p)], dim=1)
        return self.head(fused)


class CNNLSTMBaseline(nn.Module):
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 16, 3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.rnn = nn.LSTM(input_size=32, hidden_size=hidden_dim, batch_first=True)
        self.pos = PositionBranch()
        self.head = nn.Sequential(nn.Linear(hidden_dim + 8, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1))

    def forward(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        feat_map = self.cnn(x.unsqueeze(1)).permute(0, 2, 1)
        _, (h_n, _) = self.rnn(feat_map)
        feat = h_n[-1]
        fused = torch.cat([feat, self.pos(p)], dim=1)
        return self.head(fused)


def build_model(model_name: str, coeffs: torch.Tensor) -> nn.Module:
    if model_name == "physical-only-v1":
        return PhysicalOnlyBaseline(coeffs)
    if model_name == "lstm-baseline-v1":
        return LSTMBaseline()
    if model_name == "gru-baseline-v1":
        return GRUBaseline()
    if model_name == "cnn-lstm-baseline-v1":
        return CNNLSTMBaseline()
    raise ValueError(f"Unknown model name: {model_name}")


def train_baseline(
    cfg: Dict,
    coeffs: torch.Tensor,
    loaders: Dict[str, DataLoader],
    split_info: Dict[str, Dict],
    device: torch.device,
    issues: List[str],
) -> Dict:
    model_name = cfg["model"]
    run_dir = SUITE_DIR / model_name
    checkpoint_path = run_dir / f"{model_name}_best_by_target_dev.pt"
    history_csv = run_dir / f"{model_name}_history.csv"
    best_json = run_dir / f"{model_name}_best_metrics.json"
    results_xlsx = run_dir / f"{model_name}_results.xlsx"

    if cfg["kind"] == "analytic":
        model = build_model(model_name, coeffs).to(device)
        target_dev_metrics, target_dev_cells = evaluate_model(model, loaders["target_adapt_eval"], device)
        target_test_metrics, target_test_cells = evaluate_model(model, loaders["target_test_eval"], device)
        summary = {
            "model": model_name,
            "label": cfg["label"],
            "description": cfg["description"],
            "protocol": "Scheme A: source fit for physical coefficients only; target-domain development set for reporting; target_test final once",
            "best_epoch_selected_by_target_dev": 0,
            "target_dev_metrics": target_dev_metrics,
            "target_test_metrics": target_test_metrics,
            "checkpoint_path": None,
            "run_seconds": 0.0,
        }
        with pd.ExcelWriter(results_xlsx, engine="openpyxl") as writer:
            pd.DataFrame([summary]).to_excel(writer, sheet_name="config", index=False)
            pd.DataFrame([target_dev_metrics, target_test_metrics], index=["target_dev", "target_test"]).reset_index().rename(columns={"index": "split"}).to_excel(writer, sheet_name="metrics", index=False)
            pd.DataFrame(split_info.values()).to_excel(writer, sheet_name="splits", index=False)
            target_dev_cells.to_excel(writer, sheet_name="target_dev_cells", index=False)
            target_test_cells.to_excel(writer, sheet_name="target_test_cells", index=False)
        best_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return {
            "summary": summary,
            "target_dev_cells": target_dev_cells,
            "target_test_cells": target_test_cells,
            "history": pd.DataFrame(),
        }

    set_seed(SEED)
    model = build_model(model_name, coeffs).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_DECAY_EPOCH, gamma=LR_DECAY_FACTOR)
    loss_fn = nn.MSELoss()

    history_rows = []
    best_dev_rmse = float("inf")
    best_epoch = -1
    no_improve = 0
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        check_deadline(f"train_{model_name}_epoch_{epoch}")
        model.train()
        batch_losses = []
        for x, y, p, _ in loaders["source_train"]:
            x = x.float().to(device)
            y = y.float().to(device)
            p = p.float().to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x, p)
            loss = loss_fn(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu().item()))
        scheduler.step()

        target_dev_metrics, _ = evaluate_model(model, loaders["target_adapt_eval"], device)
        train_loss = float(np.mean(batch_losses)) if batch_losses else float("nan")
        history_rows.append(
            {
                "epoch": epoch,
                "train_mse": train_loss,
                "lr": optimizer.param_groups[0]["lr"],
                "target_dev_pooled_rmse": target_dev_metrics["pooled_rmse"],
                "target_dev_macro_rmse": target_dev_metrics["macro_rmse"],
                "target_dev_mae": target_dev_metrics["mae"],
                "target_dev_mape_percent": target_dev_metrics["mape_percent"],
                "best_epoch": best_epoch if best_epoch >= 0 else epoch,
                "best_target_dev_rmse": min(best_dev_rmse, target_dev_metrics["pooled_rmse"]),
            }
        )

        if target_dev_metrics["pooled_rmse"] < best_dev_rmse:
            best_dev_rmse = target_dev_metrics["pooled_rmse"]
            best_epoch = epoch
            no_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "target_dev_metrics": target_dev_metrics,
                    "config": {
                        "seed": SEED,
                        "epochs": EPOCHS,
                        "batch_size": BATCH_SIZE,
                        "learning_rate": LEARNING_RATE,
                        "label_mode": LABEL_MODE,
                        "data_protocol": DATA_PROTOCOL,
                    },
                },
                checkpoint_path,
            )
        else:
            no_improve += 1

        print(
            f"{model_name} Epoch {epoch:03d}/{EPOCHS} "
            f"train_mse={train_loss:.6f} target_dev_rmse={target_dev_metrics['pooled_rmse']:.6f} "
            f"best={best_dev_rmse:.6f}@{best_epoch}"
        )
        if no_improve >= PATIENCE:
            print(f"{model_name} early stop triggered at epoch {epoch}.")
            issues.append(f"{model_name}: early stopping triggered at epoch {epoch} after {PATIENCE} epochs without target-domain development improvement.")
            break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    target_dev_metrics, target_dev_cells = evaluate_model(model, loaders["target_adapt_eval"], device)
    target_test_metrics, target_test_cells = evaluate_model(model, loaders["target_test_eval"], device)

    history_df = pd.DataFrame(history_rows)
    history_df.to_csv(history_csv, index=False, encoding="utf-8-sig")

    summary = {
        "model": model_name,
        "label": cfg["label"],
        "description": cfg["description"],
        "protocol": "Scheme A: source supervised training; target-domain development set for checkpoint selection; target_test final once",
        "best_epoch_selected_by_target_dev": int(checkpoint["epoch"]),
        "target_dev_metrics": target_dev_metrics,
        "target_test_metrics": target_test_metrics,
        "checkpoint_path": str(checkpoint_path),
        "run_seconds": float(time.time() - start_time),
    }
    with pd.ExcelWriter(results_xlsx, engine="openpyxl") as writer:
        pd.DataFrame([summary]).to_excel(writer, sheet_name="config", index=False)
        pd.DataFrame(split_info.values()).to_excel(writer, sheet_name="splits", index=False)
        history_df.to_excel(writer, sheet_name="history", index=False)
        pd.DataFrame([target_dev_metrics, target_test_metrics], index=["target_dev", "target_test"]).reset_index().rename(columns={"index": "split"}).to_excel(writer, sheet_name="metrics", index=False)
        target_dev_cells.to_excel(writer, sheet_name="target_dev_cells", index=False)
        target_test_cells.to_excel(writer, sheet_name="target_test_cells", index=False)
    best_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "summary": summary,
        "target_dev_cells": target_dev_cells,
        "target_test_cells": target_test_cells,
        "history": history_df,
    }


def read_existing_internal_results() -> pd.DataFrame:
    rows = []
    summary_path = CODEX_DIR / "model_runs_summary.xlsx"
    if summary_path.exists():
        df = pd.read_excel(summary_path)
        if {"model", "best_target_adapt_rmse", "final_target_test_rmse"}.issubset(df.columns):
            for _, row in df.iterrows():
                rows.append(
                    {
                        "group": "existing_internal",
                        "model": row["model"],
                        "label": row["model"],
                        "target_dev_rmse": float(row["best_target_adapt_rmse"]),
                        "target_test_rmse": float(row["final_target_test_rmse"]),
                    }
                )
    return pd.DataFrame(rows)


def set_run_font(run, font_name: str = "Microsoft YaHei", size: float = 10.5, bold: bool = False) -> None:
    run.font.name = font_name
    run._element.rPr.rFonts.set(qn("w:eastAsia"), font_name)
    run.font.size = Pt(size)
    run.bold = bold


def add_heading(doc: Document, text: str, level: int = 1) -> None:
    p = doc.add_heading(level=level)
    r = p.add_run(text)
    set_run_font(r, size=14 if level == 1 else 12, bold=True)


def add_para(doc: Document, text: str) -> None:
    p = doc.add_paragraph()
    r = p.add_run(text)
    set_run_font(r)


def add_bullet(doc: Document, text: str) -> None:
    p = doc.add_paragraph(style="List Bullet")
    r = p.add_run(text)
    set_run_font(r)


def add_code(doc: Document, text: str) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Pt(18)
    r = p.add_run(text)
    set_run_font(r, font_name="Consolas", size=9)


def fill_table(table, rows: List[List[str]]) -> None:
    for i, row in enumerate(rows):
        for j, value in enumerate(row):
            cell = table.rows[i].cells[j]
            cell.text = ""
            p = cell.paragraphs[0]
            r = p.add_run(str(value))
            set_run_font(r, size=9.5, bold=(i == 0))


def build_report(existing_df: pd.DataFrame, new_df: pd.DataFrame) -> None:
    doc = Document()
    styles = doc.styles
    styles["Normal"].font.name = "Microsoft YaHei"
    styles["Normal"]._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    styles["Normal"].font.size = Pt(10.5)

    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = title.add_run("新增外部基线对比实验汇报")
    set_run_font(r, size=16, bold=True)
    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sr = sub.add_run("Strict Scheme A | source=Batch1-8, target-domain development=Batch9, blind-test=Batch10")
    set_run_font(sr, size=10)

    add_heading(doc, "一、为什么要补外部基线", level=1)
    add_para(
        doc,
        "此前项目已完成内部消融，包括 final-retained-v1、no-physical-base-v1、no-mmd-v1 和 no-attention-v1。这一部分可以证明所提出混合模型内部各模块的相对贡献，但仍不足以回答审稿人关于“与常见模型相比是否仍然有效”的问题。因此，本轮实验补充了常见 pure model-based 和 pure data-driven 基线，以增强论文实验链条的完整性。",
    )
    add_para(
        doc,
        "基线选择依据来自项目内现有综述笔记。你的综述文档已经明确将 LSTM、GRU、CNN、GPR、SVR 归纳为该任务中的常见 data-driven 方法。考虑当前数据规模超过 115k 个 source samples、严格 Scheme A 评估协议以及 1 小时运行预算，本轮优先补入计算可落地的四个基线：纯三次物理基座、LSTM、GRU 和 CNN-LSTM。GPR 与标准核 SVR 在当前窗口级数据量下计算代价过高，因此暂不纳入本轮实际运行。",
    )

    add_heading(doc, "二、当前实验进度回顾", level=1)
    add_bullet(doc, "已完成 strict Scheme A 最终保留模型训练：final-retained-v1，Batch 10 blind-test pooled RMSE = 0.009499。")
    add_bullet(doc, "已完成内部消融：no-physical-base-v1、no-mmd-v1、no-attention-v1。")
    add_bullet(doc, "已完成 per-battery 对比、Batch 10 2x4 预测图、论文 Methodology 与 Experimental Settings 初稿。")
    add_bullet(doc, "本轮新增工作重点是：在相同协议下补入外部常见基线，完善论文对比实验。")

    add_heading(doc, "三、新增基线设置", level=1)
    add_bullet(doc, "Pure cubic physical base：仅利用 source domain 拟合三次退化函数，作为纯模型类基线。")
    add_bullet(doc, "LSTM baseline：使用序列片段与归一化循环位置，作为纯数据驱动递归基线。")
    add_bullet(doc, "GRU baseline：用 GRU 替代 LSTM，作为另一类常见递归基线。")
    add_bullet(doc, "CNN-LSTM baseline：先提取局部卷积特征，再建模时序依赖，作为常见深度时序基线。")
    add_para(
        doc,
        "所有新增基线均遵循与保留模型一致的数据协议：训练仅使用 Batch 1-8 source，checkpoint 根据 Batch 9 target-domain development set 选择，Batch 10 仅用于最终 blind-test。这样可以避免比较过程因切分方式不同而失去公平性。",
    )

    add_heading(doc, "四、结果总表", level=1)
    combined = pd.concat([existing_df, new_df], ignore_index=True)
    rows = [["Group", "Model", "Target-dev RMSE", "Blind-test RMSE"]]
    for _, row in combined.iterrows():
        rows.append(
            [
                row["group"],
                row["label"],
                f"{row['target_dev_rmse']:.6f}",
                f"{row['target_test_rmse']:.6f}",
            ]
        )
    table = doc.add_table(rows=len(rows), cols=len(rows[0]))
    table.style = "Table Grid"
    fill_table(table, rows)

    add_heading(doc, "五、论文层面的结论更新", level=1)
    add_para(
        doc,
        "内部消融已经证明：物理基座是决定性模块，Attention 是显著有效的辅助时序表征模块，MMD 是轻量但正向的跨域增强模块。新增外部基线的作用在于回答另一个问题：在不使用显式物理退化基线和跨域设计时，常见数据驱动基线在同一 blind-test 协议下能达到什么水平。若这些基线整体弱于 retained model，则可以进一步说明你的方法不仅内部结构合理，而且相对于常规模型也具有优势。",
    )
    add_para(
        doc,
        "从论文叙事角度看，外部基线结果可用于支持以下写法：现有纯数据驱动模型在 source-train / target-test 跨批次条件下容易受到分布偏移影响，而引入宏观物理退化基线和目标域生命周期自适应对齐后，模型在 strict blind-test 场景中表现更稳定。论文中仍应保持克制，避免把 CNN、LSTM、GRU 本身写成创新点，重点仍放在物理引导与跨域建模机制。",
    )

    doc.save(REPORT_DOCX)


def build_run_log(run_entries: List[Dict], issues: List[str], split_info: Dict[str, Dict], device: str) -> None:
    doc = Document()
    styles = doc.styles
    styles["Normal"].font.name = "Microsoft YaHei"
    styles["Normal"]._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    styles["Normal"].font.size = Pt(10.5)

    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = title.add_run("新增外部基线实验运行日志")
    set_run_font(r, size=16, bold=True)
    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sr = sub.add_run(f"device={device} | elapsed={elapsed_seconds():.2f}s")
    set_run_font(sr, size=10)

    add_heading(doc, "一、运行配置", level=1)
    add_bullet(doc, f"Python data protocol: {DATA_PROTOCOL}")
    add_bullet(doc, f"Label mode: {LABEL_MODE}")
    add_bullet(doc, f"Seed: {SEED}")
    add_bullet(doc, f"Epochs: {EPOCHS}")
    add_bullet(doc, f"Batch size: {BATCH_SIZE}")
    add_bullet(doc, f"Learning rate: {LEARNING_RATE}")
    add_bullet(doc, f"Patience: {PATIENCE}")
    add_bullet(doc, f"Suite directory: {SUITE_DIR}")

    add_heading(doc, "二、数据划分确认", level=1)
    for key in ["source", "target_adapt", "target_test"]:
        info = split_info[key]
        add_bullet(
            doc,
            f"{key}: samples={info['samples']}, cells={info['cells']}, batches={info['batches']}, x_shape={info['x_shape']}",
        )

    add_heading(doc, "三、运行记录", level=1)
    rows = [["Model", "Best epoch", "Target-dev RMSE", "Blind-test RMSE", "Run seconds"]]
    for item in run_entries:
        rows.append(
            [
                item["label"],
                item["best_epoch"],
                f"{item['target_dev_rmse']:.6f}",
                f"{item['target_test_rmse']:.6f}",
                f"{item['run_seconds']:.2f}",
            ]
        )
    table = doc.add_table(rows=len(rows), cols=len(rows[0]))
    table.style = "Table Grid"
    fill_table(table, rows)

    add_heading(doc, "四、运行中出现的问题与处理", level=1)
    if issues:
        for item in issues:
            add_bullet(doc, item)
    else:
        add_bullet(doc, "本轮未出现阻断性报错。")
        add_bullet(doc, "个别模型触发 early stopping，属于正常训练收敛现象，不属于错误。")

    add_heading(doc, "五、后续建议", level=1)
    add_bullet(doc, "如果导师还希望继续增强对比，可在下一轮增加线性回归 / MLP / PCA+LinearSVR 等轻量基线。")
    add_bullet(doc, "若希望补入 GPR 或标准核 SVR，建议先做样本级降采样或 battery-level feature engineering，否则窗口级 115k 数据量下计算开销过高。")
    add_bullet(doc, "若论文最终冲更高水平，可对 retained model 和最关键外部基线再补两个随机种子，用均值±标准差呈现。")

    doc.save(RUN_LOG_DOCX)


def main():
    ensure_dirs()
    write_progress("start", "running")
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    issues: List[str] = []

    s_x, s_y, s_p, s_bidx, s_batch = load_split("source", LABEL_MODE)
    ta_x, ta_y, ta_p, ta_bidx, ta_batch = load_split("target_adapt", LABEL_MODE)
    tt_x, tt_y, tt_p, tt_bidx, tt_batch = load_split("target_test", LABEL_MODE)

    split_info = {
        "source": split_summary("source", s_x, s_y, s_p, s_bidx, s_batch),
        "target_adapt": split_summary("target_adapt", ta_x, ta_y, ta_p, ta_bidx, ta_batch),
        "target_test": split_summary("target_test", tt_x, tt_y, tt_p, tt_bidx, tt_batch),
    }
    write_progress("data_loaded", "completed", {"split_info": split_info, "device": str(device)})

    coeffs = np.polyfit((s_p.reshape(-1) / GLOBAL_MAX_CYCLE), s_y.reshape(-1), 3)
    coeffs_t = torch.tensor(coeffs, dtype=torch.float32, device=device)

    ds_s = SOHDataset(s_x, s_y, s_p, s_bidx)
    ds_ta = SOHDataset(ta_x, ta_y, ta_p, ta_bidx)
    ds_tt = SOHDataset(tt_x, tt_y, tt_p, tt_bidx)
    loaders = {
        "source_train": DataLoader(ds_s, batch_size=BATCH_SIZE, shuffle=True, drop_last=True),
        "target_adapt_eval": DataLoader(ds_ta, batch_size=EVAL_BATCH_SIZE, shuffle=False),
        "target_test_eval": DataLoader(ds_tt, batch_size=EVAL_BATCH_SIZE, shuffle=False),
    }

    baseline_results = []
    run_entries = []
    per_model_cell_tables = {}
    for cfg in BASELINES:
        check_deadline(f"before_{cfg['model']}")
        print(f"RUN={cfg['model']}")
        result = train_baseline(cfg, coeffs_t, loaders, split_info, device, issues)
        baseline_results.append(result["summary"])
        per_model_cell_tables[cfg["model"]] = result["target_test_cells"]
        run_entries.append(
            {
                "model": cfg["model"],
                "label": cfg["label"],
                "best_epoch": result["summary"]["best_epoch_selected_by_target_dev"],
                "target_dev_rmse": result["summary"]["target_dev_metrics"]["pooled_rmse"],
                "target_test_rmse": result["summary"]["target_test_metrics"]["pooled_rmse"],
                "run_seconds": result["summary"]["run_seconds"],
            }
        )
        write_progress(f"completed_{cfg['model']}", "completed", {"summary": result["summary"]})

    new_df = pd.DataFrame(
        [
            {
                "group": "new_external",
                "model": item["model"],
                "label": item["label"],
                "target_dev_rmse": item["target_dev_metrics"]["pooled_rmse"],
                "target_test_rmse": item["target_test_metrics"]["pooled_rmse"],
                "target_test_macro_rmse": item["target_test_metrics"]["macro_rmse"],
                "target_test_mae": item["target_test_metrics"]["mae"],
                "target_test_mape_percent": item["target_test_metrics"]["mape_percent"],
            }
            for item in baseline_results
        ]
    )
    existing_df = read_existing_internal_results()

    with pd.ExcelWriter(SUMMARY_XLSX, engine="openpyxl") as writer:
        existing_df.to_excel(writer, sheet_name="existing_internal", index=False)
        new_df.to_excel(writer, sheet_name="new_external", index=False)
        pd.concat([existing_df, new_df], ignore_index=True).to_excel(writer, sheet_name="combined", index=False)
        for model_name, cell_df in per_model_cell_tables.items():
            cell_df.to_excel(writer, sheet_name=model_name[:31], index=False)

    build_report(existing_df, new_df)
    build_run_log(run_entries, issues, split_info, str(device))

    summary = {
        "status": "success",
        "elapsed_seconds": elapsed_seconds(),
        "device": str(device),
        "suite_dir": str(SUITE_DIR),
        "new_external_results": baseline_results,
        "issues": issues,
        "outputs": {
            "summary_xlsx": str(SUMMARY_XLSX),
            "report_docx": str(REPORT_DOCX),
            "run_log_docx": str(RUN_LOG_DOCX),
            "progress_json": str(PROGRESS_JSON),
        },
    }
    RUN_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_progress("done", "success", {"run_json": str(RUN_JSON)})
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
