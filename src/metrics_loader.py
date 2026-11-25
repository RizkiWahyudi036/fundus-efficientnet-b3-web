import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.config import DATASETS, LABELS, SCENARIO_ORDER, SCENARIOS


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_data(show_spinner=False)
def load_dataset_bundle(dataset_key: str) -> dict[str, Any]:
    data_dir = DATASETS[dataset_key].data_dir
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Direktori data evaluasi tidak ditemukan: {data_dir}")

    summary_df = pd.read_csv(data_dir / "summary.csv")
    distribution_df = pd.read_csv(data_dir / "distribution_summary.csv")
    class_weights_df = pd.read_csv(data_dir / "class_weights.csv")
    grouping_audit = _read_json(data_dir / "grouping_audit.json")
    protocol = _read_json(data_dir / "protocol.json")
    onnx_bench = _read_json(data_dir / "onnx_benchmark.json")

    models: dict[str, dict[str, Any]] = {}
    histories: dict[str, pd.DataFrame] = {}
    confusion_matrices: dict[str, pd.DataFrame] = {}
    class_reports: dict[str, pd.DataFrame] = {}

    for sc in SCENARIO_ORDER:
        models[sc] = _read_json(data_dir / f"model_{sc}.json")
        histories[sc] = pd.read_csv(data_dir / f"history_{sc}.csv")
        confusion_matrices[sc] = pd.read_csv(data_dir / f"confusion_matrix_{sc}.csv", index_col=0)
        class_reports[sc] = pd.read_csv(data_dir / f"classification_report_{sc}.csv", index_col=0)

    onnx_by_scenario = {item["scenario"]: item for item in onnx_bench}
    summary_by_scenario = {row["scenario"]: row.to_dict() for _, row in summary_df.iterrows()}

    return {
        "summary_df": summary_df,
        "summary_by_scenario": summary_by_scenario,
        "distribution_df": distribution_df,
        "class_weights_df": class_weights_df,
        "grouping_audit": grouping_audit,
        "protocol": protocol,
        "onnx_by_scenario": onnx_by_scenario,
        "models": models,
        "histories": histories,
        "confusion_matrices": confusion_matrices,
        "class_reports": class_reports,
    }


def build_comparison_table(dataset_key: str) -> pd.DataFrame:
    bundle = load_dataset_bundle(dataset_key)
    rows = []
    for sc in SCENARIO_ORDER:
        s = bundle["summary_by_scenario"][sc]
        m = bundle["models"][sc]
        o = bundle["onnx_by_scenario"][sc]
        active_pct = (m["trainable_params"] / m["total_params"]) * 100.0
        rows.append(
            {
                "Skenario": SCENARIOS[sc].short_label,
                "Blok Beku": f"{m['frozen_backbone_blocks']}/{m['total_backbone_blocks']} ({m['realized_block_ratio'] * 100:.2f}%)",
                "Parameter Aktif": f"{m['trainable_params']:,} ({active_pct:.2f}%)".replace(",", "."),
                "Best Epoch": int(s["best_epoch"]),
                "Akurasi Uji": f"{s['accuracy'] * 100:.2f}%",
                "CI 95% Akurasi": f"[{s['accuracy_ci95_low'] * 100:.2f}%, {s['accuracy_ci95_high'] * 100:.2f}%]",
                "Macro F1": f"{s['macro_f1']:.4f}",
                "CI 95% Macro F1": f"[{s['macro_f1_ci95_low']:.4f}, {s['macro_f1_ci95_high']:.4f}]",
                "Macro PR-AUC": f"{s['macro_ap']:.4f}",
                "Macro ROC-AUC": f"{s['macro_roc_auc']:.4f}",
                "Latensi ONNX CPU": f"{o['cpu_onnx_ms_mean']:.2f} ms",
            }
        )
    return pd.DataFrame(rows)


def build_per_class_table(dataset_key: str, scenario_key: str) -> pd.DataFrame:
    bundle = load_dataset_bundle(dataset_key)
    rep = bundle["class_reports"][scenario_key]
    rows = []
    for code in LABELS:
        row = rep.loc[code]
        rows.append(
            {
                "Kode": code,
                "Precision": round(float(row["precision"]), 4),
                "Recall": round(float(row["recall"]), 4),
                "F1-Score": round(float(row["f1-score"]), 4),
                "Jumlah Uji": int(round(float(row["support"]))),
            }
        )
    return pd.DataFrame(rows)
