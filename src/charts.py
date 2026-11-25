from typing import Any

import altair as alt
import pandas as pd

from src.config import CLASS_INFO, LABELS, SCENARIO_ORDER, SCENARIOS


def _chart_palette(theme_mode: str) -> dict[str, Any]:
    if theme_mode == "dark":
        return {
            "bg": "#171F24",
            "text": "#EEF3F5",
            "muted": "#A6BAC2",
            "grid": "#2A373F",
            "primary": "#4FD1C5",
            "secondary": "#6E8591",
            "series": ["#A6BAC2", "#F6C768", "#4FD1C5"],
            "heat_low": "#1B262C",
            "heat_high": "#4FD1C5",
        }
    return {
        "bg": "#FFFFFF",
        "text": "#101B1E",
        "muted": "#41545C",
        "grid": "#E2E9EC",
        "primary": "#0D5C63",
        "secondary": "#7A8E96",
        "series": ["#5B6F78", "#9A5B00", "#0D5C63"],
        "heat_low": "#EEF4F5",
        "heat_high": "#0D5C63",
    }


def _apply_common_config(chart: alt.Chart, pal: dict[str, Any]) -> alt.Chart:
    return (
        chart.configure_view(strokeWidth=0, fill=pal["bg"])
        .configure(background=pal["bg"], padding={"top": 12, "right": 28, "bottom": 10, "left": 10})
        .configure_axis(
            labelColor=pal["muted"],
            titleColor=pal["text"],
            gridColor=pal["grid"],
            domainColor=pal["grid"],
            tickColor=pal["grid"],
            labelFont="IBM Plex Sans, sans-serif",
            titleFont="IBM Plex Sans, sans-serif",
            labelFontSize=12,
            titleFontSize=12,
        )
        .configure_legend(
            labelColor=pal["text"],
            titleColor=pal["text"],
            labelFont="IBM Plex Sans, sans-serif",
            titleFont="IBM Plex Sans, sans-serif",
            labelFontSize=12,
            orient="top",
            direction="horizontal",
        )
    )


def build_probability_chart(probabilities: dict[str, float], theme_mode: str) -> alt.Chart:
    pal = _chart_palette(theme_mode)
    top_code = max(probabilities, key=probabilities.get)
    rows = []
    for code in LABELS:
        info = CLASS_INFO[code]
        pct = probabilities[code] * 100.0
        rows.append(
            {
                "Kelas": f"{code} ({info.name_id})",
                "Kode": code,
                "Probabilitas (%)": pct,
                "Label": f"{pct:.2f}%",
                "Status": "Prediksi Utama" if code == top_code else "Kelas Lain",
            }
        )
    df = pd.DataFrame(rows).sort_values("Probabilitas (%)", ascending=False)
    ordered_classes = df["Kelas"].tolist()

    base = alt.Chart(df).encode(
        y=alt.Y("Kelas:N", sort=ordered_classes, title=None, axis=alt.Axis(labelLimit=210)),
        x=alt.X(
            "Probabilitas (%):Q",
            scale=alt.Scale(domain=[0, 112]),
            title="Probabilitas Softmax (%)",
        ),
    )

    bars = base.mark_bar(cornerRadiusEnd=3, height=22).encode(
        color=alt.Color(
            "Status:N",
            scale=alt.Scale(
                domain=["Prediksi Utama", "Kelas Lain"],
                range=[pal["primary"], pal["secondary"]],
            ),
            legend=None,
        ),
        tooltip=[
            alt.Tooltip("Kelas:N", title="Kondisi"),
            alt.Tooltip("Probabilitas (%):Q", title="Probabilitas (%)", format=".2f"),
        ],
    )

    labels = base.mark_text(
        align="left",
        dx=6,
        font="IBM Plex Mono, monospace",
        fontSize=12,
        fontWeight=600,
        color=pal["text"],
    ).encode(text="Label:N")

    return _apply_common_config((bars + labels).properties(height=270), pal)


def build_val_f1_curve_chart(histories: dict[str, pd.DataFrame], theme_mode: str) -> alt.Chart:
    pal = _chart_palette(theme_mode)
    frames = []
    for sc in SCENARIO_ORDER:
        h = histories[sc][["epoch", "val_f1_macro"]].copy()
        h["Skenario"] = SCENARIOS[sc].short_label
        frames.append(h)
    df = pd.concat(frames, ignore_index=True)

    domain = [SCENARIOS[sc].short_label for sc in SCENARIO_ORDER]
    dash_scale = [[6, 4], [2, 2], [1, 0]]

    chart = (
        alt.Chart(df)
        .mark_line(point=alt.OverlayMarkDef(size=42), strokeWidth=2.4)
        .encode(
            x=alt.X("epoch:Q", title="Epoch", axis=alt.Axis(tickMinStep=2)),
            y=alt.Y("val_f1_macro:Q", title="Validation Macro F1-Score", scale=alt.Scale(zero=False)),
            color=alt.Color("Skenario:N", scale=alt.Scale(domain=domain, range=pal["series"]), title=None),
            strokeDash=alt.StrokeDash("Skenario:N", scale=alt.Scale(domain=domain, range=dash_scale), title=None),
            tooltip=[
                alt.Tooltip("Skenario:N", title="Skenario"),
                alt.Tooltip("epoch:Q", title="Epoch"),
                alt.Tooltip("val_f1_macro:Q", title="Val Macro F1", format=".4f"),
            ],
        )
        .properties(height=300)
    )
    return _apply_common_config(chart, pal)


def build_loss_curve_chart(history_df: pd.DataFrame, theme_mode: str) -> alt.Chart:
    pal = _chart_palette(theme_mode)
    melted = history_df[["epoch", "train_loss", "val_loss"]].melt(
        id_vars="epoch",
        value_vars=["train_loss", "val_loss"],
        var_name="Tipe",
        value_name="Loss",
    )
    melted["Tipe"] = melted["Tipe"].map({"train_loss": "Training Loss", "val_loss": "Validation Loss"})

    chart = (
        alt.Chart(melted)
        .mark_line(point=alt.OverlayMarkDef(size=38), strokeWidth=2.3)
        .encode(
            x=alt.X("epoch:Q", title="Epoch", axis=alt.Axis(tickMinStep=2)),
            y=alt.Y("Loss:Q", title="Nilai Loss", scale=alt.Scale(zero=False)),
            color=alt.Color(
                "Tipe:N",
                scale=alt.Scale(domain=["Training Loss", "Validation Loss"], range=[pal["secondary"], pal["primary"]]),
                title=None,
            ),
            strokeDash=alt.StrokeDash(
                "Tipe:N",
                scale=alt.Scale(domain=["Training Loss", "Validation Loss"], range=[[5, 4], [1, 0]]),
                title=None,
            ),
            tooltip=[
                alt.Tooltip("Tipe:N", title="Kurva"),
                alt.Tooltip("epoch:Q", title="Epoch"),
                alt.Tooltip("Loss:Q", title="Loss", format=".4f"),
            ],
        )
        .properties(height=300)
    )
    return _apply_common_config(chart, pal)


def build_confusion_matrix_chart(cm_df: pd.DataFrame, theme_mode: str) -> alt.Chart:
    pal = _chart_palette(theme_mode)
    rows = []
    for actual in LABELS:
        row_total = float(cm_df.loc[actual, LABELS].sum())
        for pred in LABELS:
            count = int(cm_df.loc[actual, pred])
            pct = (count / row_total * 100.0) if row_total > 0 else 0.0
            rows.append(
                {
                    "Aktual": actual,
                    "Prediksi": pred,
                    "Jumlah": count,
                    "Proporsi Baris (%)": pct,
                }
            )
    df = pd.DataFrame(rows)

    base = alt.Chart(df).encode(
        x=alt.X("Prediksi:N", sort=LABELS, title="Label Prediksi"),
        y=alt.Y("Aktual:N", sort=LABELS, title="Label Aktual"),
    )

    rect = base.mark_rect(stroke=pal["bg"], strokeWidth=2).encode(
        color=alt.Color(
            "Proporsi Baris (%):Q",
            scale=alt.Scale(domain=[0, 100], range=[pal["heat_low"], pal["heat_high"]]),
            title="Recall Baris (%)",
        ),
        tooltip=[
            alt.Tooltip("Aktual:N", title="Aktual"),
            alt.Tooltip("Prediksi:N", title="Prediksi"),
            alt.Tooltip("Jumlah:Q", title="Jumlah Citra"),
            alt.Tooltip("Proporsi Baris (%):Q", title="Proporsi Baris (%)", format=".1f"),
        ],
    )

    # Flip annotation ink at 50% fill intensity to preserve text legibility.
    high_ink = "#0B1618" if theme_mode == "dark" else "#FFFFFF"
    text = base.mark_text(font="IBM Plex Mono, monospace", fontSize=12, fontWeight=600).encode(
        text="Jumlah:Q",
        color=alt.condition(
            alt.datum["Proporsi Baris (%)"] >= 50,
            alt.value(high_ink),
            alt.value(pal["text"]),
        ),
    )

    return _apply_common_config((rect + text).properties(height=340), pal)


def build_latency_chart(onnx_by_scenario: dict[str, dict[str, Any]], theme_mode: str) -> alt.Chart:
    pal = _chart_palette(theme_mode)
    rows = []
    for sc in SCENARIO_ORDER:
        item = onnx_by_scenario[sc]
        sc_name = SCENARIOS[sc].short_label
        rows.extend(
            [
                {"Skenario": sc_name, "Runtime": "PyTorch CPU", "Latensi (ms)": float(item["cpu_pytorch_ms_mean"])},
                {"Skenario": sc_name, "Runtime": "ONNX Runtime CPU", "Latensi (ms)": float(item["cpu_onnx_ms_mean"])},
                {"Skenario": sc_name, "Runtime": "PyTorch GPU", "Latensi (ms)": float(item["gpu_pytorch_ms_mean"])},
            ]
        )
    df = pd.DataFrame(rows)
    df["Label"] = df["Latensi (ms)"].map(lambda v: f"{v:.1f} ms")

    rt_domain = ["PyTorch CPU", "ONNX Runtime CPU", "PyTorch GPU"]
    rt_colors = [pal["secondary"], pal["primary"], pal["series"][1]]

    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            y=alt.Y("Skenario:N", sort=[SCENARIOS[s].short_label for s in SCENARIO_ORDER], title=None),
            yOffset=alt.YOffset("Runtime:N", sort=rt_domain),
            x=alt.X("Latensi (ms):Q", title="Latensi Rata-Rata per Citra (ms)"),
            color=alt.Color("Runtime:N", scale=alt.Scale(domain=rt_domain, range=rt_colors), title=None),
            tooltip=[
                alt.Tooltip("Skenario:N", title="Skenario"),
                alt.Tooltip("Runtime:N", title="Runtime"),
                alt.Tooltip("Latensi (ms):Q", title="Latensi (ms)", format=".2f"),
            ],
        )
        .properties(height=270)
    )
    return _apply_common_config(chart, pal)
