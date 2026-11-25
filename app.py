import base64
import html

import pandas as pd
import streamlit as st

from src.charts import (
    build_confusion_matrix_chart,
    build_latency_chart,
    build_loss_curve_chart,
    build_probability_chart,
    build_val_f1_curve_chart,
)
from src.config import ASSETS_DIR, CLASS_INFO, DATASETS, LABELS, SCENARIO_ORDER, SCENARIOS
from src.inference import InferenceError, run_prediction
from src.metrics_loader import build_comparison_table, build_per_class_table, load_dataset_bundle
from src.theme import inject_theme_css

st.set_page_config(
    page_title="Klasifikasi Gangguan Mata EfficientNet-B3",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_data(show_spinner=False)
def _load_usu_logo_data_uri() -> str:
    logo_path = ASSETS_DIR / "logo_usu.png"
    if not logo_path.is_file():
        return ""
    encoded = base64.b64encode(logo_path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _render_html_table(df: pd.DataFrame, numeric_cols: set[str] | None = None) -> None:
    numeric_cols = numeric_cols or set()
    headers = []
    for col in df.columns:
        cls = ' class="num"' if col in numeric_cols else ""
        headers.append(f"<th{cls}>{html.escape(str(col))}</th>")

    body_rows = []
    for _, row in df.iterrows():
        cells = []
        for col in df.columns:
            cls = ' class="num"' if col in numeric_cols else ""
            cells.append(f"<td{cls}>{html.escape(str(row[col]))}</td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")

    table_html = (
        '<div class="table-wrap"><table class="clinical-table">'
        f"<thead><tr>{''.join(headers)}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table></div>"
    )
    st.markdown(table_html, unsafe_allow_html=True)


def _render_header() -> None:
    logo_uri = _load_usu_logo_data_uri()
    logo_html = (
        f'<img src="{logo_uri}" alt="Logo Universitas Sumatera Utara" '
        'width="46" height="46" class="masthead-logo" />'
        if logo_uri
        else ""
    )

    st.markdown(
        f"""
        <header class="masthead" role="banner">
          {logo_html}
          <div>
            <h1 class="masthead-title">Klasifikasi Gangguan Mata Berbasis Citra Fundus (EfficientNet-B3)</h1>
            <p class="masthead-sub">
              Evaluasi Strategi Layer Freezing pada ODIR-5K dan Retinal Fundus 50K dengan Inferensi ONNX Runtime (CPU)
            </p>
          </div>
        </header>
        """,
        unsafe_allow_html=True,
    )

    nav_col1, nav_col2, spacer_col, theme_col = st.columns([2.2, 2.2, 3.6, 1.8], gap="small")
    with nav_col1:
        if st.button(
            "Pengujian Citra Fundus",
            key="nav_predict",
            type="primary" if st.session_state.page == "predict" else "secondary",
            width="stretch",
        ):
            st.session_state.page = "predict"
            st.rerun()
    with nav_col2:
        if st.button(
            "Dasbor Evaluasi",
            key="nav_dashboard",
            type="primary" if st.session_state.page == "dashboard" else "secondary",
            width="stretch",
        ):
            st.session_state.page = "dashboard"
            st.rerun()
    with theme_col:
        next_label = "Mode Gelap" if st.session_state.theme == "light" else "Mode Terang"
        if st.button(next_label, key="toggle_theme", width="stretch"):
            st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"
            st.rerun()


def _render_prediction_page() -> None:
    ctrl_col, main_col = st.columns([4.4, 7.6], gap="large")

    with ctrl_col:
        st.markdown('<p class="section-kicker">Konfigurasi Model ONNX</p>', unsafe_allow_html=True)

        dataset_key = st.selectbox(
            "Dataset Referensi Model",
            options=list(DATASETS.keys()),
            format_func=lambda k: DATASETS[k].label,
            key="pred_dataset",
        )

        scenario_key = st.radio(
            "Skenario Layer Freezing",
            options=SCENARIO_ORDER,
            format_func=lambda k: SCENARIOS[k].label,
            key="pred_scenario",
        )

        uploaded = st.file_uploader(
            "Unggah Citra Fundus Retina",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            help="Format JPG, PNG, BMP, atau WEBP dengan ukuran maksimum 15 MB.",
            key="fundus_uploader",
        )

        try:
            bundle = load_dataset_bundle(dataset_key)
            s_row = bundle["summary_by_scenario"][scenario_key]
            m_info = bundle["models"][scenario_key]
            o_info = bundle["onnx_by_scenario"][scenario_key]
            active_pct = (m_info["trainable_params"] / m_info["total_params"]) * 100.0

            st.markdown(
                f"""
                <div class="panel">
                  <p class="section-kicker">Profil Model Aktif</p>
                  <p style="margin:0;font-size:0.88rem;color:var(--muted);line-height:1.45;">
                    {html.escape(SCENARIOS[scenario_key].block_note)}
                  </p>
                  <div class="meta-list">
                    <div class="meta-cell">
                      <p class="meta-cell-k">Macro F1 Uji</p>
                      <p class="meta-cell-v">{s_row["macro_f1"]:.4f}</p>
                    </div>
                    <div class="meta-cell">
                      <p class="meta-cell-k">Macro PR-AUC</p>
                      <p class="meta-cell-v">{s_row["macro_ap"]:.4f}</p>
                    </div>
                    <div class="meta-cell">
                      <p class="meta-cell-k">Parameter Aktif</p>
                      <p class="meta-cell-v">{active_pct:.2f}%</p>
                    </div>
                    <div class="meta-cell">
                      <p class="meta-cell-k">Ukuran ONNX</p>
                      <p class="meta-cell-v">{o_info["onnx_mb"]:.2f} MB</p>
                    </div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        except Exception:
            st.markdown(
                '<div class="error-box"><strong>Data profil model tidak terbaca.</strong> '
                "Periksa kelengkapan folder data evaluasi sebelum melanjutkan.</div>",
                unsafe_allow_html=True,
            )

    with main_col:
        if uploaded is None:
            st.markdown(
                """
                <div class="empty-box">
                  <p class="section-kicker">Siap Menerima Citra Fundus</p>
                  <h3>Unggah foto fundus retina di panel kiri untuk memulai prediksi</h3>
                  <p>
                    Sistem akan menjalankan alur prapemrosesan identik dengan protokol pengujian skripsi,
                    lalu menghitung probabilitas tujuh kelas kondisi mata melalui model ONNX pada CPU.
                  </p>
                  <div class="step-row">
                    <div class="step-pill"><span class="step-num">01</span>Koreksi orientasi EXIF dan konversi RGB</div>
                    <div class="step-pill"><span class="step-num">02</span>Pemotongan tepi hitam (black-margin crop)</div>
                    <div class="step-pill"><span class="step-num">03</span>Letterbox padding 300 x 300 piksel</div>
                    <div class="step-pill"><span class="step-num">04</span>Normalisasi ImageNet dan inferensi ONNX</div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            ref_df = pd.DataFrame(
                [
                    {
                        "Kode": CLASS_INFO[c].code,
                        "Kondisi Mata": f"{CLASS_INFO[c].name_id} ({CLASS_INFO[c].name_en})",
                        "Ciri Visual Utama pada Citra Fundus": CLASS_INFO[c].visual_cue,
                    }
                    for c in LABELS
                ]
            )
            st.markdown('<p class="chart-title">Daftar Tujuh Kelas Kondisi Mata Target</p>', unsafe_allow_html=True)
            _render_html_table(ref_df, numeric_cols={"Kode"})
        else:
            raw_bytes = uploaded.getvalue()
            try:
                with st.spinner("Menjalankan prapemrosesan citra dan inferensi ONNX pada CPU..."):
                    result = run_prediction(raw_bytes, uploaded.name, dataset_key, scenario_key)
            except InferenceError as err:
                st.markdown(
                    f'<div class="error-box" role="alert"><strong>Gagal memproses citra:</strong> '
                    f"{html.escape(str(err))}</div>",
                    unsafe_allow_html=True,
                )
                return
            except Exception:
                st.markdown(
                    '<div class="error-box" role="alert"><strong>Terjadi gangguan saat inferensi ONNX.</strong> '
                    "Pastikan berkas gambar utuh dan model ONNX sesuai, lalu coba unggah kembali.</div>",
                    unsafe_allow_html=True,
                )
                return

            sorted_probs = sorted(result.probabilities.items(), key=lambda kv: kv[1], reverse=True)
            top_code, top_prob = sorted_probs[0]
            second_code, second_prob = sorted_probs[1]
            margin_pp = (top_prob - second_prob) * 100.0
            top_info = CLASS_INFO[top_code]

            st.markdown(
                f"""
                <section class="focal-panel" aria-live="polite">
                  <p class="section-kicker">Hasil Prediksi Utama ({html.escape(DATASETS[dataset_key].short_label)} · {html.escape(SCENARIOS[scenario_key].short_label)})</p>
                  <p class="focal-diagnosis">{html.escape(top_info.code)} : {html.escape(top_info.name_id)}</p>
                  <p class="focal-sub">{html.escape(top_info.name_en)} &middot; {html.escape(top_info.visual_cue)}</p>
                  <div class="stat-strip">
                    <div>
                      <p class="stat-item-label">Probabilitas Kelas Teratas</p>
                      <p class="stat-item-value">{top_prob * 100:.2f}%</p>
                    </div>
                    <div>
                      <p class="stat-item-label">Selisih dari Peringkat 2 ({html.escape(second_code)})</p>
                      <p class="stat-item-value">+{margin_pp:.2f} pp</p>
                    </div>
                    <div>
                      <p class="stat-item-label">Waktu Inferensi ONNX (CPU)</p>
                      <p class="stat-item-value">{result.latency_ms:.1f} ms</p>
                    </div>
                  </div>
                </section>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                '<p class="chart-title">Distribusi Probabilitas Prediksi pada Tujuh Kelas Kondisi Mata</p>'
                '<p class="chart-caption">Diurutkan dari probabilitas tertinggi berdasarkan keluaran softmax model ONNX.</p>',
                unsafe_allow_html=True,
            )
            st.altair_chart(
                build_probability_chart(result.probabilities, st.session_state.theme),
                width="stretch",
            )

            img_col1, img_col2 = st.columns(2, gap="medium")
            orig_w, orig_h = result.original_size
            crop_w, crop_h = result.cropped_size
            with img_col1:
                st.image(
                    result.original_rgb,
                    caption=f"Citra Unggahan Asli ({orig_w} x {orig_h} piksel)",
                    width="stretch",
                )
            with img_col2:
                st.image(
                    result.preprocessed_rgb,
                    caption=f"Hasil Black-Margin Crop ({crop_w} x {crop_h}) & Letterbox 300 x 300",
                    width="stretch",
                )

            rank_df = pd.DataFrame(
                [
                    {
                        "Peringkat": idx,
                        "Kode": code,
                        "Kondisi Mata": f"{CLASS_INFO[code].name_id} ({CLASS_INFO[code].name_en})",
                        "Probabilitas": f"{prob * 100:.2f}%",
                    }
                    for idx, (code, prob) in enumerate(sorted_probs[:3], start=1)
                ]
            )
            st.markdown('<p class="chart-title">Tiga Prediksi Probabilitas Tertinggi</p>', unsafe_allow_html=True)
            _render_html_table(rank_df, numeric_cols={"Peringkat", "Kode", "Probabilitas"})

        st.markdown(
            """
            <div class="disclaimer-box" role="note">
              <strong>Batas Penggunaan Klinis:</strong> Keluaran aplikasi ini merupakan hasil model penelitian akademik
              berbasis <em>deep learning</em> untuk keperluan evaluasi ilmiah dan demonstrasi skripsi. Prediksi ini
              bukan alat diagnostik medis resmi dan tidak menggantikan pemeriksaan dokter spesialis mata.
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_dashboard_page() -> None:
    st.markdown('<p class="section-kicker">Rekapitulasi Pengujian Data Uji Terkunci</p>', unsafe_allow_html=True)

    filter_col1, filter_col2 = st.columns([5, 5], gap="medium")
    with filter_col1:
        dataset_key = st.radio(
            "Pilih Dataset Evaluasi",
            options=list(DATASETS.keys()),
            format_func=lambda k: DATASETS[k].label,
            horizontal=True,
            key="dash_dataset",
        )

    try:
        bundle = load_dataset_bundle(dataset_key)
    except Exception:
        st.markdown(
            '<div class="error-box" role="alert"><strong>Berkas hasil evaluasi tidak dapat dimuat.</strong> '
            "Pastikan folder data/odir5k dan data/fundus50k berisi berkas CSV dan JSON hasil pengujian.</div>",
            unsafe_allow_html=True,
        )
        return

    with filter_col2:
        focus_scenario = st.selectbox(
            "Skenario Detail untuk Confusion Matrix & Kurva Loss",
            options=SCENARIO_ORDER,
            index=2,
            format_func=lambda k: SCENARIOS[k].label,
            key="dash_scenario",
        )

    best_f1_sc = max(SCENARIO_ORDER, key=lambda sc: bundle["summary_by_scenario"][sc]["macro_f1"])
    best_s = bundle["summary_by_scenario"][best_f1_sc]
    best_onnx = bundle["onnx_by_scenario"][best_f1_sc]
    speedup = best_onnx["cpu_pytorch_ms_mean"] / best_onnx["cpu_onnx_ms_mean"]
    audit = bundle["grouping_audit"]

    retained_str = f"{audit['retained_images']:,}".replace(",", ".")
    groups_str = f"{audit['groups']:,}".replace(",", ".")

    st.markdown(
        f"""
        <section class="focal-panel">
          <p class="section-kicker">Ringkasan Kinerja Tertinggi pada {html.escape(DATASETS[dataset_key].short_label)}</p>
          <p class="focal-diagnosis">{html.escape(SCENARIOS[best_f1_sc].label)}</p>
          <p class="focal-sub">
            {html.escape(DATASETS[dataset_key].description)} Total {retained_str} citra bersih
            dalam {groups_str} grup pasien unik tanpa kebocoran data antarpartisi.
          </p>
          <div class="stat-strip">
            <div>
              <p class="stat-item-label">Macro F1 Uji (CI 95%)</p>
              <p class="stat-item-value">{best_s["macro_f1"]:.4f} [{best_s["macro_f1_ci95_low"]:.3f}, {best_s["macro_f1_ci95_high"]:.3f}]</p>
            </div>
            <div>
              <p class="stat-item-label">Macro PR-AUC / ROC-AUC</p>
              <p class="stat-item-value">{best_s["macro_ap"]:.4f} / {best_s["macro_roc_auc"]:.4f}</p>
            </div>
            <div>
              <p class="stat-item-label">Percepatan ONNX vs PyTorch CPU</p>
              <p class="stat-item-value">{speedup:.2f}x ({best_onnx["cpu_onnx_ms_mean"]:.1f} ms)</p>
            </div>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    if dataset_key == "fundus50k":
        st.markdown(
            """
            <div class="disclaimer-box" role="note">
              <strong>Catatan Metodologis Retinal Fundus 50K:</strong> Kelas <em>Diabetic Retinopathy</em> (D) pada repositori
              ini berasal dari koleksi EyePACS beresolusi tinggi, sedangkan enam kelas lainnya berasal dari ODIR berukuran
              512 x 512 piksel sehingga skor F1 kelas D mencapai 0,990 sampai 0,994. Audit lanjutan pada enam kelas non-D
              tetap mengonfirmasi urutan kinerja monotonik: Moderate Fine-Tuning (0,6703) lebih tinggi daripada Shallow
              Fine-Tuning (0,6404) dan Feature Extraction Only (0,6232).
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        '<p class="chart-title">Perbandingan Metrik Uji dan Kompleksitas Ketiga Skenario Layer Freezing</p>',
        unsafe_allow_html=True,
    )
    comp_df = build_comparison_table(dataset_key)
    _render_html_table(
        comp_df,
        numeric_cols={
            "Blok Beku",
            "Parameter Aktif",
            "Best Epoch",
            "Akurasi Uji",
            "CI 95% Akurasi",
            "Macro F1",
            "CI 95% Macro F1",
            "Macro PR-AUC",
            "Macro ROC-AUC",
            "Latensi ONNX CPU",
        },
    )

    curve_col1, curve_col2 = st.columns(2, gap="large")
    with curve_col1:
        st.markdown(
            '<p class="chart-title">Bagaimana perkembangan Macro F1-score validasi tiap epoch?</p>'
            '<p class="chart-caption">Perbandingan stabilitas konvergensi ketiga skenario pada data validasi.</p>',
            unsafe_allow_html=True,
        )
        st.altair_chart(
            build_val_f1_curve_chart(bundle["histories"], st.session_state.theme),
            width="stretch",
        )

    with curve_col2:
        st.markdown(
            f'<p class="chart-title">Bagaimana kurva Training Loss dan Validation Loss ({html.escape(SCENARIOS[focus_scenario].short_label)})?</p>'
            '<p class="chart-caption">Memantau kesenjangan generalisasi selama pelatihan hingga titik early stopping.</p>',
            unsafe_allow_html=True,
        )
        st.altair_chart(
            build_loss_curve_chart(bundle["histories"][focus_scenario], st.session_state.theme),
            width="stretch",
        )

    err_col1, err_col2 = st.columns([6.2, 5.8], gap="large")
    with err_col1:
        st.markdown(
            f'<p class="chart-title">Di kelas mana model paling sering keliru ({html.escape(SCENARIOS[focus_scenario].short_label)})?</p>'
            '<p class="chart-caption">Confusion matrix 7 x 7 pada data uji terkunci; intensitas warna menunjukkan recall per baris.</p>',
            unsafe_allow_html=True,
        )
        st.altair_chart(
            build_confusion_matrix_chart(bundle["confusion_matrices"][focus_scenario], st.session_state.theme),
            width="stretch",
        )

    with err_col2:
        st.markdown(
            f'<p class="chart-title">Rincian Precision, Recall, dan F1-Score per Kelas ({html.escape(SCENARIOS[focus_scenario].short_label)})</p>'
            '<p class="chart-caption">Menunjukkan ketimpangan performa pada kelas langka seperti Hipertensi (H).</p>',
            unsafe_allow_html=True,
        )
        per_class_df = build_per_class_table(dataset_key, focus_scenario)
        _render_html_table(per_class_df, numeric_cols={"Kode", "Precision", "Recall", "F1-Score", "Jumlah Uji"})

    eff_col1, eff_col2 = st.columns([6, 6], gap="large")
    with eff_col1:
        st.markdown(
            '<p class="chart-title">Seberapa cepat inferensi ONNX CPU dibanding PyTorch?</p>'
            '<p class="chart-caption">Rata-rata 100 iterasi inferensi tunggal setelah 20 warmup pada input 300 x 300.</p>',
            unsafe_allow_html=True,
        )
        st.altair_chart(
            build_latency_chart(bundle["onnx_by_scenario"], st.session_state.theme),
            width="stretch",
        )

    with eff_col2:
        st.markdown(
            '<p class="chart-title">Verifikasi Paritas Numerik dan Ukuran Berkas ONNX</p>'
            '<p class="chart-caption">Batas toleransi selisih logit maksimum PyTorch vs ONNX ditetapkan kurang dari 1e-4.</p>',
            unsafe_allow_html=True,
        )
        onnx_rows = []
        for sc in SCENARIO_ORDER:
            o = bundle["onnx_by_scenario"][sc]
            onnx_rows.append(
                {
                    "Skenario": SCENARIOS[sc].short_label,
                    "Ukuran ONNX": f"{o['onnx_mb']:.3f} MB",
                    "Selisih Logit Maks.": f"{o['max_abs_logit_diff']:.2e}",
                    "Status Paritas": "Lulus" if o["parity_passed"] else "Gagal",
                }
            )
        _render_html_table(
            pd.DataFrame(onnx_rows),
            numeric_cols={"Ukuran ONNX", "Selisih Logit Maks."},
        )


def main() -> None:
    if "page" not in st.session_state:
        st.session_state.page = "predict"
    if "theme" not in st.session_state:
        st.session_state.theme = "light"

    inject_theme_css(st.session_state.theme)
    _render_header()

    if st.session_state.page == "predict":
        _render_prediction_page()
    else:
        _render_dashboard_page()

    st.markdown(
        """
        <footer class="site-footer" role="contentinfo">
          <span>Rizki Wahyudi (211401036) &middot; S-1 Ilmu Komputer, Fasilkom-TI Universitas Sumatera Utara</span>
          <span>Model EfficientNet-B3 ONNX Opset 17 &middot; Prototipe Penelitian Akademik</span>
        </footer>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
