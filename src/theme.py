import streamlit as st

PALETTES = {
    "light": {
        "bg": "#F7F9F8",
        "surface": "#FFFFFF",
        "surface_subtle": "#EFF4F5",
        "text": "#101B1E",
        "muted": "#41545C",
        "border": "#7A8E96",
        "border_soft": "#D1DCE0",
        "accent": "#0D5C63",
        "accent_hover": "#094348",
        "accent_ink": "#FFFFFF",
        "warn_bg": "#FEF6E6",
        "warn_border": "#9A5B00",
        "warn_text": "#5C3600",
        "error_bg": "#FDF2F2",
        "error_border": "#8E1F1F",
        "error_text": "#8E1F1F",
        "focus": "#0D5C63",
        "shadow": "0 2px 10px rgba(16, 27, 30, 0.06)",
    },
    "dark": {
        "bg": "#0F1417",
        "surface": "#171F24",
        "surface_subtle": "#1E2930",
        "text": "#EEF3F5",
        "muted": "#A6BAC2",
        "border": "#6B808A",
        "border_soft": "#2C3B44",
        "accent": "#4FD1C5",
        "accent_hover": "#6EE0D5",
        "accent_ink": "#0B1618",
        "warn_bg": "#292010",
        "warn_border": "#F2B84B",
        "warn_text": "#F6C768",
        "error_bg": "#2B1417",
        "error_border": "#F87171",
        "error_text": "#FCA5A5",
        "focus": "#4FD1C5",
        "shadow": "0 2px 12px rgba(0, 0, 0, 0.35)",
    },
}


def inject_theme_css(theme_mode: str) -> None:
    p = PALETTES[theme_mode]
    css = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@500;600&family=IBM+Plex+Sans:wght@400;500;600;700&family=Source+Serif+4:opsz,wght@8..60,600;8..60,700&display=swap');

:root {{
    --bg: {p["bg"]};
    --surface: {p["surface"]};
    --surface-subtle: {p["surface_subtle"]};
    --text: {p["text"]};
    --muted: {p["muted"]};
    --border: {p["border"]};
    --border-soft: {p["border_soft"]};
    --accent: {p["accent"]};
    --accent-hover: {p["accent_hover"]};
    --accent-ink: {p["accent_ink"]};
    --warn-bg: {p["warn_bg"]};
    --warn-border: {p["warn_border"]};
    --warn-text: {p["warn_text"]};
    --error-bg: {p["error_bg"]};
    --error-border: {p["error_border"]};
    --error-text: {p["error_text"]};
    --focus: {p["focus"]};
    --shadow: {p["shadow"]};
}}

html, body, .stApp {{
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    overflow-x: clip;
}}

#MainMenu, footer, header[data-testid="stHeader"] {{
    visibility: hidden;
    height: 0;
}}

[data-testid="stDecoration"] {{
    display: none;
}}

.block-container {{
    max-width: 1180px;
    padding: 1.25rem 1.5rem 2.75rem 1.5rem !important;
}}

h1, h2, h3 {{
    font-family: 'Source Serif 4', Georgia, serif !important;
    color: var(--text) !important;
    letter-spacing: -0.01em;
}}

p, li, label, span, div {{
    color: inherit;
}}

a[href],
button,
[role="button"],
[role="tab"],
select,
summary,
input[type="radio"],
input[type="checkbox"],
[data-testid="stSelectbox"] > div,
[data-testid="stRadio"] label {{
    cursor: pointer !important;
}}

button:disabled,
[aria-disabled="true"] {{
    cursor: not-allowed !important;
    opacity: 0.55;
}}

button:focus-visible,
a:focus-visible,
input:focus-visible,
select:focus-visible,
[tabindex]:focus-visible {{
    outline: 2px solid var(--focus) !important;
    outline-offset: 2px !important;
}}

/* Clinical masthead */
.masthead {{
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 0.9rem 1.15rem;
    background: var(--surface);
    border: 1px solid var(--border-soft);
    border-top: 3px solid var(--accent);
    border-radius: 6px;
    margin-bottom: 0.9rem;
}}

.masthead-logo {{
    flex: 0 0 auto;
    border-radius: 4px;
    background: #FFFFFF;
    padding: 3px;
    border: 1px solid var(--border-soft);
}}

.masthead-title {{
    font-family: 'Source Serif 4', Georgia, serif;
    font-size: clamp(1.2rem, 2.2vw, 1.65rem);
    font-weight: 700;
    line-height: 1.22;
    color: var(--text);
    margin: 0;
}}

.masthead-sub {{
    font-size: 0.88rem;
    color: var(--muted);
    margin: 0.25rem 0 0 0;
    line-height: 1.4;
}}

/* Streamlit interactive button standardization (minimum 44px touch target) */
.stButton > button {{
    min-height: 44px !important;
    padding: 0.5rem 1rem !important;
    border-radius: 5px !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    border: 1px solid var(--border) !important;
    background-color: var(--surface) !important;
    color: var(--text) !important;
    transition: background-color 150ms ease, border-color 150ms ease;
}}

.stButton > button:hover {{
    background-color: var(--surface-subtle) !important;
    border-color: var(--accent) !important;
}}

.stButton > button[kind="primary"] {{
    background-color: var(--accent) !important;
    border-color: var(--accent) !important;
    color: var(--accent-ink) !important;
}}

.stButton > button[kind="primary"]:hover {{
    background-color: var(--accent-hover) !important;
    border-color: var(--accent-hover) !important;
}}

/* Selectbox, radio, and file uploader surfaces */
[data-testid="stSelectbox"] label,
[data-testid="stRadio"] > label,
[data-testid="stFileUploader"] label {{
    color: var(--text) !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
}}

[data-testid="stSelectbox"] div[data-baseweb="select"] > div {{
    min-height: 44px;
    background-color: var(--surface) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 5px !important;
}}

div[data-baseweb="popover"],
ul[role="listbox"],
ul[role="listbox"] li {{
    background-color: var(--surface) !important;
    color: var(--text) !important;
}}

ul[role="listbox"] li:hover {{
    background-color: var(--surface-subtle) !important;
}}

[data-testid="stRadio"] div[role="radiogroup"] {{
    gap: 0.45rem;
}}

[data-testid="stRadio"] div[role="radiogroup"] label {{
    min-height: 44px;
    display: flex;
    align-items: center;
    padding: 0.35rem 0.65rem;
    background: var(--surface);
    border: 1px solid var(--border-soft);
    border-radius: 5px;
}}

[data-testid="stFileUploaderDropzone"] {{
    background-color: var(--surface) !important;
    border: 1.5px dashed var(--border) !important;
    border-radius: 6px !important;
    color: var(--text) !important;
    padding: 1rem !important;
}}

[data-testid="stFileUploaderDropzone"] small,
[data-testid="stFileUploaderDropzone"] span {{
    color: var(--muted) !important;
}}

[data-testid="stFileUploaderDropzone"] button {{
    min-height: 44px !important;
    background-color: var(--surface-subtle) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
}}

/* Structural panels */
.panel {{
    background: var(--surface);
    border: 1px solid var(--border-soft);
    border-radius: 6px;
    padding: 1.1rem 1.2rem;
    margin-bottom: 1rem;
}}

.focal-panel {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-top: 3px solid var(--accent);
    border-radius: 6px;
    padding: 1.25rem 1.35rem;
    box-shadow: var(--shadow);
    margin-bottom: 1rem;
}}

.section-kicker {{
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--accent);
    margin: 0 0 0.35rem 0;
}}

.focal-diagnosis {{
    font-family: 'Source Serif 4', Georgia, serif;
    font-size: clamp(1.35rem, 2.4vw, 1.9rem);
    font-weight: 700;
    color: var(--text);
    line-height: 1.2;
    margin: 0.15rem 0 0.25rem 0;
}}

.focal-sub {{
    font-size: 0.92rem;
    color: var(--muted);
    margin: 0 0 0.9rem 0;
}}

.stat-strip {{
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.75rem;
    padding-top: 0.85rem;
    border-top: 1px solid var(--border-soft);
}}

.stat-item-label {{
    font-size: 0.78rem;
    color: var(--muted);
    margin: 0 0 0.2rem 0;
}}

.stat-item-value {{
    font-family: 'IBM Plex Mono', monospace;
    font-variant-numeric: tabular-nums;
    font-size: clamp(1.05rem, 1.7vw, 1.3rem);
    font-weight: 600;
    color: var(--text);
    margin: 0;
}}

.meta-list {{
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 0.65rem;
    margin-top: 0.6rem;
}}

.meta-cell {{
    background: var(--surface-subtle);
    border: 1px solid var(--border-soft);
    border-radius: 4px;
    padding: 0.6rem 0.75rem;
}}

.meta-cell-k {{
    font-size: 0.75rem;
    color: var(--muted);
    margin: 0;
}}

.meta-cell-v {{
    font-family: 'IBM Plex Mono', monospace;
    font-variant-numeric: tabular-nums;
    font-size: 0.94rem;
    font-weight: 600;
    color: var(--text);
    margin: 0.15rem 0 0 0;
}}

/* Empty, notice, and error states */
.empty-box {{
    background: var(--surface);
    border: 1px solid var(--border-soft);
    border-radius: 6px;
    padding: 1.5rem 1.4rem;
    margin-bottom: 1rem;
}}

.empty-box h3 {{
    margin: 0 0 0.45rem 0;
    font-size: 1.25rem;
}}

.empty-box p {{
    color: var(--muted);
    margin: 0 0 0.9rem 0;
    line-height: 1.55;
}}

.step-row {{
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 0.6rem;
    margin-top: 0.5rem;
}}

.step-pill {{
    background: var(--surface-subtle);
    border: 1px solid var(--border-soft);
    border-radius: 4px;
    padding: 0.55rem 0.65rem;
    font-size: 0.82rem;
    color: var(--text);
    line-height: 1.35;
}}

.step-num {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.74rem;
    font-weight: 600;
    color: var(--accent);
    display: block;
    margin-bottom: 0.15rem;
}}

.disclaimer-box {{
    background: var(--warn-bg);
    border: 1px solid var(--warn-border);
    border-radius: 5px;
    padding: 0.85rem 1rem;
    color: var(--warn-text);
    font-size: 0.88rem;
    line-height: 1.5;
    margin-top: 0.9rem;
}}

.error-box {{
    background: var(--error-bg);
    border: 1px solid var(--error-border);
    border-radius: 5px;
    padding: 0.9rem 1rem;
    color: var(--error-text);
    font-size: 0.9rem;
    line-height: 1.5;
    margin-bottom: 1rem;
}}

/* Accessible HTML tables with horizontal scroll on narrow screens */
.table-wrap {{
    width: 100%;
    overflow-x: auto;
    border: 1px solid var(--border-soft);
    border-radius: 6px;
    background: var(--surface);
    margin: 0.5rem 0 1rem 0;
}}

table.clinical-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.87rem;
}}

table.clinical-table th {{
    text-align: left;
    padding: 0.65rem 0.75rem;
    background: var(--surface-subtle);
    color: var(--text);
    font-weight: 600;
    border-bottom: 1px solid var(--border);
    white-space: nowrap;
}}

table.clinical-table td {{
    padding: 0.6rem 0.75rem;
    border-bottom: 1px solid var(--border-soft);
    color: var(--text);
    vertical-align: top;
}}

table.clinical-table tr:last-child td {{
    border-bottom: none;
}}

table.clinical-table td.num,
table.clinical-table th.num {{
    font-family: 'IBM Plex Mono', monospace;
    font-variant-numeric: tabular-nums;
    white-space: nowrap;
}}

.chart-title {{
    font-family: 'Source Serif 4', Georgia, serif;
    font-size: 1.08rem;
    font-weight: 700;
    color: var(--text);
    margin: 0 0 0.25rem 0;
}}

.chart-caption {{
    font-size: 0.84rem;
    color: var(--muted);
    margin: 0 0 0.65rem 0;
}}

[data-testid="stVegaLiteChart"] {{
    background: var(--surface);
    border: 1px solid var(--border-soft);
    border-radius: 6px;
    padding: 0.4rem 0.5rem 0.25rem 0.5rem;
    margin-bottom: 1rem;
}}

[data-testid="stImage"] {{
    background: var(--surface);
    border: 1px solid var(--border-soft);
    border-radius: 6px;
    padding: 0.5rem;
}}

[data-testid="stImageCaption"] {{
    color: var(--muted) !important;
    font-size: 0.82rem !important;
}}

.site-footer {{
    margin-top: 2rem;
    padding-top: 0.9rem;
    border-top: 1px solid var(--border-soft);
    font-size: 0.82rem;
    color: var(--muted);
    display: flex;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 0.5rem;
}}

/* Tablet breakpoint */
@media (max-width: 960px) {{
    .block-container {{
        padding: 1rem 1rem 2rem 1rem !important;
    }}
    .step-row {{
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }}
}}

/* Mobile breakpoint */
@media (max-width: 640px) {{
    .block-container {{
        padding: 0.75rem 0.75rem 1.75rem 0.75rem !important;
    }}
    .masthead {{
        padding: 0.75rem 0.85rem;
        gap: 0.75rem;
    }}
    .stat-strip {{
        grid-template-columns: 1fr;
        gap: 0.6rem;
    }}
    .meta-list,
    .step-row {{
        grid-template-columns: 1fr;
    }}
}}

@media (prefers-reduced-motion: reduce) {{
    *, *::before, *::after {{
        transition: none !important;
        animation: none !important;
    }}
}}
</style>
"""
    st.markdown(css, unsafe_allow_html=True)
