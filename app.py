import streamlit as st
import pandas as pd
import numpy as np
import re
import csv
import io
import warnings
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Analisis FP-Growth · Toko Yunita",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

/* ── Main background ── */
.stApp {
    background: #0f1117;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #161b27 !important;
    border-right: 1px solid #1e2535;
}
[data-testid="stSidebar"] * {
    color: #c8d0e0 !important;
}

/* ── Header banner ── */
.hero-banner {
    background: linear-gradient(135deg, #0d1b2a 0%, #1a2744 50%, #0d1b2a 100%);
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    padding: 32px 40px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(59,130,246,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-size: 28px;
    font-weight: 700;
    color: #f0f4ff;
    margin: 0 0 6px 0;
    letter-spacing: -0.5px;
}
.hero-subtitle {
    font-size: 14px;
    color: #6b7fa3;
    margin: 0;
    font-weight: 400;
}
.hero-badge {
    display: inline-block;
    background: #1e3a5f;
    color: #60a5fa;
    border: 1px solid #2d5a8e;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.5px;
    margin-bottom: 12px;
    text-transform: uppercase;
}

/* ── Metric cards ── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
    margin-bottom: 28px;
}
.metric-card {
    background: #161b27;
    border: 1px solid #1e2535;
    border-radius: 12px;
    padding: 20px;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: #2d4a7a; }
.metric-label {
    font-size: 11px;
    font-weight: 600;
    color: #4a5568;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 8px;
}
.metric-value {
    font-size: 28px;
    font-weight: 700;
    color: #f0f4ff;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1;
}
.metric-sub {
    font-size: 12px;
    color: #4a5568;
    margin-top: 4px;
}
.metric-card.blue  { border-left: 3px solid #3b82f6; }
.metric-card.green { border-left: 3px solid #10b981; }
.metric-card.amber { border-left: 3px solid #f59e0b; }
.metric-card.coral { border-left: 3px solid #ef4444; }

/* ── Section headers ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 28px 0 16px 0;
}
.section-title {
    font-size: 16px;
    font-weight: 600;
    color: #d1d9ef;
    margin: 0;
}
.section-line {
    flex: 1;
    height: 1px;
    background: #1e2535;
}
.section-count {
    font-size: 11px;
    font-weight: 600;
    background: #1e2d4a;
    color: #60a5fa;
    border-radius: 20px;
    padding: 2px 10px;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Tables ── */
.dataframe {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 13px !important;
    border-collapse: collapse !important;
    width: 100% !important;
}
.dataframe thead tr th {
    background: #1a2133 !important;
    color: #6b7fa3 !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
    padding: 10px 14px !important;
    border-bottom: 1px solid #1e2535 !important;
    border-top: none !important;
}
.dataframe tbody tr { background: #161b27 !important; }
.dataframe tbody tr:nth-child(even) { background: #131720 !important; }
.dataframe tbody tr:hover { background: #1c2340 !important; }
.dataframe tbody tr td {
    color: #c8d0e0 !important;
    padding: 9px 14px !important;
    border: none !important;
    border-bottom: 1px solid #1a2030 !important;
}

/* ── Upload area ── */
[data-testid="stFileUploader"] {
    background: #161b27 !important;
    border: 2px dashed #1e3a5f !important;
    border-radius: 12px !important;
    padding: 20px !important;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover {
    border-color: #3b82f6 !important;
}

/* ── Slider ── */
.stSlider > div > div > div > div {
    background: #3b82f6 !important;
}

/* ── Buttons ── */
.stDownloadButton > button {
    background: #1e3a5f !important;
    color: #60a5fa !important;
    border: 1px solid #2d5a8e !important;
    border-radius: 8px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    padding: 8px 18px !important;
    transition: all 0.2s !important;
}
.stDownloadButton > button:hover {
    background: #2d5a8e !important;
    border-color: #3b82f6 !important;
    color: #93c5fd !important;
}
.stButton > button {
    background: #1d4ed8 !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 600 !important;
    padding: 10px 24px !important;
    font-size: 14px !important;
    transition: background 0.2s !important;
    width: 100% !important;
}
.stButton > button:hover { background: #1e40af !important; }

/* ── Alert / info boxes ── */
.stAlert {
    border-radius: 10px !important;
    border: none !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 13px !important;
}

/* ── Rule cards ── */
.rule-card {
    background: #161b27;
    border: 1px solid #1e2535;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 12px;
    font-size: 13px;
}
.rule-arrow {
    color: #3b82f6;
    font-size: 18px;
    font-weight: 700;
}
.rule-item { color: #d1d9ef; font-weight: 500; }
.rule-badges { margin-left: auto; display: flex; gap: 8px; }
.badge {
    border-radius: 6px;
    padding: 3px 9px;
    font-size: 11px;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
}
.badge-sup  { background: #1e3a5f; color: #60a5fa; }
.badge-conf { background: #1a3028; color: #34d399; }
.badge-lift { background: #3a2010; color: #fb923c; }

/* ── Lift indicator ── */
.lift-high { color: #10b981 !important; font-weight: 600 !important; }
.lift-med  { color: #f59e0b !important; }
.lift-low  { color: #6b7280 !important; }

/* ── Step indicator ── */
.step-box {
    background: #161b27;
    border: 1px solid #1e2535;
    border-radius: 12px;
    padding: 18px 20px;
    text-align: center;
    color: #6b7fa3;
    font-size: 13px;
}
.step-icon { font-size: 28px; margin-bottom: 8px; }
.step-text { color: #4a5568; font-size: 12px; margin-top: 4px; }

/* ── Success state ── */
.success-bar {
    background: #052e16;
    border: 1px solid #166534;
    border-radius: 10px;
    padding: 12px 18px;
    color: #4ade80;
    font-size: 13px;
    font-weight: 500;
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 16px;
}

/* ── Selectbox & number input ── */
.stSelectbox > div > div, .stNumberInput > div > div > input {
    background: #161b27 !important;
    border-color: #1e2535 !important;
    color: #c8d0e0 !important;
    border-radius: 8px !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0f1117; }
::-webkit-scrollbar-thumb { background: #1e2535; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #2d3748; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid #1e2535 !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #4a5568 !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    padding: 10px 20px !important;
    border: none !important;
}
.stTabs [aria-selected="true"] {
    color: #60a5fa !important;
    border-bottom: 2px solid #3b82f6 !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: #161b27 !important;
    border: 1px solid #1e2535 !important;
    border-radius: 8px !important;
    color: #c8d0e0 !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 13px !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
SATUAN = {
    'PCS','PACK','BH','BKS','RCG','BTL','KG',
    'GR','LTR','DOS','SGT','RLL','LSN','CTN'
}


# ─────────────────────────────────────────────
# PARSING & PROCESSING FUNCTIONS
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def parse_csv_report(file_bytes: bytes) -> pd.DataFrame:
    """
    Parse laporan kasir CSV Toko Yunita menjadi DataFrame bersih.
    Menangani format report campuran dengan baris transaksi & item.
    """
    teks   = file_bytes.decode("latin1")
    reader = csv.reader(io.StringIO(teks))
    rows   = list(reader)

    current_faktur  = None
    current_tanggal = None
    hasil = []

    for row in rows:
        b = [v.strip() for v in row if v.strip() != ""]
        if not b:
            continue

        is_txn = (
            re.match(r"^\d{1,4}$", b[0])
            and len(b) > 2
            and re.match(r"^\d+/\d+/\d+$", b[1])
        )
        is_item = any(s in b for s in SATUAN)

        if is_txn:
            current_tanggal = b[1]
            current_faktur  = b[2]

        elif is_item and current_faktur:
            for idx, val in enumerate(b):
                if val in SATUAN and idx >= 1:
                    nama  = b[idx - 1]
                    qty   = b[idx + 1] if idx + 1 < len(b) else "1"
                    harga = b[idx + 2] if idx + 2 < len(b) else "0"
                    if nama and not re.match(r"^\d+$", nama):
                        hasil.append({
                            "No_Faktur"  : current_faktur,
                            "Tanggal"    : current_tanggal,
                            "Nama_Barang": nama.upper().strip(),
                            "Satuan"     : val,
                            "QTY"        : qty,
                            "Harga"      : harga,
                        })
                    break

    if not hasil:
        return pd.DataFrame()

    df = pd.DataFrame(hasil)
    df["Nama_Barang"] = df["Nama_Barang"].str.strip().str.replace(r"\s+", " ", regex=True)
    df["QTY"]         = pd.to_numeric(df["QTY"], errors="coerce").fillna(1).astype(int)
    df["Tanggal"]     = pd.to_datetime(df["Tanggal"], format="%m/%d/%Y", errors="coerce")
    df["Harga_Num"]   = (
        df["Harga"]
        .str.replace(",", "", regex=False)
        .str.replace(r"[^\d]", "", regex=True)
    )
    df["Harga_Num"] = pd.to_numeric(df["Harga_Num"], errors="coerce").fillna(0).astype(int)
    df = df.dropna(subset=["No_Faktur", "Nama_Barang"])
    df = df[df["Nama_Barang"] != ""]
    return df.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def run_fpgrowth(
    df: pd.DataFrame,
    min_support: float,
    min_confidence: float,
) -> tuple:
    """
    Jalankan FP-Growth dan hasilkan frequent itemset + aturan asosiasi.
    Return: (frequent_itemsets_df, rules_df, basket_df)
    """
    basket = (
        df.groupby("No_Faktur")["Nama_Barang"]
        .apply(list)
        .reset_index(name="Produk")
    )

    te       = TransactionEncoder()
    te_array = te.fit_transform(basket["Produk"])
    df_enc   = pd.DataFrame(te_array, columns=te.columns_)

    fi = fpgrowth(df_enc, min_support=min_support, use_colnames=True)
    if fi.empty:
        return pd.DataFrame(), pd.DataFrame(), basket

    fi["jumlah_item"]  = fi["itemsets"].apply(len)
    fi["itemsets_str"] = fi["itemsets"].apply(lambda x: " + ".join(sorted(x)))
    fi["support_pct"]  = (fi["support"] * 100).round(2)
    fi = fi.sort_values(["jumlah_item", "support"], ascending=[True, False]).reset_index(drop=True)
    fi.index += 1

    rules = association_rules(fi, metric="confidence", min_threshold=min_confidence)
    if rules.empty:
        return fi, pd.DataFrame(), basket

    rules["antecedents_str"] = rules["antecedents"].apply(lambda x: " + ".join(sorted(x)))
    rules["consequents_str"] = rules["consequents"].apply(lambda x: " + ".join(sorted(x)))
    rules["support"]         = rules["support"].round(4)
    rules["confidence"]      = rules["confidence"].round(4)
    rules["lift"]            = rules["lift"].round(4)
    rules["conviction"]      = rules["conviction"].round(4) if "conviction" in rules.columns else np.nan
    rules = rules.sort_values("lift", ascending=False).reset_index(drop=True)
    rules.index += 1

    return fi, rules, basket


def to_excel_bytes(fi: pd.DataFrame, rules: pd.DataFrame, df_raw: pd.DataFrame) -> bytes:
    """Export hasil ke Excel dengan 3 sheet."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        tabel_fi = fi[["itemsets_str", "support", "support_pct", "jumlah_item"]].rename(columns={
            "itemsets_str": "Itemset (Nama Barang)",
            "support"     : "Support (Desimal)",
            "support_pct" : "Support (%)",
            "jumlah_item" : "Jumlah Item",
        })
        tabel_fi.to_excel(writer, sheet_name="Frequent Itemset", index=True)

        if not rules.empty:
            tabel_rules = rules[[
                "antecedents_str", "consequents_str",
                "support", "confidence", "lift"
            ]].rename(columns={
                "antecedents_str": "Jika Membeli (Antecedent)",
                "consequents_str": "Maka Membeli (Consequent)",
                "support"        : "Support",
                "confidence"     : "Confidence",
                "lift"           : "Lift Ratio",
            })
            tabel_rules.to_excel(writer, sheet_name="Aturan Asosiasi", index=True)

        tabel_raw = df_raw[["No_Faktur", "Tanggal", "Nama_Barang", "Satuan", "QTY", "Harga"]].copy()
        tabel_raw["Tanggal"] = tabel_raw["Tanggal"].dt.strftime("%d/%m/%Y")
        tabel_raw.to_excel(writer, sheet_name="Data Transaksi", index=False)

    return buf.getvalue()


def render_metric(label, value, sub, color_class):
    return f"""
    <div class="metric-card {color_class}">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-sub">{sub}</div>
    </div>"""


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Pengaturan Analisis")
    st.markdown("---")

    st.markdown("**📁 Upload File**")
    uploaded = st.file_uploader(
        "Pilih file laporan CSV kasir",
        type=["csv"],
        help="File laporan penjualan dari sistem kasir Toko Yunita (format .csv)",
    )

    st.markdown("---")
    st.markdown("**🎚️ Parameter FP-Growth**")

    min_support = st.slider(
        "Minimum Support (%)",
        min_value=1, max_value=20, value=2, step=1,
        help="Frekuensi minimum sebuah itemset muncul dalam transaksi. Makin kecil = makin banyak hasil.",
    ) / 100

    min_confidence = st.slider(
        "Minimum Confidence (%)",
        min_value=10, max_value=90, value=30, step=5,
        help="Tingkat kepercayaan minimum aturan asosiasi. Makin besar = aturan makin kuat.",
    ) / 100

    st.markdown("---")
    st.markdown("**🔍 Filter Hasil**")

    filter_itemset = st.selectbox(
        "Tampilkan itemset",
        ["Semua", "1-itemset saja", "2-itemset saja", "3-itemset ke atas"],
    )

    min_lift = st.slider(
        "Lift minimum (aturan asosiasi)",
        min_value=1.0, max_value=10.0, value=1.0, step=0.5,
    )

    st.markdown("---")
    st.caption("🛒 Analisis FP-Growth · Toko Yunita")
    st.caption("Dibuat untuk keperluan skripsi")


# ─────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────

# Hero banner
st.markdown("""
<div class="hero-banner">
    <div class="hero-badge">🛒 Data Mining · FP-Growth</div>
    <div class="hero-title">Analisis Pola Pembelian</div>
    <p class="hero-subtitle">Upload laporan CSV kasir → analisis otomatis → download hasil Excel · Toko Yunita</p>
</div>
""", unsafe_allow_html=True)


# ── STATE: belum ada file
if uploaded is None:
    c1, c2, c3 = st.columns(3)
    for col, icon, title, desc in [
        (c1, "📤", "Upload File CSV", "Pilih file laporan kasir dari sidebar kiri"),
        (c2, "⚙️", "Atur Parameter", "Geser slider support & confidence sesuai kebutuhan"),
        (c3, "📊", "Lihat Hasil", "Tabel frequent itemset & aturan asosiasi muncul otomatis"),
    ]:
        with col:
            st.markdown(f"""
            <div class="step-box">
                <div class="step-icon">{icon}</div>
                <div style="color:#c8d0e0;font-weight:600;font-size:14px">{title}</div>
                <div class="step-text">{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.info("⬅️  Upload file CSV laporan kasir di sidebar untuk memulai analisis.", icon="💡")
    st.stop()


# ── PARSE FILE
with st.spinner("📂 Membaca dan mem-parsing file…"):
    file_bytes = uploaded.read()
    df_raw = parse_csv_report(file_bytes)

if df_raw.empty:
    st.error("❌ File tidak dapat dibaca. Pastikan format file CSV sesuai laporan kasir Toko Yunita.")
    st.stop()

# ── SUCCESS BAR
n_txn  = df_raw["No_Faktur"].nunique()
n_prod = df_raw["Nama_Barang"].nunique()
n_item = len(df_raw)
period_start = df_raw["Tanggal"].min().strftime("%d %b %Y") if not df_raw["Tanggal"].isna().all() else "-"
period_end   = df_raw["Tanggal"].max().strftime("%d %b %Y") if not df_raw["Tanggal"].isna().all() else "-"

st.markdown(f"""
<div class="success-bar">
    ✅ &nbsp; File berhasil dibaca &nbsp;·&nbsp; 
    <strong>{n_txn:,}</strong> transaksi &nbsp;·&nbsp; 
    <strong>{n_prod:,}</strong> produk unik &nbsp;·&nbsp; 
    <strong>{n_item:,}</strong> item &nbsp;·&nbsp; 
    Periode: <strong>{period_start} – {period_end}</strong>
</div>
""", unsafe_allow_html=True)


# ── RUN FP-GROWTH
with st.spinner("🔄 Menjalankan algoritma FP-Growth…"):
    fi, rules, basket = run_fpgrowth(df_raw, min_support, min_confidence)


# ── METRICS
avg_item  = basket["Produk"].apply(len).mean()
n_fi      = len(fi)
n_rules   = len(rules)

st.markdown(f"""
<div class="metric-grid">
    {render_metric("Total Transaksi",     f"{n_txn:,}",         f"Periode {period_start}–{period_end}", "blue")}
    {render_metric("Frequent Itemset",    f"{n_fi:,}",           f"Min. support {min_support*100:.0f}%", "green")}
    {render_metric("Aturan Asosiasi",     f"{n_rules:,}",        f"Min. confidence {min_confidence*100:.0f}%", "amber")}
    {render_metric("Avg Item/Transaksi",  f"{avg_item:.1f}",     f"{n_prod:,} produk unik", "coral")}
</div>
""", unsafe_allow_html=True)


# ── HANDLE EMPTY RESULTS
if fi.empty:
    st.warning(
        f"⚠️ Tidak ada frequent itemset ditemukan dengan support {min_support*100:.0f}%. "
        "Coba kurangi nilai minimum support di sidebar."
    )
    st.stop()


# ── TABS
tab1, tab2, tab3 = st.tabs([
    "📦  Frequent Itemset",
    "📋  Aturan Asosiasi",
    "📊  Data Transaksi",
])


# ══════════════════════════════
# TAB 1: FREQUENT ITEMSET
# ══════════════════════════════
with tab1:
    # Filter
    fi_display = fi.copy()
    if filter_itemset == "1-itemset saja":
        fi_display = fi_display[fi_display["jumlah_item"] == 1]
    elif filter_itemset == "2-itemset saja":
        fi_display = fi_display[fi_display["jumlah_item"] == 2]
    elif filter_itemset == "3-itemset ke atas":
        fi_display = fi_display[fi_display["jumlah_item"] >= 3]

    # Search
    search_fi = st.text_input(
        "🔍 Cari nama barang",
        placeholder="Contoh: rinso, sampoerna…",
        key="search_fi",
    )
    if search_fi:
        fi_display = fi_display[
            fi_display["itemsets_str"].str.contains(search_fi.upper(), na=False)
        ]

    count_display = len(fi_display)
    st.markdown(f"""
    <div class="section-header">
        <span class="section-title">Hasil Frequent Itemset</span>
        <span class="section-count">{count_display} itemset</span>
        <div class="section-line"></div>
    </div>""", unsafe_allow_html=True)

    if fi_display.empty:
        st.info("Tidak ada hasil untuk filter yang dipilih.")
    else:
        tabel_fi_show = fi_display[["itemsets_str", "support_pct", "support", "jumlah_item"]].rename(
            columns={
                "itemsets_str": "Itemset (Nama Barang)",
                "support_pct" : "Support (%)",
                "support"     : "Support (Desimal)",
                "jumlah_item" : "Jumlah Item",
            }
        )
        tabel_fi_show["Support (%)"] = tabel_fi_show["Support (%)"].apply(lambda x: f"{x:.2f}%")

        st.dataframe(
            tabel_fi_show,
            use_container_width=True,
            height=420,
        )

        # Download
        col_d1, col_d2, _ = st.columns([2, 2, 5])
        with col_d1:
            csv_fi = tabel_fi_show.to_csv(index=True).encode("utf-8-sig")
            st.download_button(
                "⬇️ Download CSV",
                data=csv_fi,
                file_name="frequent_itemset.csv",
                mime="text/csv",
                key="dl_fi_csv",
            )
        with col_d2:
            excel_bytes = to_excel_bytes(fi, rules, df_raw)
            st.download_button(
                "⬇️ Download Excel (Semua)",
                data=excel_bytes,
                file_name="Hasil_FP_Growth_TokoYunita.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="dl_all_xlsx",
            )

    # Top 10 visual bar
    st.markdown("---")
    st.markdown("**10 Produk Paling Sering Muncul (1-itemset)**")
    top10 = fi[fi["jumlah_item"] == 1].head(10)[["itemsets_str", "support_pct"]].copy()
    top10.columns = ["Nama Barang", "Support (%)"]
    if not top10.empty:
        st.bar_chart(top10.set_index("Nama Barang")["Support (%)"], height=260)


# ══════════════════════════════
# TAB 2: ATURAN ASOSIASI
# ══════════════════════════════
with tab2:
    if rules.empty:
        st.warning(
            f"⚠️ Tidak ada aturan asosiasi dengan confidence {min_confidence*100:.0f}%. "
            "Coba kurangi nilai minimum confidence di sidebar."
        )
    else:
        # Filter lift
        rules_display = rules[rules["lift"] >= min_lift].copy()

        # Search
        search_rule = st.text_input(
            "🔍 Cari dalam aturan",
            placeholder="Contoh: sampoerna, rinso…",
            key="search_rule",
        )
        if search_rule:
            kw = search_rule.upper()
            rules_display = rules_display[
                rules_display["antecedents_str"].str.contains(kw, na=False)
                | rules_display["consequents_str"].str.contains(kw, na=False)
            ]

        st.markdown(f"""
        <div class="section-header">
            <span class="section-title">Aturan Asosiasi</span>
            <span class="section-count">{len(rules_display)} aturan</span>
            <div class="section-line"></div>
        </div>""", unsafe_allow_html=True)

        # Table view
        tabel_rules_show = rules_display[[
            "antecedents_str", "consequents_str",
            "support", "confidence", "lift"
        ]].rename(columns={
            "antecedents_str": "Jika Membeli (Antecedent)",
            "consequents_str": "Maka Membeli (Consequent)",
            "support"        : "Support",
            "confidence"     : "Confidence",
            "lift"           : "Lift Ratio",
        })
        tabel_rules_show["Support"]    = tabel_rules_show["Support"].apply(lambda x: f"{x:.4f}")
        tabel_rules_show["Confidence"] = tabel_rules_show["Confidence"].apply(lambda x: f"{x*100:.1f}%")
        tabel_rules_show["Lift Ratio"] = tabel_rules_show["Lift Ratio"].apply(lambda x: f"{x:.4f}")

        st.dataframe(tabel_rules_show, use_container_width=True, height=380)

        # Download
        col_d1, col_d2, _ = st.columns([2, 2, 5])
        with col_d1:
            csv_rules = tabel_rules_show.to_csv(index=True).encode("utf-8-sig")
            st.download_button(
                "⬇️ Download CSV",
                data=csv_rules,
                file_name="aturan_asosiasi.csv",
                mime="text/csv",
                key="dl_rules_csv",
            )
        with col_d2:
            excel_bytes2 = to_excel_bytes(fi, rules, df_raw)
            st.download_button(
                "⬇️ Download Excel (Semua)",
                data=excel_bytes2,
                file_name="Hasil_FP_Growth_TokoYunita.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="dl_all_xlsx2",
            )

        # Top 5 highlight cards
        st.markdown("---")
        st.markdown("**✨ Top 5 Aturan dengan Lift Tertinggi**")
        for _, row in rules_display.head(5).iterrows():
            lift_val  = float(row["lift"])
            lift_class = "lift-high" if lift_val >= 2 else ("lift-med" if lift_val >= 1.5 else "lift-low")
            lift_icon  = "🔥" if lift_val >= 2 else ("⚡" if lift_val >= 1.5 else "·")
            sup_pct    = float(row["support"]) * 100
            conf_pct   = float(row["confidence"]) * 100
            st.markdown(f"""
            <div class="rule-card">
                <span class="rule-item">{row['antecedents_str']}</span>
                <span class="rule-arrow">→</span>
                <span class="rule-item">{row['consequents_str']}</span>
                <div class="rule-badges">
                    <span class="badge badge-sup">Sup {sup_pct:.1f}%</span>
                    <span class="badge badge-conf">Conf {conf_pct:.1f}%</span>
                    <span class="badge badge-lift">{lift_icon} Lift {lift_val:.2f}</span>
                </div>
            </div>""", unsafe_allow_html=True)

        # Ringkasan untuk skripsi
        with st.expander("📝 Ringkasan Hasil untuk BAB IV Skripsi"):
            best = rules_display.iloc[0]
            sup_b  = float(best["support"])
            conf_b = float(best["confidence"])
            lift_b = float(best["lift"])
            st.markdown(f"""
**Hasil Analisis FP-Growth — Siap Copy-Paste ke Skripsi:**

---
Dataset yang digunakan adalah data transaksi penjualan Toko Yunita dengan total **{n_txn:,} transaksi** 
dan **{n_prod:,} jenis produk** selama periode {period_start} hingga {period_end}.

Analisis menggunakan algoritma FP-Growth dengan nilai *minimum support* **{min_support*100:.0f}%** 
dan *minimum confidence* **{min_confidence*100:.0f}%** menghasilkan **{n_fi} frequent itemset** 
dan **{len(rules)} aturan asosiasi**.

Aturan dengan nilai lift tertinggi adalah:

> **Jika membeli [{best['antecedents_str']}], maka akan membeli [{best['consequents_str']}]**
> - Support    = {sup_b:.4f} ({sup_b*100:.2f}%)
> - Confidence = {conf_b:.4f} ({conf_b*100:.2f}%)
> - Lift Ratio = {lift_b:.4f}

Nilai *lift ratio* yang lebih besar dari 1 menunjukkan bahwa kedua produk tersebut memiliki 
kecenderungan kuat untuk dibeli secara bersamaan oleh pelanggan.
            """)


# ══════════════════════════════
# TAB 3: DATA TRANSAKSI
# ══════════════════════════════
with tab3:
    st.markdown(f"""
    <div class="section-header">
        <span class="section-title">Data Transaksi (Hasil Parsing)</span>
        <span class="section-count">{len(df_raw):,} baris</span>
        <div class="section-line"></div>
    </div>""", unsafe_allow_html=True)

    # Filter tanggal
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        search_prod = st.text_input("🔍 Cari nama produk", placeholder="Contoh: apollo, gudang garam…", key="search_raw")
    with col_f2:
        filter_faktur = st.text_input("🔍 Cari No. Faktur", placeholder="Contoh: R43-010525001", key="search_faktur")

    df_show = df_raw.copy()
    df_show["Tanggal"] = df_show["Tanggal"].dt.strftime("%d/%m/%Y")

    if search_prod:
        df_show = df_show[df_show["Nama_Barang"].str.contains(search_prod.upper(), na=False)]
    if filter_faktur:
        df_show = df_show[df_show["No_Faktur"].str.contains(filter_faktur.upper(), na=False)]

    display_cols = ["No_Faktur", "Tanggal", "Nama_Barang", "Satuan", "QTY", "Harga"]
    st.dataframe(df_show[display_cols], use_container_width=True, height=420)

    col_d1, _ = st.columns([3, 6])
    with col_d1:
        csv_raw = df_show[display_cols].to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "⬇️ Download Data Transaksi (CSV)",
            data=csv_raw,
            file_name="data_transaksi_bersih.csv",
            mime="text/csv",
            key="dl_raw_csv",
        )
