"""
Analisis FP-Growth — Universal
Mendukung 2 format CSV:
  • Format Laporan Kasir (struk kasir Toko Yunita dll)
  • Format CSV Biasa (kolom: id_transaksi/tanggal + nama_produk/nama_barang)
"""

import streamlit as st
import pandas as pd
import numpy as np
import re, csv, io, warnings
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Analisis FP-Growth",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
.stApp { background: #0f1117; }
[data-testid="stSidebar"] { background: #161b27 !important; border-right: 1px solid #1e2535; }
[data-testid="stSidebar"] * { color: #c8d0e0 !important; }
.hero-banner {
    background: linear-gradient(135deg, #0d1b2a 0%, #1a2744 50%, #0d1b2a 100%);
    border: 1px solid #1e3a5f; border-radius: 16px;
    padding: 28px 36px; margin-bottom: 24px; position: relative; overflow: hidden;
}
.hero-banner::before {
    content:''; position:absolute; top:-60px; right:-60px; width:200px; height:200px;
    background:radial-gradient(circle,rgba(59,130,246,.15) 0%,transparent 70%); border-radius:50%;
}
.hero-title { font-size:26px; font-weight:700; color:#f0f4ff; margin:0 0 4px; letter-spacing:-.5px; }
.hero-sub   { font-size:13px; color:#6b7fa3; margin:0; }
.hero-badge {
    display:inline-block; background:#1e3a5f; color:#60a5fa; border:1px solid #2d5a8e;
    border-radius:20px; padding:3px 12px; font-size:10px; font-weight:600;
    letter-spacing:.5px; margin-bottom:10px; text-transform:uppercase;
}
.metric-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-bottom:24px; }
.metric-card {
    background:#161b27; border:1px solid #1e2535; border-radius:12px; padding:18px; transition:border-color .2s;
}
.metric-card:hover { border-color:#2d4a7a; }
.metric-label { font-size:10px; font-weight:600; color:#4a5568; text-transform:uppercase; letter-spacing:.8px; margin-bottom:6px; }
.metric-value { font-size:26px; font-weight:700; color:#f0f4ff; font-family:'JetBrains Mono',monospace; line-height:1; }
.metric-sub   { font-size:11px; color:#4a5568; margin-top:3px; }
.metric-card.blue  { border-left:3px solid #3b82f6; }
.metric-card.green { border-left:3px solid #10b981; }
.metric-card.amber { border-left:3px solid #f59e0b; }
.metric-card.coral { border-left:3px solid #ef4444; }
.section-header { display:flex; align-items:center; gap:10px; margin:24px 0 14px; }
.section-title  { font-size:15px; font-weight:600; color:#d1d9ef; margin:0; }
.section-line   { flex:1; height:1px; background:#1e2535; }
.section-count  {
    font-size:10px; font-weight:600; background:#1e2d4a; color:#60a5fa;
    border-radius:20px; padding:2px 10px; font-family:'JetBrains Mono',monospace;
}
.success-bar {
    background:#052e16; border:1px solid #166534; border-radius:10px;
    padding:11px 16px; color:#4ade80; font-size:13px; font-weight:500;
    display:flex; align-items:center; gap:10px; margin-bottom:16px; flex-wrap:wrap;
}
.format-badge { display:inline-block; border-radius:6px; padding:2px 10px; font-size:11px; font-weight:600; font-family:'JetBrains Mono',monospace; }
.format-kasir { background:#1e3a5f; color:#60a5fa; }
.format-biasa { background:#1a3028; color:#34d399; }
.rule-card {
    background:#161b27; border:1px solid #1e2535; border-radius:10px;
    padding:13px 16px; margin-bottom:7px; display:flex; align-items:center;
    gap:10px; flex-wrap:wrap; font-size:13px;
}
.rule-arrow { color:#3b82f6; font-size:16px; font-weight:700; }
.rule-item  { color:#d1d9ef; font-weight:500; }
.rule-badges { margin-left:auto; display:flex; gap:7px; flex-wrap:wrap; }
.badge { border-radius:6px; padding:2px 8px; font-size:10px; font-weight:600; font-family:'JetBrains Mono',monospace; }
.badge-sup  { background:#1e3a5f; color:#60a5fa; }
.badge-conf { background:#1a3028; color:#34d399; }
.badge-lift { background:#3a2010; color:#fb923c; }
.step-box { background:#161b27; border:1px solid #1e2535; border-radius:12px; padding:18px; text-align:center; }
.step-icon { font-size:26px; margin-bottom:8px; }
.step-text { color:#4a5568; font-size:11px; margin-top:3px; }
.info-box {
    background:#0c1929; border:1px solid #1e3a5f; border-radius:10px;
    padding:14px 18px; font-size:12px; color:#6b9fd4; margin-bottom:16px; line-height:1.8;
}
.info-box strong { color:#93c5fd; }
[data-testid="stFileUploader"] { background:#161b27 !important; border:2px dashed #1e3a5f !important; border-radius:12px !important; }
.stSlider > div > div > div > div { background:#3b82f6 !important; }
.stDownloadButton > button {
    background:#1e3a5f !important; color:#60a5fa !important; border:1px solid #2d5a8e !important;
    border-radius:8px !important; font-family:'Plus Jakarta Sans',sans-serif !important;
    font-weight:600 !important; font-size:12px !important; padding:7px 14px !important;
}
.stDownloadButton > button:hover { background:#2d5a8e !important; border-color:#3b82f6 !important; }
.stButton > button {
    background:#1d4ed8 !important; color:#fff !important; border:none !important;
    border-radius:8px !important; font-family:'Plus Jakarta Sans',sans-serif !important;
    font-weight:600 !important; padding:10px 24px !important; width:100% !important;
}
.stButton > button:hover { background:#1e40af !important; }
.stTabs [data-baseweb="tab-list"] { background:transparent !important; border-bottom:1px solid #1e2535 !important; }
.stTabs [data-baseweb="tab"] {
    background:transparent !important; color:#4a5568 !important;
    font-family:'Plus Jakarta Sans',sans-serif !important; font-weight:500 !important;
    font-size:13px !important; padding:10px 18px !important; border:none !important;
}
.stTabs [aria-selected="true"] { color:#60a5fa !important; border-bottom:2px solid #3b82f6 !important; }
.dataframe thead tr th {
    background:#1a2133 !important; color:#6b7fa3 !important; font-size:10px !important;
    font-weight:600 !important; text-transform:uppercase !important; letter-spacing:.5px !important;
    padding:9px 12px !important; border-bottom:1px solid #1e2535 !important;
}
.dataframe tbody tr { background:#161b27 !important; }
.dataframe tbody tr:nth-child(even) { background:#131720 !important; }
.dataframe tbody tr:hover { background:#1c2340 !important; }
.dataframe tbody tr td {
    color:#c8d0e0 !important; padding:8px 12px !important;
    border-bottom:1px solid #1a2030 !important; border-top:none !important;
    border-left:none !important; border-right:none !important;
}
::-webkit-scrollbar { width:5px; height:5px; }
::-webkit-scrollbar-track { background:#0f1117; }
::-webkit-scrollbar-thumb { background:#1e2535; border-radius:3px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
SATUAN_KASIR = {
    'PCS','PACK','BH','BKS','RCG','BTL','KG',
    'GR','LTR','DOS','SGT','RLL','LSN','CTN'
}
ALIAS_ID     = {'id_transaksi','no_faktur','faktur','transaction_id','no_struk','kode_transaksi','invoice','nota','id','no'}
ALIAS_TANGGAL= {'tanggal','date','tgl','transaction_date','waktu'}
ALIAS_PRODUK = {'nama_barang','nama_produk','produk','barang','item','name','product','product_name','item_name','keterangan'}


# ─────────────────────────────────────────────────────────────
# DETECT FORMAT
# ─────────────────────────────────────────────────────────────
def detect_format(file_bytes: bytes) -> str:
    try:
        teks  = file_bytes.decode("latin1")
        lines = [l for l in teks.splitlines() if l.strip()]
        if not lines:
            return "biasa"
        # Scan 40 baris pertama (file kasir sering dimulai dengan header toko)
        for line in lines[:40]:
            reader  = csv.reader(io.StringIO(line))
            row     = next(reader, [])
            cleaned = [v.strip() for v in row if v.strip()]
            if not cleaned:
                continue
            # Tanda 1: baris transaksi (No. urut kecil + tanggal dd/mm/yyyy)
            if (re.match(r"^\d{1,4}$", cleaned[0])
                    and len(cleaned) > 2
                    and re.match(r"^\d+/\d+/\d+$", cleaned[1])):
                return "kasir"
            # Tanda 2: baris item dengan satuan
            if any(s in cleaned for s in SATUAN_KASIR):
                return "kasir"
        return "biasa"
    except Exception:
        return "biasa"


# ─────────────────────────────────────────────────────────────
# PARSE KASIR
# ─────────────────────────────────────────────────────────────
def parse_kasir(file_bytes: bytes) -> pd.DataFrame:
    teks   = file_bytes.decode("latin1")
    reader = csv.reader(io.StringIO(teks))
    rows   = list(reader)
    current_faktur = current_tanggal = None
    hasil = []
    for row in rows:
        b = [v.strip() for v in row if v.strip() != ""]
        if not b:
            continue
        is_txn = (
            re.match(r"^\d{1,4}$", b[0]) and len(b) > 2
            and re.match(r"^\d+/\d+/\d+$", b[1])
        )
        is_item = any(s in b for s in SATUAN_KASIR)
        if is_txn:
            current_tanggal = b[1]
            current_faktur  = b[2]
        elif is_item and current_faktur:
            for idx, val in enumerate(b):
                if val in SATUAN_KASIR and idx >= 1:
                    nama  = b[idx - 1]
                    qty   = b[idx + 1] if idx + 1 < len(b) else "1"
                    harga = b[idx + 2] if idx + 2 < len(b) else "0"
                    if nama and not re.match(r"^\d+$", nama):
                        hasil.append({
                            "No_Faktur"  : current_faktur,
                            "Tanggal"    : current_tanggal,
                            "Nama_Barang": nama.upper().strip(),
                            "Satuan": val, "QTY": qty, "Harga": harga,
                        })
                    break
    if not hasil:
        return pd.DataFrame()
    df = pd.DataFrame(hasil)
    df["Nama_Barang"] = df["Nama_Barang"].str.strip().str.replace(r"\s+", " ", regex=True)
    df["QTY"]         = pd.to_numeric(df["QTY"], errors="coerce").fillna(1).astype(int)
    df["Tanggal"]     = pd.to_datetime(df["Tanggal"], format="%m/%d/%Y", errors="coerce")
    df = df.dropna(subset=["No_Faktur", "Nama_Barang"])
    df = df[df["Nama_Barang"] != ""]
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────
# PARSE CSV BIASA (otomatis deteksi kolom)
# ─────────────────────────────────────────────────────────────
def parse_biasa(file_bytes: bytes) -> tuple:
    for enc in ["utf-8-sig", "utf-8", "latin1"]:
        try:
            teks = file_bytes.decode(enc)
            break
        except Exception:
            continue
    df = pd.read_csv(io.StringIO(teks), dtype=str)
    df.columns = df.columns.str.strip()
    cols_lower = {c.lower().replace(" ", "_"): c for c in df.columns}

    # Cari kolom produk
    kolom_produk = None
    for alias in ALIAS_PRODUK:
        if alias in cols_lower:
            kolom_produk = cols_lower[alias]; break
    if kolom_produk is None:
        str_cols = df.select_dtypes(include="object").columns.tolist()
        if str_cols:
            kolom_produk = max(str_cols, key=lambda c: df[c].str.len().mean())

    # Cari kolom ID
    kolom_id, mode = None, "lainnya"
    for alias in ALIAS_ID:
        if alias in cols_lower:
            kolom_id = cols_lower[alias]; mode = "id"; break
    if kolom_id is None:
        for alias in ALIAS_TANGGAL:
            if alias in cols_lower:
                kolom_id = cols_lower[alias]; mode = "tanggal"; break
    if kolom_id is None:
        kolom_id = df.columns[0]; mode = "lainnya"

    df_out = df[[kolom_id, kolom_produk]].copy().dropna()
    df_out.columns = ["No_Faktur", "Nama_Barang"]
    df_out["Nama_Barang"] = df_out["Nama_Barang"].str.upper().str.strip()
    df_out["Tanggal"] = None
    df_out["Satuan"]  = "-"
    df_out["QTY"]     = 1
    df_out["Harga"]   = "0"
    df_out = df_out[df_out["Nama_Barang"] != ""]
    return df_out.reset_index(drop=True), kolom_id, kolom_produk, mode


# ─────────────────────────────────────────────────────────────
# MASTER PARSER
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, max_entries=5)
def parse_file(file_bytes: bytes, filename: str) -> tuple:
    fmt = detect_format(file_bytes)
    if fmt == "kasir":
        df   = parse_kasir(file_bytes)
        info = "Format Laporan Kasir terdeteksi — parsing otomatis berhasil."
        return df, "kasir", info
    else:
        df, kolom_id, kolom_produk, mode = parse_biasa(file_bytes)
        info = f"Format CSV Biasa — ID: **{kolom_id}** · Produk: **{kolom_produk}** · Mode: **{mode}**"
        return df, "biasa", info


# ─────────────────────────────────────────────────────────────
# FP-GROWTH
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, max_entries=10)
def run_fpgrowth(df_hash: str, _df: pd.DataFrame,
                 min_support: float, min_confidence: float) -> tuple:
    basket = (
        _df.groupby("No_Faktur")["Nama_Barang"]
        .apply(list).reset_index(name="Produk")
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
    fi = fi.sort_values(["jumlah_item","support"], ascending=[True,False]).reset_index(drop=True)
    fi.index += 1
    rules = association_rules(fi, metric="confidence", min_threshold=min_confidence)
    if rules.empty:
        return fi, pd.DataFrame(), basket
    rules["antecedents_str"] = rules["antecedents"].apply(lambda x: " + ".join(sorted(x)))
    rules["consequents_str"] = rules["consequents"].apply(lambda x: " + ".join(sorted(x)))
    for col in ["support","confidence","lift"]:
        rules[col] = rules[col].round(4)
    rules = rules.sort_values("lift", ascending=False).reset_index(drop=True)
    rules.index += 1
    return fi, rules, basket


# ─────────────────────────────────────────────────────────────
# EXPORT EXCEL
# ─────────────────────────────────────────────────────────────
def to_excel(fi: pd.DataFrame, rules: pd.DataFrame, df_raw: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        tfi = fi[["itemsets_str","support","support_pct","jumlah_item"]].rename(columns={
            "itemsets_str":"Itemset","support":"Support (Desimal)",
            "support_pct":"Support (%)","jumlah_item":"Jumlah Item"})
        tfi.to_excel(writer, sheet_name="Frequent Itemset", index=True)
        if not rules.empty:
            tr = rules[["antecedents_str","consequents_str","support","confidence","lift"]].rename(columns={
                "antecedents_str":"Jika Membeli","consequents_str":"Maka Membeli",
                "support":"Support","confidence":"Confidence","lift":"Lift Ratio"})
            tr.to_excel(writer, sheet_name="Aturan Asosiasi", index=True)
        dr = df_raw[["No_Faktur","Tanggal","Nama_Barang","QTY","Harga"]].copy()
        if pd.api.types.is_datetime64_any_dtype(dr.get("Tanggal",pd.Series())):
            dr["Tanggal"] = dr["Tanggal"].dt.strftime("%d/%m/%Y")
        dr.to_excel(writer, sheet_name="Data Transaksi", index=False)
    return buf.getvalue()


def metric_card(label, value, sub, cls):
    return (f'<div class="metric-card {cls}">'
            f'<div class="metric-label">{label}</div>'
            f'<div class="metric-value">{value}</div>'
            f'<div class="metric-sub">{sub}</div></div>')


# ═════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ Pengaturan")
    st.markdown("---")
    st.markdown("**📁 Upload File CSV**")
    uploaded = st.file_uploader(
        "Pilih file laporan atau data transaksi", type=["csv"],
        help="Mendukung laporan kasir (struk) & CSV biasa dengan header",
    )
    st.markdown("---")
    st.markdown("**🎚️ Parameter FP-Growth**")
   # UNTUK ATUR MINIMUM SUPPORT DAN CONFIDENCE
support_pct    = st.slider("Minimum Support (%)", 
                            min_value=0.1, max_value=10.0, 
                            value=0.5, step=0.1,
                            help="Untuk 6.000+ transaksi, coba 0.1%–1%")
min_support    = support_pct / 100

min_confidence = st.slider("Minimum Confidence (%)", 
                            10, 80, 20, 5,
                            help="Disarankan 20%–40% untuk skripsi") / 100)
    st.markdown("---")
    st.markdown("**🔍 Filter**")
    filter_itemset = st.selectbox("Tampilkan itemset",
                                  ["Semua","1-itemset saja","2-itemset saja","3-itemset ke atas"])
    min_lift = st.slider("Lift minimum", 1.0, 10.0, 1.0, 0.5)
    st.markdown("---")
    with st.expander("📋 Format CSV yang didukung"):
        st.markdown("""
**Format 1 — Laporan Kasir**
File report dari mesin kasir. Tidak perlu diubah.

**Format 2 — CSV Biasa**
Contoh isi file:
```
id_transaksi,nama_produk
T001,Rinso 1kg
T001,Softener Molto
T002,Sabun Mandi
```
Nama kolom dikenali otomatis:
- ID: `id_transaksi`, `no_faktur`, `invoice`
- Produk: `nama_barang`, `nama_produk`, `item`
- Tanggal: `tanggal`, `date`, `tgl`
        """)
    st.caption("🛒 Analisis FP-Growth · Universal")


# ═════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-banner">
    <div class="hero-badge">🛒 Data Mining · FP-Growth</div>
    <div class="hero-title">Analisis Pola Pembelian</div>
    <p class="hero-sub">Upload CSV laporan kasir atau data transaksi → analisis otomatis → download Excel</p>
</div>
""", unsafe_allow_html=True)

if uploaded is None:
    c1, c2, c3 = st.columns(3)
    for col, icon, title, desc in [
        (c1, "📤", "Upload File CSV", "Sidebar kiri — mendukung laporan kasir & CSV biasa"),
        (c2, "⚙️", "Atur Parameter",  "Geser slider support & confidence sesuai kebutuhan"),
        (c3, "📊", "Lihat Hasil",      "Tabel & aturan asosiasi muncul otomatis"),
    ]:
        with col:
            st.markdown(f'<div class="step-box"><div class="step-icon">{icon}</div>'
                        f'<div style="color:#c8d0e0;font-weight:600;font-size:13px">{title}</div>'
                        f'<div class="step-text">{desc}</div></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""<div class="info-box">
    💡 <strong>App ini mendukung 2 format CSV:</strong><br>
    &nbsp;&nbsp;① <strong>Laporan Kasir</strong> — file report dari mesin kasir (seperti Toko Yunita)<br>
    &nbsp;&nbsp;② <strong>CSV Biasa</strong> — file dengan kolom header seperti <code>id_transaksi, nama_produk</code><br>
    &nbsp;&nbsp;Format dideteksi <strong>otomatis</strong> — tidak perlu setting apapun.
    </div>""", unsafe_allow_html=True)
    st.info("⬅️  Upload file CSV di sidebar untuk memulai analisis.", icon="💡")
    st.stop()

# ── Parse file
with st.spinner("📂 Membaca file…"):
    file_bytes = uploaded.read()
    df_raw, fmt, info_str = parse_file(file_bytes, uploaded.name)

if df_raw.empty:
    st.error("❌ File tidak dapat dibaca. Pastikan format CSV sesuai (lihat panduan di sidebar).")
    st.stop()

# ── Status bar
fmt_label, fmt_cls = ("Laporan Kasir","format-kasir") if fmt=="kasir" else ("CSV Biasa","format-biasa")
n_txn  = df_raw["No_Faktur"].nunique()
n_prod = df_raw["Nama_Barang"].nunique()
n_item = len(df_raw)
has_date = (pd.api.types.is_datetime64_any_dtype(df_raw.get("Tanggal", pd.Series()))
            and not df_raw["Tanggal"].isna().all())
period = (f"{df_raw['Tanggal'].min().strftime('%d %b %Y')} – {df_raw['Tanggal'].max().strftime('%d %b %Y')}"
          if has_date else "-")

st.markdown(f"""<div class="success-bar">
    ✅ &nbsp;
    <span class="format-badge {fmt_cls}">{fmt_label}</span>
    &nbsp;·&nbsp; <strong>{n_txn:,}</strong> transaksi
    &nbsp;·&nbsp; <strong>{n_prod:,}</strong> produk unik
    &nbsp;·&nbsp; <strong>{n_item:,}</strong> item
    {f"&nbsp;·&nbsp; {period}" if has_date else ""}
</div>""", unsafe_allow_html=True)

# ── Run FP-Growth
with st.spinner("🔄 Menjalankan FP-Growth…"):
    df_hash = f"{uploaded.name}_{n_item}_{min_support}_{min_confidence}"
    fi, rules, basket = run_fpgrowth(df_hash, df_raw, min_support, min_confidence)

avg_item = basket["Produk"].apply(len).mean()
st.markdown(f"""<div class="metric-grid">
    {metric_card("Total Transaksi",    f"{n_txn:,}",      f"Periode: {period}", "blue")}
    {metric_card("Frequent Itemset", f"{len(fi):,}", 
            f"Min. support {min_support*100:.1f}%", "green")
    {metric_card("Aturan Asosiasi",    f"{len(rules):,}",  f"Min. confidence {min_confidence*100:.0f}%", "amber")}
    {metric_card("Avg Item/Transaksi", f"{avg_item:.1f}",  f"{n_prod:,} produk unik", "coral")}
</div>""", unsafe_allow_html=True)

if fi.empty:
    st.warning(f"⚠️ Tidak ada frequent itemset dengan support {min_support*100:.0f}%. "
               "Coba kurangi nilai minimum support di sidebar (misal ke 1%).")
    st.stop()

# ── Tabs
tab1, tab2, tab3 = st.tabs(["📦  Frequent Itemset","📋  Aturan Asosiasi","🗃️  Data Transaksi"])


# ════════ TAB 1 ════════
with tab1:
    fi_disp = fi.copy()
    if filter_itemset == "1-itemset saja":
        fi_disp = fi_disp[fi_disp["jumlah_item"] == 1]
    elif filter_itemset == "2-itemset saja":
        fi_disp = fi_disp[fi_disp["jumlah_item"] == 2]
    elif filter_itemset == "3-itemset ke atas":
        fi_disp = fi_disp[fi_disp["jumlah_item"] >= 3]

    cari = st.text_input("🔍 Cari nama barang", placeholder="Contoh: rinso, sampoerna…", key="s_fi")
    if cari:
        fi_disp = fi_disp[fi_disp["itemsets_str"].str.contains(cari.upper(), na=False)]

    st.markdown(f"""<div class="section-header">
        <span class="section-title">Frequent Itemset</span>
        <span class="section-count">{len(fi_disp)} itemset</span>
        <div class="section-line"></div>
    </div>""", unsafe_allow_html=True)

    tfi = fi_disp[["itemsets_str","support_pct","support","jumlah_item"]].rename(columns={
        "itemsets_str":"Itemset (Nama Barang)","support_pct":"Support (%)","support":"Support (Desimal)","jumlah_item":"Jumlah Item"})
    tfi["Support (%)"] = tfi["Support (%)"].apply(lambda x: f"{x:.2f}%")
    st.dataframe(tfi, use_container_width=True, height=400)

    ca, cb, _ = st.columns([2,2,5])
    with ca:
        st.download_button("⬇️ CSV", tfi.to_csv(index=True).encode("utf-8-sig"),
                           "frequent_itemset.csv", "text/csv", key="dl_fi")
    with cb:
        st.download_button("⬇️ Excel (Semua)", to_excel(fi, rules, df_raw),
                           "Hasil_FP_Growth.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="dl_xl1")
    st.markdown("---")
    st.markdown("**📊 Top 10 Produk Paling Sering Muncul**")
    top10 = fi[fi["jumlah_item"]==1].head(10)[["itemsets_str","support_pct"]].copy()
    top10.columns = ["Nama Barang","Support (%)"]
    if not top10.empty:
        st.bar_chart(top10.set_index("Nama Barang"), height=240)


# ════════ TAB 2 ════════
with tab2:
    if rules.empty:
        st.warning(f"⚠️ Tidak ada aturan dengan confidence {min_confidence*100:.0f}%. Coba kurangi nilai di sidebar.")
    else:
        rd = rules[rules["lift"] >= min_lift].copy()
        cari2 = st.text_input("🔍 Cari dalam aturan", placeholder="Contoh: rinso, apollo…", key="s_r")
        if cari2:
            k = cari2.upper()
            rd = rd[rd["antecedents_str"].str.contains(k,na=False)|rd["consequents_str"].str.contains(k,na=False)]

        st.markdown(f"""<div class="section-header">
            <span class="section-title">Aturan Asosiasi</span>
            <span class="section-count">{len(rd)} aturan</span>
            <div class="section-line"></div>
        </div>""", unsafe_allow_html=True)

        tr = rd[["antecedents_str","consequents_str","support","confidence","lift"]].rename(columns={
            "antecedents_str":"Jika Membeli","consequents_str":"Maka Membeli",
            "support":"Support","confidence":"Confidence","lift":"Lift Ratio"})
        tr["Support"]    = tr["Support"].apply(lambda x: f"{x:.4f}")
        tr["Confidence"] = tr["Confidence"].apply(lambda x: f"{x*100:.1f}%")
        tr["Lift Ratio"] = tr["Lift Ratio"].apply(lambda x: f"{x:.4f}")
        st.dataframe(tr, use_container_width=True, height=360)

        ca2, cb2, _ = st.columns([2,2,5])
        with ca2:
            st.download_button("⬇️ CSV", tr.to_csv(index=True).encode("utf-8-sig"),
                               "aturan_asosiasi.csv","text/csv",key="dl_r")
        with cb2:
            st.download_button("⬇️ Excel (Semua)", to_excel(fi,rules,df_raw),
                               "Hasil_FP_Growth.xlsx",
                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",key="dl_xl2")

        st.markdown("---")
        st.markdown("**✨ Top 5 Aturan Terkuat (Lift Tertinggi)**")
        for _, row in rd.head(5).iterrows():
            lift_v = float(row["lift"])
            icon   = "🔥" if lift_v>=2 else ("⚡" if lift_v>=1.5 else "·")
            st.markdown(f"""<div class="rule-card">
                <span class="rule-item">{row['antecedents_str']}</span>
                <span class="rule-arrow">→</span>
                <span class="rule-item">{row['consequents_str']}</span>
                <div class="rule-badges">
                    <span class="badge badge-sup">Sup {float(row['support'])*100:.1f}%</span>
                    <span class="badge badge-conf">Conf {float(row['confidence'])*100:.1f}%</span>
                    <span class="badge badge-lift">{icon} Lift {lift_v:.2f}</span>
                </div>
            </div>""", unsafe_allow_html=True)

        with st.expander("📝 Ringkasan untuk Pemilik Toko"):
            best = rd.iloc[0]
            sup_b, conf_b, lift_b = float(best["support"]), float(best["confidence"]), float(best["lift"])
            st.markdown(f"""
Dataset: **{n_txn:,} transaksi** · **{n_prod:,} produk**{f" · {period}" if has_date else ""}

FP-Growth (support **{min_support*100:.0f}%**, confidence **{min_confidence*100:.0f}%**):
→ **{len(fi)} frequent itemset** dan **{len(rules)} aturan asosiasi**

Aturan lift tertinggi:
> **{best['antecedents_str']} → {best['consequents_str']}**
> Support = {sup_b:.4f} ({sup_b*100:.2f}%) · Confidence = {conf_b:.4f} ({conf_b*100:.2f}%) · Lift = {lift_b:.4f}

Lift > 1 menunjukkan kedua produk cenderung dibeli bersamaan.
            """)


# ════════ TAB 3 ════════
with tab3:
    st.markdown(f"""<div class="section-header">
        <span class="section-title">Data Transaksi (Hasil Parsing)</span>
        <span class="section-count">{len(df_raw):,} baris</span>
        <div class="section-line"></div>
    </div>""", unsafe_allow_html=True)

    cc1, cc2 = st.columns(2)
    with cc1:
        sp = st.text_input("🔍 Cari produk", placeholder="Contoh: apollo…", key="s_raw")
    with cc2:
        sf = st.text_input("🔍 Cari No. Faktur / ID", placeholder="Contoh: R43-01…", key="s_fak")

    ds = df_raw.copy()
    if pd.api.types.is_datetime64_any_dtype(ds.get("Tanggal", pd.Series())):
        ds["Tanggal"] = ds["Tanggal"].dt.strftime("%d/%m/%Y")
    if sp: ds = ds[ds["Nama_Barang"].str.contains(sp.upper(), na=False)]
    if sf: ds = ds[ds["No_Faktur"].str.contains(sf.upper(), na=False)]

    show_cols = [c for c in ["No_Faktur","Tanggal","Nama_Barang","Satuan","QTY","Harga"] if c in ds.columns]
    st.dataframe(ds[show_cols], use_container_width=True, height=400)
    cd, _ = st.columns([3,6])
    with cd:
        st.download_button("⬇️ Download Data Bersih (CSV)",
                           ds[show_cols].to_csv(index=False).encode("utf-8-sig"),
                           "data_transaksi_bersih.csv","text/csv",key="dl_raw")
