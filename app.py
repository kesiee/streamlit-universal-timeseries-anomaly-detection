"""
Universal Time-Series Anomaly Detector
Upload any CSV / Excel / JSON / Parquet with any timestamp format.
Pick target, optional group-by columns → Bollinger Band anomaly detection per group.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io, os
from datetime import datetime

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Anomaly Detector · KC",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

  html, body, [class*="css"] { font-family: 'Sora', sans-serif; color: #1e293b; }
  .stApp { background: #f8fafc; }

  /* ── Step cards ── */
  .step-card {
    background: #fff; border: 1px solid #e2e8f0;
    border-radius: 14px; padding: 20px 22px; margin-bottom: 14px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
  }
  .step-num {
    display:inline-flex; align-items:center; justify-content:center;
    width:24px; height:24px; border-radius:50%;
    background: linear-gradient(135deg,#6366f1,#0ea5e9);
    color:#fff; font-size:11px; font-weight:600;
    margin-right:8px; vertical-align:middle;
  }
  .step-title { font-size:13px; font-weight:600; color:#0f172a; }

  /* ── Metric chips ── */
  .metric-row { display:flex; gap:12px; flex-wrap:wrap; margin: 12px 0; }
  .metric-chip {
    background:#fff; border:1px solid #e2e8f0; border-radius:10px;
    padding:10px 18px; min-width:130px;
    box-shadow:0 1px 3px rgba(0,0,0,0.04);
  }
  .metric-chip .label { font-size:10px; color:#94a3b8; font-family:'JetBrains Mono',monospace; letter-spacing:.06em; }
  .metric-chip .value { font-size:20px; font-weight:600; color:#0f172a; margin-top:2px; }
  .metric-chip .sub   { font-size:11px; color:#64748b; margin-top:1px; }
  .chip-red   .value  { color:#dc2626; }
  .chip-green .value  { color:#059669; }
  .chip-blue  .value  { color:#6366f1; }

  /* ── Section labels ── */
  .sec-label {
    font-family:'JetBrains Mono',monospace; font-size:9px;
    letter-spacing:.18em; color:#6366f1; text-transform:uppercase;
    margin:22px 0 10px; display:flex; align-items:center; gap:8px;
  }
  .sec-label::after { content:''; flex:1; height:1px; background:#e2e8f0; }

  /* ── Anomaly table ── */
  .anom-count {
    display:inline-block; background:#fef2f2; border:1px solid #fecaca;
    border-radius:20px; padding:2px 12px;
    font-family:'JetBrains Mono',monospace; font-size:11px; color:#dc2626;
  }

  /* ── Upload area ── */
  .upload-hint {
    background:#f0f9ff; border:1px dashed #7dd3fc; border-radius:10px;
    padding:14px 18px; font-size:12px; color:#0369a1; margin-bottom:10px;
  }

  /* ── Page title ── */
  .page-title { font-size:24px; font-weight:600; color:#0f172a; margin:0; letter-spacing:-.4px; }
  .page-sub   { color:#94a3b8; font-size:12px; margin-top:4px; font-family:'JetBrains Mono',monospace; }

  /* ── BB legend pill ── */
  .legend-pill {
    display:inline-flex; align-items:center; gap:5px;
    background:#fff; border:1px solid #e2e8f0; border-radius:20px;
    padding:3px 10px; font-size:11px; color:#475569; margin-right:6px;
  }
  .dot { width:8px;height:8px;border-radius:50%;display:inline-block; }

  #MainMenu,footer,header{visibility:hidden}
  .block-container{padding-top:1.2rem; max-width:100%;}
  [data-testid="stSidebar"]{background:#fff;border-right:1px solid #e2e8f0;}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
SAMPLE_PATH = os.path.join(os.path.dirname(__file__), "sample_data.csv")
SUPPORTED    = ["csv","xlsx","xls","json","parquet","tsv"]

# ── Helpers ───────────────────────────────────────────────────────────────────
def load_file(upload) -> pd.DataFrame:
    """Load any supported format into a DataFrame."""
    ext = upload.name.rsplit(".",1)[-1].lower()
    raw = upload.read()
    if ext == "csv":   return pd.read_csv(io.BytesIO(raw))
    if ext == "tsv":   return pd.read_csv(io.BytesIO(raw), sep="\t")
    if ext in ("xlsx","xls"): return pd.read_excel(io.BytesIO(raw))
    if ext == "json":
        try:    return pd.read_json(io.BytesIO(raw))
        except: return pd.json_normalize(__import__("json").loads(raw))
    if ext == "parquet": return pd.read_parquet(io.BytesIO(raw))
    raise ValueError(f"Unsupported format: .{ext}")

def smart_parse_timestamp(series: pd.Series) -> pd.Series:
    """Try multiple strategies to coerce a column to datetime."""
    # 1. Already datetime
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    # 2. Unix epoch (numeric)
    if pd.api.types.is_numeric_dtype(series):
        sample = series.dropna().iloc[0]
        if sample > 1e12:   return pd.to_datetime(series, unit="ms", errors="coerce")
        if sample > 1e9:    return pd.to_datetime(series, unit="s",  errors="coerce")
    # 3. String inference
    return pd.to_datetime(series, infer_datetime_format=True, errors="coerce")

def detect_ts_cols(df: pd.DataFrame) -> list:
    """Guess which columns look like timestamps."""
    candidates = []
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            candidates.append(col); continue
        if pd.api.types.is_numeric_dtype(df[col]):
            sample = df[col].dropna().iloc[0] if len(df[col].dropna()) else 0
            if 1e9 < sample < 2e10:
                candidates.append(col); continue
        if df[col].dtype == object:
            parsed = pd.to_datetime(df[col].head(20), errors="coerce", infer_datetime_format=True)
            if parsed.notna().sum() >= 15:
                candidates.append(col)
    return candidates

def bollinger_anomalies(series: pd.Series, window: int, std_mult: float):
    """Return (lower, mid, upper, is_anomaly) Series."""
    mid   = series.rolling(window, min_periods=1).mean()
    sigma = series.rolling(window, min_periods=1).std().fillna(0)
    lower = mid - std_mult * sigma
    upper = mid + std_mult * sigma
    is_anom = (series < lower) | (series > upper)
    return lower, mid, upper, is_anom

def fmt_pct(n, total): return f"{n/total*100:.1f}%" if total else "0%"

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔍 Anomaly Detector")
    st.caption("Universal time-series · Bollinger Bands")
    st.markdown("---")
    st.markdown("**⚙️ Detection Settings**")
    bb_window = st.slider("Rolling Window", 5, 100, 20, help="Periods for rolling mean/std")
    bb_std    = st.slider("Std Multiplier", 0.5, 4.0, 1.5, step=0.1,
                          help="1.5 = sensitive · 2.0 = standard · 3.0 = conservative")
    st.markdown("---")
    st.markdown("**📊 Chart Settings**")
    show_bands  = st.checkbox("Show BB bands",     value=True)
    show_mid    = st.checkbox("Show rolling mean", value=True)
    show_normal = st.checkbox("Show normal points",value=True)
    max_groups  = st.number_input("Max groups to process", 1, 50, 20,
                                  help="Cap for large cardinality group columns")
    st.markdown("---")
    st.caption("Built by Shashank (KC) · [Portfolio](https://portfolio-shashank-kammanahalli.vercel.app)")

# ── Header ────────────────────────────────────────────────────────────────────
h1, h2 = st.columns([5,1])
with h1:
    st.markdown('<p class="page-title">🔍 Time-Series Anomaly Detector</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-sub">Upload any format · any timestamp · Bollinger Band anomaly detection per group</p>',
                unsafe_allow_html=True)
st.markdown("---")

# ── Step 1 — Upload ───────────────────────────────────────────────────────────
st.markdown('<div class="sec-label">Step 1 · Load Data</div>', unsafe_allow_html=True)

col_up, col_sample = st.columns([3,1])
with col_up:
    st.markdown(
        '<div class="upload-hint">📂 Supports <b>CSV, TSV, Excel (.xlsx/.xls), JSON, Parquet</b> · '
        'Any timestamp format (ISO 8601, Unix epoch, mm/dd/yyyy, etc.)</div>',
        unsafe_allow_html=True)
    upload = st.file_uploader("Upload your dataset", type=SUPPORTED, label_visibility="collapsed")
with col_sample:
    st.markdown("<br>", unsafe_allow_html=True)
    use_sample = st.button("▶ Load Sample Data", use_container_width=True,
                           help="91-building smart energy dataset (sample)")

# Load data
df_raw = None
data_source = ""
if upload:
    try:
        df_raw = load_file(upload)
        data_source = upload.name
    except Exception as e:
        st.error(f"Could not read file: {e}")
elif use_sample or ("df_raw" in st.session_state and st.session_state.get("using_sample")):
    if use_sample:
        st.session_state["using_sample"] = True
    if os.path.exists(SAMPLE_PATH):
        df_raw = pd.read_csv(SAMPLE_PATH)
        data_source = "sample_data.csv (smart buildings)"
    else:
        st.error("Sample data not found.")
elif "df_configured" in st.session_state:
    df_raw = st.session_state["df_raw"]
    data_source = st.session_state.get("data_source","")

if df_raw is not None:
    st.session_state["df_raw"] = df_raw
    st.session_state["data_source"] = data_source
    st.success(f"✅ Loaded **{data_source}** — {len(df_raw):,} rows × {len(df_raw.columns)} columns")

    with st.expander("Preview raw data", expanded=False):
        st.dataframe(df_raw.head(10), use_container_width=True)

    # ── Step 2 — Configure columns ────────────────────────────────────────────
    st.markdown('<div class="sec-label">Step 2 · Configure Columns</div>', unsafe_allow_html=True)

    all_cols    = df_raw.columns.tolist()
    ts_guesses  = detect_ts_cols(df_raw)
    num_cols    = df_raw.select_dtypes(include="number").columns.tolist()
    cat_cols    = df_raw.select_dtypes(include=["object","category"]).columns.tolist()

    cfg1, cfg2, cfg3 = st.columns(3)
    with cfg1:
        ts_default = ts_guesses[0] if ts_guesses else all_cols[0]
        ts_col = st.selectbox("🕐 Timestamp column", all_cols,
                              index=all_cols.index(ts_default),
                              help="Auto-detected — change if wrong")
    with cfg2:
        num_default = num_cols[0] if num_cols else all_cols[0]
        target_col = st.selectbox("📈 Target (numeric)", all_cols,
                                  index=all_cols.index(num_default))
    with cfg3:
        group_cols = st.multiselect("🏷 Group-by columns (optional)",
                                    [c for c in cat_cols if c != ts_col],
                                    help="e.g. building_id, sensor_name, ticker")

    run_btn = st.button("🚀 Run Anomaly Detection", type="primary", use_container_width=False)

    if run_btn or st.session_state.get("df_configured"):
        if run_btn:
            st.session_state["df_configured"] = True
            st.session_state["ts_col"]     = ts_col
            st.session_state["target_col"] = target_col
            st.session_state["group_cols"] = group_cols
        else:
            ts_col     = st.session_state["ts_col"]
            target_col = st.session_state["target_col"]
            group_cols = st.session_state["group_cols"]

        # ── Parse & validate ──────────────────────────────────────────────────
        df = df_raw.copy()
        df[ts_col] = smart_parse_timestamp(df[ts_col])
        bad_ts = df[ts_col].isna().sum()
        if bad_ts:
            st.warning(f"⚠️ {bad_ts:,} rows had unparseable timestamps — they were dropped.")
        df = df.dropna(subset=[ts_col]).sort_values(ts_col).reset_index(drop=True)

        df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
        bad_num = df[target_col].isna().sum()
        if bad_num:
            st.warning(f"⚠️ {bad_num:,} rows had non-numeric target values — they were dropped.")
        df = df.dropna(subset=[target_col]).reset_index(drop=True)

        # ── Bollinger Band detection per group ────────────────────────────────
        if group_cols:
            groups = df.groupby(group_cols, sort=False)
            group_keys = list(groups.groups.keys())[:int(max_groups)]
        else:
            groups     = None
            group_keys = ["(all data)"]

        # Process all groups → build result df
        result_frames = []
        for gk in group_keys:
            if groups is not None:
                sub = groups.get_group(gk).copy()
                label = " | ".join([gk] if isinstance(gk, str) else [str(x) for x in gk])
            else:
                sub   = df.copy()
                label = "(all data)"
            sub = sub.sort_values(ts_col)
            lo, mid, hi, is_anom = bollinger_anomalies(sub[target_col], bb_window, bb_std)
            sub["_lower"]    = lo.values
            sub["_mid"]      = mid.values
            sub["_upper"]    = hi.values
            sub["_is_anom"]  = is_anom.values
            sub["_group_lbl"]= label
            result_frames.append(sub)

        df_result = pd.concat(result_frames, ignore_index=True)
        total_pts  = len(df_result)
        total_anom = df_result["_is_anom"].sum()
        anom_rate  = total_anom / total_pts * 100 if total_pts else 0

        # ── Step 3 — Summary metrics ──────────────────────────────────────────
        st.markdown('<div class="sec-label">Step 3 · Results</div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-row">
          <div class="metric-chip chip-blue">
            <div class="label">TOTAL POINTS</div>
            <div class="value">{total_pts:,}</div>
            <div class="sub">{len(group_keys)} group(s)</div>
          </div>
          <div class="metric-chip chip-red">
            <div class="label">ANOMALIES</div>
            <div class="value">{total_anom:,}</div>
            <div class="sub">{anom_rate:.1f}% of data</div>
          </div>
          <div class="metric-chip chip-green">
            <div class="label">NORMAL POINTS</div>
            <div class="value">{total_pts - total_anom:,}</div>
            <div class="sub">{100-anom_rate:.1f}% of data</div>
          </div>
          <div class="metric-chip">
            <div class="label">BB WINDOW</div>
            <div class="value">{bb_window}</div>
            <div class="sub">±{bb_std} std</div>
          </div>
          <div class="metric-chip">
            <div class="label">DATE RANGE</div>
            <div class="value" style="font-size:13px">{df_result[ts_col].min().strftime("%b %d %Y")}</div>
            <div class="sub">→ {df_result[ts_col].max().strftime("%b %d %Y")}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Step 4 — Chart ────────────────────────────────────────────────────
        st.markdown('<div class="sec-label">Step 4 · Visualise</div>', unsafe_allow_html=True)

        # Group selector
        if len(group_keys) > 1:
            display_labels = [
                " | ".join([gk] if isinstance(gk, str) else [str(x) for x in gk])
                for gk in group_keys
            ]
            selected_label = st.selectbox(
                "Select group to view",
                display_labels,
                format_func=lambda x: f"📂 {x}",
            )
        else:
            selected_label = group_keys[0] if group_cols else "(all data)"
            if not isinstance(selected_label, str):
                selected_label = " | ".join(str(x) for x in selected_label)

        # Filter to selected group
        plot_df = df_result[df_result["_group_lbl"] == selected_label].sort_values(ts_col)
        normal  = plot_df[~plot_df["_is_anom"]]
        anom    = plot_df[plot_df["_is_anom"]]
        g_anom_rate = len(anom)/len(plot_df)*100 if len(plot_df) else 0

        st.markdown(
            f'<span class="anom-count">⚡ {len(anom):,} anomalies in this group '
            f'({g_anom_rate:.1f}%)</span>',
            unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        fig = go.Figure()

        # BB fill
        if show_bands:
            fig.add_trace(go.Scatter(
                x=pd.concat([plot_df[ts_col], plot_df[ts_col][::-1]]),
                y=pd.concat([plot_df["_upper"], plot_df["_lower"][::-1]]),
                fill="toself", fillcolor="rgba(99,102,241,0.07)",
                line=dict(width=0), name="BB Band", showlegend=True,
                hoverinfo="skip",
            ))
            fig.add_trace(go.Scatter(
                x=plot_df[ts_col], y=plot_df["_upper"],
                mode="lines", name="Upper Band",
                line=dict(color="#a5b4fc", width=1, dash="dot"),
            ))
            fig.add_trace(go.Scatter(
                x=plot_df[ts_col], y=plot_df["_lower"],
                mode="lines", name="Lower Band",
                line=dict(color="#a5b4fc", width=1, dash="dot"),
            ))

        # Rolling mean
        if show_mid:
            fig.add_trace(go.Scatter(
                x=plot_df[ts_col], y=plot_df["_mid"],
                mode="lines", name=f"Mean ({bb_window})",
                line=dict(color="#6366f1", width=1.5),
            ))

        # Normal points
        if show_normal:
            fig.add_trace(go.Scatter(
                x=normal[ts_col], y=normal[target_col],
                mode="markers", name="Normal",
                marker=dict(color="#94a3b8", size=3, opacity=0.5),
            ))

        # Anomaly points — prominent
        if len(anom):
            fig.add_trace(go.Scatter(
                x=anom[ts_col], y=anom[target_col],
                mode="markers", name="⚡ Anomaly",
                marker=dict(
                    color="#dc2626", size=7,
                    symbol="circle",
                    line=dict(color="#ffffff", width=1.5),
                ),
                hovertemplate=(
                    "<b>ANOMALY</b><br>"
                    f"{ts_col}: %{{x}}<br>"
                    f"{target_col}: %{{y:.3f}}<br>"
                    "<extra></extra>"
                ),
            ))

        fig.update_layout(
            height=460,
            title=dict(
                text=f"<b>{selected_label}</b> — {target_col}",
                font=dict(family="Sora", size=14, color="#0f172a"),
                x=0,
            ),
            paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
            font=dict(family="JetBrains Mono, monospace", size=10, color="#94a3b8"),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02,
                bgcolor="rgba(0,0,0,0)", font=dict(size=10),
                bordercolor="#e2e8f0", borderwidth=1,
            ),
            margin=dict(l=10, r=10, t=50, b=10),
            hovermode="x unified",
            hoverlabel=dict(bgcolor="white", bordercolor="#e2e8f0",
                            font=dict(family="JetBrains Mono", size=11)),
            xaxis=dict(gridcolor="#f1f5f9", zerolinecolor="#f1f5f9",
                       showspikes=True, spikecolor="#cbd5e1",
                       spikedash="dot", spikethickness=1),
            yaxis=dict(gridcolor="#f1f5f9", zerolinecolor="#f1f5f9",
                       title=target_col, side="left"),
        )

        st.plotly_chart(fig, use_container_width=True,
                        config={"displayModeBar": True, "displaylogo": False,
                                "modeBarButtonsToRemove": ["lasso2d","select2d"]})

        # Per-group anomaly breakdown (if multiple groups)
        if len(group_keys) > 1:
            st.markdown('<div class="sec-label">Anomaly Breakdown by Group</div>',
                        unsafe_allow_html=True)
            summary_rows = []
            for lbl in display_labels:
                g = df_result[df_result["_group_lbl"] == lbl]
                na = g["_is_anom"].sum()
                summary_rows.append({
                    "Group"     : lbl,
                    "Points"    : len(g),
                    "Anomalies" : na,
                    "Rate %"    : f"{na/len(g)*100:.1f}%" if len(g) else "—",
                    "Min"       : f"{g[target_col].min():.3f}",
                    "Max"       : f"{g[target_col].max():.3f}",
                    "Mean"      : f"{g[target_col].mean():.3f}",
                })
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

        # ── Step 5 — Download ─────────────────────────────────────────────────
        st.markdown('<div class="sec-label">Step 5 · Export</div>', unsafe_allow_html=True)

        anom_only = df_result[df_result["_is_anom"]].drop(
            columns=["_lower","_mid","_upper","_is_anom","_group_lbl"])

        dl1, dl2 = st.columns(2)
        with dl1:
            csv_all = df_result.drop(
                columns=["_lower","_mid","_upper","_is_anom","_group_lbl"]
            ).to_csv(index=False).encode()
            st.download_button(
                "⬇ Download full results (with flags)",
                data=df_result.assign(
                    is_anomaly=df_result["_is_anom"]
                ).drop(columns=["_lower","_mid","_upper","_is_anom","_group_lbl"]
                ).to_csv(index=False).encode(),
                file_name="anomaly_results_all.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with dl2:
            st.download_button(
                "⬇ Download anomalies only",
                data=anom_only.to_csv(index=False).encode(),
                file_name="anomaly_results_flagged.csv",
                mime="text/csv",
                use_container_width=True,
            )

else:
    # Landing state
    st.markdown("""
    <div class="step-card">
      <span class="step-num">1</span><span class="step-title">Upload your dataset</span><br>
      <span style="font-size:12px;color:#64748b;margin-left:32px;">
        CSV · TSV · Excel · JSON · Parquet — any timestamp format
      </span>
    </div>
    <div class="step-card">
      <span class="step-num">2</span><span class="step-title">Pick timestamp, target, and optional group columns</span><br>
      <span style="font-size:12px;color:#64748b;margin-left:32px;">
        Auto-detects timestamp columns · handles Unix epoch, ISO 8601, mixed formats
      </span>
    </div>
    <div class="step-card">
      <span class="step-num">3</span><span class="step-title">Bollinger Bands run per group</span><br>
      <span style="font-size:12px;color:#64748b;margin-left:32px;">
        Rolling mean ± 1.5σ (adjustable) · anomalies flagged and visualised
      </span>
    </div>
    <div class="step-card">
      <span class="step-num">4</span><span class="step-title">Export flagged rows as CSV</span><br>
      <span style="font-size:12px;color:#64748b;margin-left:32px;">
        Download full results or anomalies-only
      </span>
    </div>
    """, unsafe_allow_html=True)
    st.info("👆 Upload a file above or click **▶ Load Sample Data** to try with a smart buildings energy dataset.")

