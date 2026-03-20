# 🔍 Universal Time-Series Anomaly Detector
Streamlitt link: https://app-crypto-live-dashboard-shashank.streamlit.app/
Upload **any time-series dataset** in any format → get Bollinger Band anomaly detection per group.

## Features
- 📂 **Any format**: CSV, TSV, Excel, JSON, Parquet
- 🕐 **Any timestamp**: ISO 8601, Unix epoch, mm/dd/yyyy, mixed — auto-detected
- 🏷 **Group-by support**: runs BB independently per entity (building, sensor, ticker, etc.)
- 📊 **Adjustable**: rolling window (default 20) + std multiplier (default 1.5)
- ⬇ **Export**: download flagged rows or full results as CSV

## Stack
`streamlit` · `pandas` · `plotly` · `numpy`

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy
Push to GitHub → [share.streamlit.io](https://share.streamlit.io) → select repo → `app.py`

---
Built by **Shashank (KC)** · [Portfolio](https://portfolio-shashank-kammanahalli.vercel.app)
