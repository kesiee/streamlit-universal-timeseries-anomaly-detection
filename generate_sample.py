import pandas as pd
import numpy as np

np.random.seed(42)
buildings = ["Building_A", "Building_B", "Building_C"]
dates = pd.date_range("2024-01-01", periods=365*24, freq="h")
rows = []
for b in buildings:
    base = {"Building_A": 120, "Building_B": 85, "Building_C": 200}[b]
    n = len(dates)
    noise      = np.random.randn(n) * 8
    hour_cycle = 20 * np.sin(2 * np.pi * dates.hour / 24)
    week_cycle = 10 * np.sin(2 * np.pi * dates.dayofweek / 7)
    energy     = base + hour_cycle + week_cycle + noise

    # inject anomalies
    anom_idx = np.random.choice(n, size=int(0.02*n), replace=False)
    signs    = np.random.choice([-1,1], size=len(anom_idx))
    mags     = np.random.uniform(40, 80, size=len(anom_idx))
    energy   = energy.copy()
    energy[anom_idx] += signs * mags

    for i, dt in enumerate(dates):
        rows.append({"timestamp": dt, "building_id": b, "energy_kwh": round(float(energy[i]),3)})

df = pd.DataFrame(rows)
df.to_csv("/home/claude/anomaly-detector/sample_data.csv", index=False)
print(f"Generated {len(df):,} rows — {df['building_id'].nunique()} buildings")
