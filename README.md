# HTM-Monitor

System anomalies fire only when multiple grouped models agree **and** remain anomalous over time.

## Demo

### May 2020 System Anomaly
https://github.com/user-attachments/assets/https://github.com/gotham29/htm-monitor/blob/master/assets/powergrid_ca_may2020.mp4

---

### Aug 2020 System Anomaly
https://github.com/user-attachments/assets/https://github.com/gotham29/htm-monitor/blob/master/assets/powergrid_ca_aug2020.mp4

---

### Sept 2022 System Anomaly
https://github.com/user-attachments/assets/https://github.com/gotham29/htm-monitor/blob/master/assets/powergrid_ca_sept2022.mp4

---

## Problem
Real-time anomaly detection produces too many false positives to be actionable.

## Approach
- one HTM model per signal
- related signals grouped together
- system anomaly requires **agreement + persistence**

## Results

- **Precision:** 0.67  
- **Recall:** 1.00  

The single false positive coincides with a large-scale demand and generation shift during early COVID-19.

## Quickstart

```bash
# 1. Generate a usecase config
python -m htm_monitor.cli.usecase_wizard \
  --out-dir configs/generated \
  --spec-out specs/powergrid_ca.build.yaml

# 2. Run pipeline (no plot)
python -m htm_monitor.cli.run_pipeline \
  --defaults configs/htm_defaults.yaml \
  --config configs/generated/powergrid_ca.yaml \
  --run-dir outputs/powergrid_ca_run \
  --no-plot

# 3. Analyze results
python -m htm_monitor.cli.analyze_run \
  --run-dir outputs/powergrid_ca_run \
  --config configs/generated/powergrid_ca.yaml
```

## Why it matters

Agreement + persistence turns anomaly detection from noisy output into actionable system alerts.

