# HTM-Monitor

**signals → models → groups → temporal filter → system anomaly**

Detects only **3 system anomalies across years of continuous grid data**

## Problem
Real-time anomaly detection produces too many false positives to be actionable.

In continuous, multi-signal systems, transient spikes frequently trigger alerts that do not correspond to meaningful events.

## Approach
HTM-Monitor uses **sustained consensus across grouped models**:

- Each signal is modeled independently (HTM)
- Related signals are grouped (e.g., imbalance signals)
- A system anomaly is triggered only when:
  - multiple models are anomalous **at the same time**, and  
  - the condition **persists over time**

This suppresses isolated spikes and requires **coherent, sustained system behavior**.

> **System anomalies are triggered only when multiple models agree and remain anomalous over time.**

## Demo

### May 2020 System Anomaly
Coordinated deviation across imbalance signals produces sustained anomaly scores and triggers a system anomaly.

<video src="assets/Grid_Alert_May2020_final.mp4" controls width="700"></video>

---

### Aug 2020 System Anomaly
Similar signal behavior occurs, but without sustained consensus across models, no system anomaly is triggered.

<video src="assets/Grid_Alert_Aug2020_final.mp4" controls width="700"></video>

---

### Sept 2022 System Anomaly
A later event again shows coordinated, sustained deviation across signals, resulting in a system anomaly.

<video src="assets/Grid_Alert_Sept2022_final.mp4" controls width="700"></video>

## Results

- **Precision:** 0.67  
- **Recall:** 1.00  

Across years of continuous data, only a small number of system anomalies are detected, while transient spikes are ignored.

The single false positive coincides with a large-scale demand and generation shift during early COVID-19, suggesting sensitivity to real structural changes in the grid.

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

Reducing false positives transforms anomaly detection from a noisy signal into an actionable system.

Requiring **agreement + persistence** produces alerts that reflect meaningful system-level behavior rather than isolated noise.

