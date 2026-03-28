# HTM-Monitor

System anomalies fire only when grouped signals agree and persist.

## Method
- one HTM model per signal
- related signals grouped together
- system anomaly requires group **agreement + persistence**

## Use Cases

### CA Power Grid (2020-22)

#### Labeled Anomalies
- Outage Aug 14-16, 2020
- Outage Sept 6-9, 2022

#### Current Status
- Two labeled outage windows are consistently detected
- Additional spring 2020 anomalies are treated as **explanatory** rather than false positives due to onset of COVID

#### Interpretation
This use case contains at least three anomaly classes:

1. **Primary ground truth**
   - major outage windows in Aug 2020 and Sept 2022
2. **Explanatory non-outage anomalies**
   - spring 2020 COVID-era transition behavior, which produces real multi-signal disruption
3. **Residual non-GT anomalies**
   - shorter episodes not tied to the labeled outages; these appear more consistent with temporary breakdowns in predictability or unlabeled operational shifts than with the primary outage events

##### May 2020 System Anomaly
![May 2020 System Anomaly](assets/powergrid_ca_may2020.gif)

---
##### Aug 2020 System Anomaly
![Aug 2020 System Anomaly](assets/powergrid_ca_aug2020.gif)

---
##### Sept 2022 System Anomaly
![Sept 2022 System Anomaly](assets/powergrid_ca_sept2022.gif)

---

### SWaT Water Treatment (yy-yy)

Industrial control system with coordinated cyber-physical attacks.

#### Status
in progress

##### -- Attack Example

---
##### -- Attack Example

---
##### -- Attack Example

---

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


