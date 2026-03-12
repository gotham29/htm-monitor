
# HTM-Monitor Use Case Semantics

This document defines the conceptual task modes supported by HTM-Monitor.
The goal is to separate **problem semantics** from **implementation details** so
the repository can support multiple real-world monitoring applications without
ad-hoc logic.

---

## 1. Task Types

HTM-Monitor currently supports two conceptual monitoring tasks.

## 1.1 Onset Detection

**Definition**  
Detect the beginning of an abnormal regime in a streaming system.

**Examples**
- Synthetic demo datasets
- NAB anomaly detection benchmarks
- Infrastructure monitoring
- Cybersecurity anomaly detection

**Ground Truth Semantics**
Ground truth timestamps represent the **start of abnormal behavior**.

**Desired Alert Behavior**
Alerts should occur **shortly after the onset** of the abnormal regime.

**Evaluation Characteristics**
Typical metrics include:

- Detection lag after onset
- Missed detections
- False positives outside anomaly windows
- Episode matching between alerts and GT onsets

This is the evaluation mode currently implemented in the repository.

## 1.2 Predictive Warning (Predictive Maintenance)

**Definition**  
Detect degradation early enough to warn before a failure or critical event.

**Examples**
- Predictive maintenance
- Turbofan degradation (CMAPSS)
- Tool wear monitoring
- Battery health prediction
- Healthcare deterioration monitoring

**Ground Truth Semantics**
Ground truth timestamps represent a **future event**, not the regime change.

Examples:
- component failure
- maintenance threshold
- critical RUL horizon

**Desired Alert Behavior**

Alerts should occur:

- **after degradation begins**
- **before the failure horizon**
- **within an actionable warning window**

Earlier is **not always better**. Alerts that occur far too early may be
operationally useless.

**Evaluation Characteristics**

Evaluation should measure:

- lead time before event
- first alert timing relative to warning band
- alerts that are too early
- alerts that are too late
- missing alerts

This mode will be introduced to support CMAPSS and similar datasets.

---

## 2. Learning Policies

Monitoring systems may use different learning strategies.

## 2.1 Online Continuous Learning

Model learns continuously while detecting anomalies.

Typical for:

- infrastructure monitoring
- telemetry anomaly detection
- unknown evolving environments

## 2.2 Warmup then Online

Model observes an initial warmup period and then begins detection.

Used in many streaming anomaly detection systems.

Current repo support:

```yaml
run:
  warmup_steps: N
  learn_after_warmup: true
```

## 2.3 Train then Freeze

Model is trained on a historical dataset and then **frozen** for inference.

Typical for:

- predictive maintenance
- reliability monitoring
- supervised operational deployment

Example workflow:

1. Train on fleet historical data
2. Freeze model
3. Monitor new unit
4. Issue alerts

This learning mode will support the CMAPSS predictive maintenance demo.

---

## 3. Ground Truth Semantics

Ground truth annotations may represent different meanings.

## Onset

Timestamp represents the **start of abnormal behavior**.

Used in:

- synthetic demos
- NAB benchmark

## Event

Timestamp represents the **occurrence of a critical event**.

Example:

- machine failure
- crash
- service outage

## Event Horizon

Timestamp represents a **future event**, but alerts should occur **before it**
within a useful warning band.

Example:

- failure within RUL horizon
- maintenance threshold approaching

This is the semantic used for **predictive maintenance datasets like CMAPSS**.

---

## 4. Evaluation Modes

Evaluation should align with task semantics.

## Onset Detection Evaluation

Measures:

- detection lag after onset
- false positives
- missed detections
- alert episode matching

## Predictive Warning Evaluation

Measures:

- lead time before event
- alert placement relative to warning band
- too-early warnings
- too-late warnings
- missed warnings


---

## 5. CMAPSS Example

CMAPSS represents **predictive maintenance** rather than onset detection.

Characteristics:

- failure occurs at end of trajectory
- degradation begins earlier
- sensors gradually drift

Desired demo behavior:

1. Learn normal fleet behavior
2. Monitor a held-out engine
3. Detect deviation from fleet patterns
4. Raise warning before failure horizon

Evaluation should reward alerts occurring within a meaningful warning band.

---

## 6. Design Goal

Separate **monitoring semantics** from **HTM implementation details**.

This enables HTM-Monitor to support multiple application domains:

- infrastructure monitoring
- predictive maintenance
- cyber anomaly detection
- telemetry monitoring
- healthcare deterioration detection

without rewriting core evaluation logic for each dataset.