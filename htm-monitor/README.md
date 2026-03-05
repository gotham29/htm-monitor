# HTM-Monitor

Real-time anomaly detection for streaming time series using **Hierarchical Temporal Memory (HTM)**.

HTM-Monitor learns normal behavior from live signals and detects structural deviations **in real time**, converting per-signal anomaly probabilities into a single **system-level alert** for the monitored system.

![Live demo](assets/live_demo.gif)

Signals stream in → HTM learns normal behavior online → anomaly probability rises → a **system alert** triggers.

*Example: three signals monitored simultaneously with overlapping anomaly events.*

---

## Where this is useful

HTM-Monitor is designed for **systems where behavior unfolds over time** and deviations matter more than static thresholds.

Typical applications include:

- **Industrial monitoring** — machines, manufacturing lines, process signals  
- **Infrastructure monitoring** — servers, cloud systems, observability metrics  
- **Cybersecurity signals** — network traffic or behavioral indicators  
- **IoT / sensor networks** — multi-sensor environments with evolving patterns

Unlike many anomaly detection systems, HTM learns **temporal structure online** and adapts continuously as new data arrives.

---

## How HTM differs from typical anomaly detection

Many anomaly detection systems rely on approaches such as:

- static thresholds
- batch-trained statistical models
- or models that must be retrained when behavior shifts

HTM takes a different approach. It is designed for **continuous learning on streaming data**.

Key differences:

- **Online learning** — the model updates continuously as new data arrives  
- **Temporal memory** — it learns sequences and transitions, not just value distributions  
- **No retraining cycles** — the system adapts automatically as patterns evolve  
- **Multi-signal monitoring** — signals can be combined into a single system-level alert

This makes HTM particularly well suited for systems where anomalies appear as **changes in temporal structure**, not just extreme values.

---

## Live demo (streaming detection)

**Legend**
- **value(s)** — incoming signal values
- **HTM raw anomaly score** — raw HTM anomaly signal
- **HTM anomaly probability** — interpretable score (0..1)
- **pink spans** — model considered “hot”
- **purple dotted lines** — ground-truth anomaly timestamps
- **system alert** — final decision combining models

---

## System evaluation

This example run is scored against known ground-truth anomaly timestamps.

![System eval scorecard](assets/system_eval_scorecard.png)

Static overview of the same run:

![Run overview](assets/run_overview.png)

Each run produces a self-contained directory:

```
outputs/<usecase>/<run_id>/
  run.csv
  run.manifest.json
  analysis/
    run_summary.json
    run_summary.md
```

Runs are therefore:
- reproducible
- easy to diff
- easy to share (just send the run folder)

---

## Quickstart: run the demo locally

### 1) Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Run one command

```bash
python quickstart.py --usecase demo_synth --mode synth --run-id run_001 --make-gif
```

This will:
1. generate synthetic data with subtle injected anomalies
2. build a demo config (`configs/demo_synth.yaml`)
3. run the pipeline
4. analyze results
5. optionally record frames and write assets/live_demo.gif

Outputs you’ll care about:
- `assets/live_demo.gif` *(README asset)*
- `assets/system_eval_scorecard.png` *(README asset)*
- `assets/run_overview.png` *(README asset)*

---

## Bring your own dataset

HTM-Monitor expects each source CSV to have:

```csv
timestamp,value
2015-03-01 00:00:00,123.4
2015-03-01 00:30:00,121.9
...
```

### Wizard mode (recommended for first-time real data)

```bash
python quickstart.py --mode wizard --usecase my_dataset --run-id run_001
```

Wizard mode writes:
- `configs/my_dataset.build.yaml` (build spec: sources + knobs)
- `configs/my_dataset.yaml` (the runnable config)

Then it runs:
- `src.htm_monitor.cli.run_pipeline`
- `src.htm_monitor.cli.analyze_run`

---

## How “system alerts” are computed

The system separates:
- **Engine** — produces per-model anomaly probabilities
- **Decision** — converts those into:
  - per-model “hot” status
  - a final **system alert**

In the demo config we use a *k-of-n window* decision:
- a model becomes hot if its probability exceeds a threshold enough times
- system alert triggers if at least **k** models are hot within a sliding window

This gives you:
- robustness to single-sensor noise
- a clean “system is abnormal now” signal

---

## Repository layout

```
src/htm_monitor/
  cli/             # run_pipeline, analyze_run, wizard/build tools
  htm_src/         # HTM components (encoding, TM/SP integration, anomaly likelihood)
  orchestration/   # engine + decision logic
  viz/             # live plotting
  diagnostics/     # diagnostics CSVs + health checks
```

---

## Notes / FAQ

### Why HTM?
HTM is well-suited to streaming settings where:
- patterns are temporal
- you want online learning
- you want early detection of structural deviations

### Can I run without the live plot?
Yes:

```bash
python quickstart.py --usecase demo_synth --run-id run_001 --no-plot
```

You’ll still get:
- `outputs/.../analysis/run_summary.md`
- `outputs/.../analysis/system_eval_scorecard.png`
- `outputs/.../analysis/run_overview.png`

---

## License

(add your license here)