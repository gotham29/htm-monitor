# HTM-Monitor

+Real-time multi-signal anomaly detection for streaming time series using **Hierarchical Temporal Memory (HTM)**.

![Live demo](assets/demo_synth.gif)

Signals stream in → HTM learns online → anomaly probability rises → **system alert** triggers.

*Example: three signals monitored simultaneously with overlapping anomaly events.*

Each run writes reproducible artifacts (`run.csv`, `run.manifest.json`, evaluation summaries`) for inspection and sharing.

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

## Quickstart (synthetic demo)

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
1. generate synthetic signals with injected anomalies
2. build a demo config
3. run the pipeline
4. analyze results
5. optionally write `assets/demo_synth.gif`

Outputs you’ll care about:
- `assets/demo_synth.gif` *(README asset)*
- `assets/system_eval_scorecard.png` *(README asset)*
- `assets/run_overview.png` *(README asset)*

---

## Real data example: NAB (exchange-4 CPC/CPM)

This walkthrough runs HTM-Monitor on two real time series from the Numenta Anomaly Benchmark (NAB):
`exchange-4_cpc_results.csv` and `exchange-4_cpm_results.csv`.

Note: the NAB dataset is typically kept **outside** the repo. The generated config will reference your local CSV paths.

### 1) Generate the usecase config (interactive wizard)

This writes:
- `specs/nab_exchange4.build.yaml` *(reproducible “build spec”)*
- `configs/nab_exchange4.yaml` *(runnable config)*

```bash
python -m htm_monitor.cli.usecase_wizard \
  --spec-out specs/nab_exchange4.build.yaml \
  --out-dir configs
```

### 2) Run the pipeline

`--out` is a **file path** for `run.csv` (not a directory). The manifest is written next to it automatically.

```bash
python -m htm_monitor.cli.run_pipeline \
  --config configs/nab_exchange4.yaml \
  --defaults configs/htm_defaults.yaml \
  --out outputs/nab_exchange4/run.csv
```

### 3) Analyze the run

`analyze_run` requires the config (for ground truth + decision semantics).

```bash
python -m htm_monitor.cli.analyze_run \
  --run-dir outputs/nab_exchange4 \
  --config configs/nab_exchange4.yaml
```

Outputs are written to:
`outputs/nab_exchange4/analysis/run_summary.{json,md}`

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

## FAQ

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