```bash
# 1. Generate a usecase config
python -m htm_monitor.cli.usecase_wizard \
  --out-dir configs/

# 2. Run pipeline (no plot)
python -m htm_monitor.cli.run_pipeline \
  --defaults configs/htm_defaults.yaml \
  --config configs/<USECASE>.yaml \
  --run-dir outputs/<USECASE> \
  --no-plot

# 3. Analyze results
python -m htm_monitor.cli.analyze_run \
  --run-dir outputs/<USECASE> \
  --config configs/<USECASE>.yaml
```
