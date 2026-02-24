#tests/test_live_plot_runs.py

from demo.live_plot import LivePlot


def test_contiguous_true_runs_endcaps_exclusive():
    xs = [10, 11, 12, 13, 14]
    flags = [0.0, 1.0, 1.0, 0.0, 1.0]

    runs = LivePlot._contiguous_true_runs(xs, flags)
    # first run should be [11, 13) => (11,13)
    # second run should end past last visible x, so (14, 15)
    assert runs == [(11, 13), (14, 15)]

def test_liveplot_update_rejects_missing_values_by_model():
    lp = LivePlot(refresh_every=1)
    try:
        lp.update(
            t=0,
            row={"timestamp": "t0"},  # missing values_by_model
            model_outputs={"m": {"raw": 0.1, "anomaly_probability": 0.2}},
            result={"alert": False},
        )
        assert False, "Expected ValueError for missing values_by_model"
    except ValueError:
        pass

def test_liveplot_update_accepts_basic_shape():
    lp = LivePlot(refresh_every=1)
    lp.update(
        t=0,
        row={
            "timestamp": "t0",
            "values_by_model": {"m": {"a": 1.0}},
        },
        model_outputs={"m": {"raw": 0.1, "anomaly_probability": 0.2}},
        result={"alert": False, "hot_by_model": {"m": False}},
    )
