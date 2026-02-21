import os

# Headless-safe backend (must be set before matplotlib.pyplot import inside LivePlot)
os.environ.setdefault("MPLBACKEND", "Agg")

from demo.live_plot import LivePlot


def test_contiguous_true_runs_endcaps_exclusive():
    xs = [10, 11, 12, 13, 14]
    flags = [0.0, 1.0, 1.0, 0.0, 1.0]

    runs = LivePlot._contiguous_true_runs(xs, flags)
    # first run should be [11, 13) => (11,13)
    # second run should end past last visible x, so (14, 15)
    assert runs == [(11, 13), (14, 15)]