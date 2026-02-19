# demo/live_plot.py
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Mapping, Optional, Set, Tuple

import matplotlib.pyplot as plt


@dataclass
class LivePlot:
    window: int = 1000
    refresh_every: int = 50
    gt_timestamps: Optional[Set[str]] = None  # timestamps with known anomalies
    timestamp_key: str = "timestamp"
    likelihood_threshold: Optional[float] = None  # draw horizontal line if set

    def __post_init__(self) -> None:
        # shared x over the sliding window
        self._t: Deque[int] = deque(maxlen=self.window)

        # system series
        self._alert: Deque[float] = deque(maxlen=self.window)  # 0/1

        # per-model series
        self._model_names: List[str] = []
        self._val: Dict[str, Deque[float]] = {}
        self._raw: Dict[str, Deque[float]] = {}
        self._lik: Dict[str, Deque[float]] = {}
        self._gt: Dict[str, Deque[float]] = {}  # 0/1 per timestep (within window)

        # matplotlib objects (built once we know models)
        self._fig = None
        self._axes_val: Dict[str, Any] = {}
        self._axes_anom: Dict[str, Any] = {}
        self._ax_sys = None

        self._l_val: Dict[str, Any] = {}
        self._l_raw: Dict[str, Any] = {}
        self._l_lik: Dict[str, Any] = {}
        self._thr_line: Dict[str, Any] = {}

        self._sys_fill = None
        self._sys_line = None

        # vline handles we explicitly create/remove (no try/except)
        self._gt_lines: Dict[str, List[Any]] = {}

        self._n = 0

        # styles (single source of truth)
        self._gt_style = dict(color="purple", linewidth=2.5, alpha=0.75, linestyle=":")

        plt.ion()

    def _init_model_series(self, model_names: List[str]) -> None:
        self._model_names = list(model_names)
        for m in self._model_names:
            self._val[m] = deque(maxlen=self.window)
            self._raw[m] = deque(maxlen=self.window)
            self._lik[m] = deque(maxlen=self.window)
            self._gt[m] = deque(maxlen=self.window)
            self._gt_lines[m] = []

    def _build_layout(self, model_names: List[str]) -> None:
        """
        Layout A:
          For each model:
            [value axis]
            [anom axis: raw+likelihood (+ threshold)]
          Bottom:
            [system alert axis]
        """
        self._init_model_series(model_names)

        n = len(self._model_names)
        nrows = (2 * n) + 1

        self._fig = plt.figure()
        gs = self._fig.add_gridspec(nrows, 1, height_ratios=[3, 1] * n + [1], hspace=0.15)

        first_ax = None
        for i, m in enumerate(self._model_names):
            axv = self._fig.add_subplot(gs[2 * i, 0], sharex=first_ax)
            axa = self._fig.add_subplot(gs[2 * i + 1, 0], sharex=first_ax)
            if first_ax is None:
                first_ax = axv

            self._axes_val[m] = axv
            self._axes_anom[m] = axa

            (lv,) = axv.plot([], [], label="value")
            (lr,) = axa.plot([], [], label="raw")
            (ll,) = axa.plot([], [], label="likelihood")
            self._l_val[m] = lv
            self._l_raw[m] = lr
            self._l_lik[m] = ll

            axv.set_title(f"{m}")
            axv.grid(True, alpha=0.3)
            axv.legend(loc="upper right")

            if self.likelihood_threshold is not None:
                self._thr_line[m] = axa.axhline(
                    self.likelihood_threshold, alpha=0.35, linestyle="--", label="threshold"
                )
            axa.grid(True, alpha=0.3)
            axa.set_ylim(0.0, 1.0)
            axa.legend(loc="upper right")

        self._ax_sys = self._fig.add_subplot(gs[-1, 0], sharex=first_ax)
        self._ax_sys.set_title("SYSTEM ALERT")
        self._ax_sys.set_ylim(-0.05, 1.05)
        self._ax_sys.grid(True, alpha=0.3)
        self._ax_sys.set_xlabel("t")

        # initialize system alert visuals
        (self._sys_line,) = self._ax_sys.step([], [], where="post", label="alert")
        self._ax_sys.legend(loc="upper right")
        self._fig.suptitle("HTM Monitor (live)", y=0.995)

    def update(
        self,
        t: int,
        row: Mapping[str, Any],
        model_outputs: Mapping[str, Mapping[str, Any]],
        result: Any,
    ) -> None:
        if not model_outputs:
            return

        # build layout on first update (or if models change)
        model_names = list(model_outputs.keys())
        if not self._model_names:
            self._build_layout(model_names)
        elif set(model_names) != set(self._model_names):
            # keep it simple: rebuild the figure if the set changes
            plt.close(self._fig)
            self._t.clear()
            self._alert.clear()
            self._val.clear()
            self._raw.clear()
            self._lik.clear()
            self._gt.clear()
            self._axes_val.clear()
            self._axes_anom.clear()
            self._l_val.clear()
            self._l_raw.clear()
            self._l_lik.clear()
            self._thr_line.clear()
            self._gt_lines.clear()
            self._build_layout(model_names)

        # multi-model values provided by run_pipeline.py (preferred)
        values_by_model = row.get("values_by_model")
        if not isinstance(values_by_model, Mapping):
            # backward compat: treat row["value"] as the first model's value
            first = next(iter(model_outputs.keys()))
            values_by_model = {first: row.get("value")}

        ts = row.get(self.timestamp_key)
        gt_by_model = row.get("gt_by_model")
        if not isinstance(gt_by_model, Mapping):
            gt_by_model = None

        alert = 1.0 if (isinstance(result, Mapping) and result.get("alert")) else 0.0

        self._t.append(int(t))
        self._alert.append(float(alert))

        # append one point per model (NaN for missing => gaps)
        for m, out in model_outputs.items():
            v = values_by_model.get(m)
            v_f = float(v) if isinstance(v, (int, float)) else float("nan")

            raw = out.get("raw")
            lik = out.get("likelihood")
            raw_f = float(raw) if isinstance(raw, (int, float)) else float("nan")
            lik_f = float(lik) if isinstance(lik, (int, float)) else float("nan")

            if gt_by_model is not None:
                gset = gt_by_model.get(m)
                gt_flag = bool(gset is not None and ts is not None and ts in gset)
            else:
                gt_flag = bool(self.gt_timestamps is not None and ts is not None and ts in self.gt_timestamps)

            self._val[m].append(v_f)
            self._raw[m].append(raw_f)
            self._lik[m].append(lik_f)
            self._gt[m].append(1.0 if gt_flag else 0.0)

        self._n += 1
        if self._n % int(self.refresh_every) != 0:
            return

        xs = list(self._t)
        # update model plots
        for m in self._model_names:
            axv = self._axes_val[m]
            axa = self._axes_anom[m]

            self._l_val[m].set_data(xs, list(self._val[m]))
            self._l_raw[m].set_data(xs, list(self._raw[m]))
            self._l_lik[m].set_data(xs, list(self._lik[m]))

            # clear + redraw GT vlines for this model (within window only)
            for ln in self._gt_lines[m]:
                ln.remove()
            self._gt_lines[m] = []

            for x, g in zip(xs, self._gt[m]):
                if g:
                    self._gt_lines[m].append(axv.axvline(x, **self._gt_style))

            axv.relim()
            axv.autoscale_view()
            axa.relim()
            axa.autoscale_view()
            axa.set_ylim(0.0, 1.0)  # keep likelihood visually comparable

        # update system alert (filled band + step)
        ys = list(self._alert)
        self._sys_line.set_data(xs, ys)
        # remove old fill, draw new fill (simple + very legible)
        if self._sys_fill is not None:
            self._sys_fill.remove()
        self._sys_fill = self._ax_sys.fill_between(xs, 0.0, ys, step="post", alpha=0.35)

        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()

        plt.pause(0.001)
