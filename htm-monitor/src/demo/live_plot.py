# demo/live_plot.py
from __future__ import annotations

import numbers
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
        # values: model -> feature -> series
        self._val: Dict[str, Dict[str, Deque[float]]] = {}
        self._val_features: Dict[str, List[str]] = {}
        self._raw: Dict[str, Deque[float]] = {}
        self._lik: Dict[str, Deque[float]] = {}
        self._gt: Dict[str, Deque[float]] = {}  # 0/1 per timestep (within window)
        self._hot: Dict[str, Deque[float]] = {}  # 0/1 per timestep (within window)

        # matplotlib objects (built once we know models)
        self._fig = None
        self._axes_val: Dict[str, Any] = {}
        self._axes_anom: Dict[str, Any] = {}
        self._ax_sys = None
        self._ax_spacer = None

        # line handles: model -> feature -> line
        self._l_val: Dict[str, Dict[str, Any]] = {}
        self._l_raw: Dict[str, Any] = {}
        self._l_lik: Dict[str, Any] = {}
        self._thr_line: Dict[str, Any] = {}

        self._sys_fill = None
        self._sys_line = None
        self._legend = None

        # vline handles we explicitly create/remove (no try/except)
        self._gt_lines: Dict[str, List[Any]] = {}
        self._hot_spans: Dict[str, List[Any]] = {}

        self._n = 0

        # styles (single source of truth)
        self._gt_style = dict(color="purple", linewidth=2.5, alpha=0.75, linestyle=":")
        self._hot_span_style = dict(alpha=0.18, linewidth=0)  # very light highlight
        # keep values visually distinct from raw/likelihood (blue/orange)
        self._value_colors = ["black", "dimgray", "gray", "slategray"]

        plt.ion()

    def _init_model_series(self, model_names: List[str]) -> None:
        self._model_names = list(model_names)
        for m in self._model_names:
            self._val[m] = {}
            self._val_features[m] = []
            self._l_val[m] = {}
            self._raw[m] = deque(maxlen=self.window)
            self._lik[m] = deque(maxlen=self.window)
            self._gt[m] = deque(maxlen=self.window)
            self._gt_lines[m] = []
            self._hot[m] = deque(maxlen=self.window)
            self._hot_spans[m] = []

    @staticmethod
    def _display_name(model_name: str) -> str:
        # Save space: "jumpsdown_model" -> "jumpsdown"
        return model_name.removesuffix("_model")

    @staticmethod
    def _contiguous_true_runs(xs: List[int], flags: List[float]) -> List[Tuple[int, int]]:
        """
        Return runs (x_start, x_end_exclusive) where flags are truthy.
        Uses xs positions (assumes xs is sorted, one per timestep).
        """
        runs: List[Tuple[int, int]] = []
        if not xs or not flags:
            return runs
        # infer a reasonable step for end-caps (t is usually integer, so dt=1)
        dt = 1
        if len(xs) >= 2:
            dx = xs[1] - xs[0]
            if dx > 0:
                dt = dx

        start: Optional[int] = None
        for x, f in zip(xs, flags):
            if f and start is None:
                start = x
            elif (not f) and start is not None:
                # end exclusive at first non-hot timestep
                runs.append((start, x))
                start = None
        if start is not None:
            # end exclusive past the final visible timestep so span is visible
            runs.append((start, xs[-1] + dt))
        return runs

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
        # Add one dedicated spacer row before SYSTEM ALERT to create visible separation.
        # Layout: (value, anom) * n, spacer, system
        nrows = (2 * n) + 2

        self._fig = plt.figure()
        # Give SYSTEM ALERT a bit more height + add more vertical breathing room overall.
        # (Keeps the demo legible to non-technical viewers.)
        gs = self._fig.add_gridspec(
            nrows,
            1,
            height_ratios=[3, 1] * n + [0.55, 1.6],
            hspace=0.18,
        )

        first_ax = None
        for i, m in enumerate(self._model_names):
            axv = self._fig.add_subplot(gs[2 * i, 0], sharex=first_ax)
            axa = self._fig.add_subplot(gs[2 * i + 1, 0], sharex=first_ax)
            if first_ax is None:
                first_ax = axv

            self._axes_val[m] = axv
            self._axes_anom[m] = axa

            # value lines are created lazily once we see feature names
            (lr,) = axa.plot([], [], label="raw")              # default blue
            (ll,) = axa.plot([], [], label="likelihood")       # default orange
            self._l_raw[m] = lr
            self._l_lik[m] = ll

            # Put model name on the left instead of crowding titles
            axv.set_ylabel(self._display_name(m), rotation=0, labelpad=30, va="center")
            axv.grid(True, alpha=0.3)

            if self.likelihood_threshold is not None:
                self._thr_line[m] = axa.axhline(
                    self.likelihood_threshold, alpha=0.35, linestyle="--", label="_nolegend_"
                )
            axa.grid(True, alpha=0.3)
            axa.set_ylim(0.0, 1.0)

            # Hide x tick labels everywhere except the very bottom system panel
            axv.tick_params(labelbottom=False)
            axa.tick_params(labelbottom=False)

        # spacer axis: share x, but hide everything (acts like a gap)
        self._ax_spacer = self._fig.add_subplot(gs[-2, 0], sharex=first_ax)
        self._ax_spacer.set_axis_off()
        # ensure no tick labels appear in the spacer row
        self._ax_spacer.tick_params(labelbottom=False)

        self._ax_sys = self._fig.add_subplot(gs[-1, 0], sharex=first_ax)
        self._ax_sys.set_title("SYSTEM ALERT", pad=10)
        self._ax_sys.set_ylim(-0.05, 1.05)
        self._ax_sys.grid(True, alpha=0.3)
        self._ax_sys.set_xlabel("t")

        # initialize system alert visuals
        (self._sys_line,) = self._ax_sys.step([], [], where="post", label="system alert", color="red")
        self._fig.suptitle("HTM Monitor (live)", y=0.995)

        # Single global legend (clean, non-crowded)
        # We add a representative "value(s)" handle later once value lines exist.
        handles = [lr, ll, self._sys_line]
        labels = ["raw anomaly", "anomaly likelihood", "system alert"]
        self._legend = self._fig.legend(handles, labels, loc="upper right")

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
            self._hot.clear()
            self._axes_val.clear()
            self._axes_anom.clear()
            self._l_val.clear()
            self._l_raw.clear()
            self._l_lik.clear()
            self._thr_line.clear()
            self._gt_lines.clear()
            self._hot_spans.clear()
            self._build_layout(model_names)

        # values_by_model: model -> {feature -> value}
        values_by_model = row.get("values_by_model")
        if not isinstance(values_by_model, Mapping):
            raise ValueError("LivePlot expects row['values_by_model'] as model -> {feature -> value}")

        ts = row.get(self.timestamp_key)
        gt_by_model = row.get("gt_by_model")
        if not isinstance(gt_by_model, Mapping):
            gt_by_model = None

        alert = 1.0 if (isinstance(result, Mapping) and result.get("alert")) else 0.0
        hot_by_model = result.get("hot_by_model") if isinstance(result, Mapping) else None
        if not isinstance(hot_by_model, Mapping):
            hot_by_model = None

        self._t.append(int(t))
        self._alert.append(float(alert))

        # append one point per model (NaN for missing => gaps)
        for m, out in model_outputs.items():
            vmap = values_by_model.get(m)
            if not isinstance(vmap, Mapping):
                vmap = {}

            # Lazily initialize per-feature series + line handles once we see features
            feat_names = list(vmap.keys())
            if self._val_features.get(m) != feat_names:
                # If features change (rare), rebuild figure (keep it simple + correct)
                if self._val_features.get(m) and self._val_features[m] != feat_names:
                    plt.close(self._fig)
                    self._t.clear()
                    self._alert.clear()
                    self._val.clear()
                    self._val_features.clear()
                    self._raw.clear()
                    self._lik.clear()
                    self._gt.clear()
                    self._hot.clear()
                    self._axes_val.clear()
                    self._axes_anom.clear()
                    self._l_val.clear()
                    self._l_raw.clear()
                    self._l_lik.clear()
                    self._thr_line.clear()
                    self._gt_lines.clear()
                    self._hot_spans.clear()
                    self._build_layout(model_names)
                self._val_features[m] = feat_names
                for f in feat_names:
                    if f not in self._val[m]:
                        self._val[m][f] = deque(maxlen=self.window)
                        c = self._value_colors[len(self._l_val[m]) % len(self._value_colors)]
                        (ln,) = self._axes_val[m].plot([], [], label="_nolegend_", color=c)
                        self._l_val[m][f] = ln

            raw = out.get("raw")
            lik = out.get("likelihood")
            raw_f = float(raw) if isinstance(raw, numbers.Real) else float("nan")
            lik_f = float(lik) if isinstance(lik, numbers.Real) else float("nan")

            if gt_by_model is not None:
                gset = gt_by_model.get(m)
                gt_flag = bool(gset is not None and ts is not None and ts in gset)
            else:
                gt_flag = bool(self.gt_timestamps is not None and ts is not None and ts in self.gt_timestamps)

            # append per-feature values
            for f in self._val_features.get(m, []):
                vv = vmap.get(f)
                v_f = float(vv) if isinstance(vv, numbers.Real) else float("nan")
                self._val[m][f].append(v_f)
            self._raw[m].append(raw_f)
            self._lik[m].append(lik_f)
            self._gt[m].append(1.0 if gt_flag else 0.0)

            # Hot flag: prefer Decision-provided truth; fallback to simple threshold
            if hot_by_model is not None:
                self._hot[m].append(1.0 if bool(hot_by_model.get(m)) else 0.0)
            else:
                is_hot = bool(
                    isinstance(lik, numbers.Real)
                    and self.likelihood_threshold is not None
                    and float(lik) >= float(self.likelihood_threshold)
                )
                self._hot[m].append(1.0 if is_hot else 0.0)

        self._n += 1
        if self._n % int(self.refresh_every) != 0:
            return

        xs = list(self._t)
        if xs:
            # Force a true sliding x-window (matplotlib otherwise keeps old limits)
            self._ax_sys.set_xlim(xs[0], xs[-1])

        # update model plots
        for m in self._model_names:
            axv = self._axes_val[m]
            axa = self._axes_anom[m]

            # value lines (multi-signal)
            for f in self._val_features.get(m, []):
                ln = self._l_val[m].get(f)
                if ln is not None:
                    ln.set_data(xs, list(self._val[m][f]))
            self._l_raw[m].set_data(xs, list(self._raw[m]))
            self._l_lik[m].set_data(xs, list(self._lik[m]))

            # clear + redraw GT vlines for this model (within window only)
            for ln in self._gt_lines[m]:
                ln.remove()
            self._gt_lines[m] = []

            for x, g in zip(xs, self._gt[m]):
                if g:
                    self._gt_lines[m].append(axv.axvline(x, **self._gt_style))

            # clear + redraw HOT spans (simple + obvious)
            for sp in self._hot_spans[m]:
                sp.remove()
            self._hot_spans[m] = []
            runs = self._contiguous_true_runs(xs, list(self._hot[m]))
            for x0, x1 in runs:
                # highlight on the value axis so it's visible even if anom axis is busy
                self._hot_spans[m].append(axv.axvspan(x0, x1, **self._hot_span_style))

            axv.relim()
            axv.autoscale_view()
            axa.relim()
            axa.autoscale_view()
            axa.set_ylim(0.0, 1.0)  # keep likelihood visually comparable

        # Ensure legend includes a representative "value(s)" handle (once lines exist)
        if self._legend is not None and self._model_names:
            m0 = self._model_names[0]
            any_val_line = next(iter(self._l_val.get(m0, {}).values()), None)
            if any_val_line is not None:
                handles = [any_val_line, self._l_raw[m0], self._l_lik[m0], self._sys_line]
                labels = ["value(s)", "raw anomaly", "anomaly likelihood", "system alert"]
                self._legend.remove()
                self._legend = self._fig.legend(handles, labels, loc="upper right")

        # update system alert (filled band + step)
        ys = list(self._alert)
        self._sys_line.set_data(xs, ys)
        # remove old fill, draw new fill (simple + very legible)
        if self._sys_fill is not None:
            self._sys_fill.remove()
        self._sys_fill = self._ax_sys.fill_between(xs, 0.0, ys, step="post", alpha=0.25)

        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()

        plt.pause(0.001)
