# src/htm_monitor/viz/live_plot.py

from __future__ import annotations

import numbers
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, List, Mapping, Optional, Set, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def _coerce_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None

@dataclass
class LivePlot:
    window: int = 1000
    refresh_every: int = 50
    gt_timestamps: Optional[Set[str]] = None  # timestamps with known anomalies
    timestamp_key: str = "timestamp"
    show_warmup_span: bool = True
    # Which per-model score to plot on the anomaly panel (must be in [0,1] for the y-limits).
    # Recommended: "anomaly_probability" (aka "p").
    plot_score_key: str = "anomaly_probability"
    # Optional: record frames for README GIFs.
    record_dir: Optional[str] = None
    record_every: int = 1
    record_dpi: int = 140

    def __post_init__(self) -> None:
        # shared x over the sliding window
        self._t: Deque[int] = deque(maxlen=self.window)
        self._ts: Deque[Optional[str]] = deque(maxlen=self.window)

        # system series
        self._alert: Deque[float] = deque(maxlen=self.window)  # 0/1
        # warmup series (0/1)
        self._warm: Deque[float] = deque(maxlen=self.window)
        # system GT series (0/1) for SYSTEM ALERT panel
        self._gt_sys: Deque[float] = deque(maxlen=self.window)

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

        self._sys_fill = None
        self._sys_line = None
        self._legend = None

        # vline handles we explicitly create/remove (no try/except)
        self._gt_lines: Dict[str, List[Any]] = {}
        self._hot_spans: Dict[str, List[Any]] = {}
        self._warmup_spans: List[Any] = []
        # GT bands (more visible than dotted vlines)
        self._gt_spans: Dict[str, List[Any]] = {}
        self._gt_sys_spans: List[Any] = []

        # recording state
        self._record_dir: Optional[Path] = Path(self.record_dir).resolve() if self.record_dir else None
        self._record_dir.mkdir(parents=True, exist_ok=True) if self._record_dir else None
        self._frame_i = 0

        # Visual defaults for demo-quality readability (screenshots + GIFs)
        # Keep global mutations minimal: only rcParams that affect this figure.
        plt.rcParams.update(
            {
                "figure.facecolor": "white",
                "axes.facecolor": "white",
                "axes.titlesize": 12,
                "axes.labelsize": 10,
                "xtick.labelsize": 9,
                "ytick.labelsize": 9,
                "legend.fontsize": 9,
            }
        )

        self._n = 0

        # styles (single source of truth)
        # NOTE: these are used in update(); they MUST exist even before we build the layout.
        # Keep them simple + readable for demo screenshots.
        # Crisp GT indicator (optional vline)
        self._gt_style: Dict[str, Any] = dict(
            color="magenta",
            linewidth=2.5,
            linestyle=":",
            alpha=0.95,
            zorder=8,
        )
        # Highly visible GT band (recommended for demo)
        self._gt_span_style: Dict[str, Any] = dict(
            facecolor="magenta",
            alpha=0.12,
            linewidth=0.0,
            zorder=1,   # behind data lines
        )
        self._hot_span_style: Dict[str, Any] = dict(facecolor="orange", alpha=0.10, linewidth=0.0, zorder=0.5)
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
            self._gt_spans[m] = []
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
            [anom axis: raw + anomaly probability (+ threshold)]
          Bottom:
            [system alert axis]
        """
        self._init_model_series(model_names)

        n = len(self._model_names)
        # Add one dedicated spacer row before SYSTEM ALERT to create visible separation.
        # Layout: (value, anom) * n, spacer, system
        nrows = (2 * n) + 2

        # Size the figure for legibility. GIFs are usually viewed smaller than the live window.
        # Tune height by number of models; cap so it doesn’t get absurd for many models.
        fig_w = 12.0
        fig_h = min(3.6 + (2.2 * n), 16.0)
        self._fig = plt.figure(figsize=(fig_w, fig_h))
        self._fig.patch.set_facecolor("white")

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
            # The “raw” series is useful for debugging but can be noisy; keep it.
            (lr,) = axa.plot([], [], label="Anomaly score", linewidth=1.0)  # default blue
            # The main story for non-technical viewers:
            # anomaly probability (or whatever plot_score_key selects).
            (ll,) = axa.plot([], [], label="Anomaly probability", linewidth=2.3, zorder=5)
            self._l_raw[m] = lr
            self._l_lik[m] = ll

            # Put model name on the left instead of crowding titles
            axv.set_ylabel(self._display_name(m), rotation=0, labelpad=30, va="center")
            axv.grid(True, alpha=0.3)
            axa.grid(True, alpha=0.3)
            # IMPORTANT: keep anomaly panel fixed to [0,1]. Do NOT autoscale it later.
            # Give headroom so lines at ~1.0 don't "disappear" into the top spine.
            axa.set_ylim(-0.05, 1.05)
            axa.set_yticks([0.0, 0.5, 1.0])

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
        (self._sys_line,) = self._ax_sys.step(
            [], [], where="post", label="system alert", color="red", linewidth=2.5
        )
        self._fig.suptitle("HTM Monitor (live)", y=0.99, fontsize=14)

        # Single global legend (clean, non-crowded)
        # We add a representative "value(s)" handle later once value lines exist.
        handles = [lr, ll, self._sys_line]
        labels = ["Anomaly score", "Anomaly probability", "system alert"]
        self._legend = self._fig.legend(
            handles,
            labels,
            loc="upper right",
            frameon=True,
            fancybox=True,
            framealpha=0.90,
        )

        # Give the plot a little padding so titles/legend don’t feel cramped.
        self._fig.subplots_adjust(left=0.08, right=0.98, top=0.96, bottom=0.06)

    def _maybe_record_frame(self) -> None:
        if self._record_dir is None or self._fig is None:
            return
        every = max(1, int(self.record_every))
        # only record on refresh ticks (this function is called after refresh gating)
        if (self._frame_i % every) != 0:
            self._frame_i += 1
            return
        out = self._record_dir / f"frame_{self._frame_i:05d}.png"
        # “Presentation-grade” frames:
        # - tight bounding box removes excess whitespace
        # - facecolor ensures consistent background in GIFs
        self._fig.savefig(
            out,
            dpi=int(self.record_dpi),
            bbox_inches="tight",
            pad_inches=0.10,
            facecolor=self._fig.get_facecolor(),
        )
        self._frame_i += 1

    def _reset_and_rebuild(self, model_names: List[str]) -> None:
        if self._fig is not None:
            plt.close(self._fig)
        self._t.clear()
        self._ts.clear()
        self._alert.clear()
        self._warm.clear()
        self._gt_sys.clear()
        self._model_names = []
        self._val.clear()
        self._val_features.clear()
        self._raw.clear()
        self._lik.clear()
        self._gt.clear()
        self._hot.clear()
        self._axes_val.clear()
        self._axes_anom.clear()
        self._ax_sys = None
        self._ax_spacer = None
        self._l_val.clear()
        self._l_raw.clear()
        self._l_lik.clear()
        self._gt_lines.clear()
        self._hot_spans.clear()
        self._sys_fill = None
        self._sys_line = None
        self._legend = None
        self._warmup_spans = []
        self._gt_sys_spans = []
        self._build_layout(model_names)

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
            self._reset_and_rebuild(model_names)

        # values_by_model: model -> {feature -> value}
        values_by_model = row.get("values_by_model")
        if not isinstance(values_by_model, Mapping):
            raise ValueError("LivePlot expects row['values_by_model'] as model -> {feature -> value}")

        ts = row.get(self.timestamp_key)
        gt_by_model = row.get("gt_by_model")
        if not isinstance(gt_by_model, Mapping):
            gt_by_model = None
        # system GT timestamps set (already computed in run_pipeline)
        gt_system = row.get("gt_system")
        if not isinstance(gt_system, (set, frozenset)):
            gt_system = None

        alert = 1.0 if (isinstance(result, Mapping) and result.get("alert")) else 0.0
        hot_by_model = result.get("hot_by_model") if isinstance(result, Mapping) else None
        if not isinstance(hot_by_model, Mapping):
            hot_by_model = None
        in_warmup = bool(row.get("in_warmup", False))

        self._t.append(int(t))
        self._ts.append(str(ts) if ts is not None else None)
        self._alert.append(float(alert))
        self._warm.append(1.0 if in_warmup else 0.0)
        # system GT is timestamp-membership, same convention as per-model GT
        sys_gt_flag = bool(gt_system is not None and ts is not None and ts in gt_system)
        self._gt_sys.append(1.0 if sys_gt_flag else 0.0)

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
                    self._reset_and_rebuild(model_names)

                self._val_features[m] = feat_names
                for f in feat_names:
                    if f not in self._val[m]:
                        self._val[m][f] = deque(maxlen=self.window)
                        c = self._value_colors[len(self._l_val[m]) % len(self._value_colors)]
                        (ln,) = self._axes_val[m].plot([], [], label="_nolegend_", color=c)
                        self._l_val[m][f] = ln

            raw = out.get("raw")
            # Plot score for the non-technical story:
            # - Primary: whatever plot_score_key says (default: anomaly_probability)
            # - Back-compat fallbacks: anomaly_probability, p, likelihood
            score = out.get(self.plot_score_key)

            raw_f = float(raw) if isinstance(raw, numbers.Real) else float("nan")
            score_f = float(score) if isinstance(score, numbers.Real) else float("nan")
 
            # Hide probability series during warmup (cleaner demo).
            # Keep raw_f (debuggable), but probability should start after warmup.
            if in_warmup:
                score_f = float("nan")

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
            self._lik[m].append(score_f)
            self._gt[m].append(1.0 if gt_flag else 0.0)

            # Hot flag: prefer Decision-provided truth; fallback to simple threshold
            if hot_by_model is not None:
                self._hot[m].append(1.0 if bool(hot_by_model.get(m)) else 0.0)
            else:
                # No threshold line => no threshold-based "hot" fallback.
                self._hot[m].append(0.0)

        self._n += 1
        if self._n % int(self.refresh_every) != 0:
            return

        xs = list(self._t)
        if xs:
            # HARD sliding window x-limits on ALL axes, every refresh.
            # This prevents the "disappearing" behavior when deque hits maxlen.
            x1 = xs[-1]
            x0 = x1 - (int(self.window) - 1)
            for mm in self._model_names:
                self._axes_val[mm].set_xlim(x0, x1)
                self._axes_anom[mm].set_xlim(x0, x1)
            if self._ax_sys is not None:
                self._ax_sys.set_xlim(x0, x1)

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

            # clear + redraw GT spans (far more visible than vlines)
            for sp in self._gt_spans[m]:
                sp.remove()
            self._gt_spans[m] = []
            for x, g in zip(xs, self._gt[m]):
                if g:
                    # a very thin band around the timestep
                    x0, x1 = x - 0.5, x + 0.5
                    self._gt_spans[m].append(axv.axvspan(x0, x1, **self._gt_span_style))
                    self._gt_spans[m].append(axa.axvspan(x0, x1, **self._gt_span_style))

            # clear + redraw GT vlines for this model (within window only)
            for ln in self._gt_lines[m]:
                ln.remove()
            self._gt_lines[m] = []

            for x, g in zip(xs, self._gt[m]):
                if g:
                    # Optional crisp line on top of the GT band
                    self._gt_lines[m].append(axv.axvline(x, **self._gt_style))
                    self._gt_lines[m].append(axa.axvline(x, **self._gt_style))

            # clear + redraw HOT spans (simple + obvious)
            for sp in self._hot_spans[m]:
                sp.remove()
            self._hot_spans[m] = []
            runs = self._contiguous_true_runs(xs, list(self._hot[m]))
            for x0, x1 in runs:
                # highlight on the value axis so it's visible even if anom axis is busy
                self._hot_spans[m].append(axv.axvspan(x0, x1, **self._hot_span_style))

            axv.relim()
            # IMPORTANT: NEVER let autoscale touch x (we control x via hard sliding window).
            axv.autoscale_view(scalex=False, scaley=True)
            # IMPORTANT: DO NOT autoscale anomaly axes (it causes the “weird” looking panels).
            # Keep anomaly y fixed; x is shared already.
            # axa.relim(); axa.autoscale_view()  # intentionally disabled
            axa.set_ylim(-0.05, 1.05)

        # Clear + redraw SYSTEM GT spans (pink band) on system axis
        for sp in self._gt_sys_spans:
            sp.remove()
        self._gt_sys_spans = []
        if self._ax_sys is not None:
            for x, g in zip(xs, list(self._gt_sys)):
                if g:
                    # thin band around timestep, same as model GT bands
                    x0, x1 = x - 0.5, x + 0.5
                    self._gt_sys_spans.append(self._ax_sys.axvspan(x0, x1, **self._gt_span_style))

        # Clear + redraw WARMUP spans (system-wide, across all axes)
        for sp in self._warmup_spans:
            sp.remove()
        self._warmup_spans = []
        if self.show_warmup_span:
            runs = self._contiguous_true_runs(xs, list(self._warm))
            for x0, x1 in runs:
                # shade on every axis so it reads immediately
                for m in self._model_names:
                    self._warmup_spans.append(self._axes_val[m].axvspan(x0, x1, alpha=0.06, linewidth=0.0, color="gray"))
                    self._warmup_spans.append(self._axes_anom[m].axvspan(x0, x1, alpha=0.06, linewidth=0.0, color="gray"))
                if self._ax_sys is not None:
                    self._warmup_spans.append(self._ax_sys.axvspan(x0, x1, alpha=0.06, linewidth=0.0, color="gray"))

        # Ensure legend includes a representative "value(s)" handle (once lines exist)
        if self._legend is not None and self._model_names:
            m0 = self._model_names[0]
            any_val_line = next(iter(self._l_val.get(m0, {}).values()), None)
            if any_val_line is not None:
                gt_handle = Line2D([0], [0], **{k: v for k, v in self._gt_style.items() if k != "alpha"})
                gt_handle.set_alpha(self._gt_style.get("alpha", 1.0))
                gt_handle.set_label("ground truth anomaly")

                handles = [any_val_line, self._l_raw[m0], self._l_lik[m0], gt_handle, self._sys_line]
                labels = ["value(s)", "Anomaly score", "Anomaly probability", "ground truth anomaly", "system alert"]
                self._legend.remove()
                self._legend = self._fig.legend(
                    handles,
                    labels,
                    loc="upper right",
                    frameon=True,
                    fancybox=True,
                    framealpha=0.90,
                )

        # update system alert (filled band + step)
        ys = list(self._alert)
        self._sys_line.set_data(xs, ys)
        # remove old fill, draw new fill (simple + very legible)
        if self._sys_fill is not None:
            self._sys_fill.remove()
        self._sys_fill = self._ax_sys.fill_between(xs, 0.0, ys, step="post", alpha=0.35, zorder=0.2)

        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()

        # optionally record a frame for README GIFs
        self._maybe_record_frame()

        plt.pause(0.001)
