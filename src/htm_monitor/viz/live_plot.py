# src/htm_monitor/viz/live_plot.py

from __future__ import annotations

import numbers
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, Deque, Dict, List, Mapping, Optional, Set, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter, NullLocator


PlotX = Union[int, datetime]


@dataclass
class LivePlot:
    window: int = 1000
    refresh_every: int = 50
    gt_timestamps: Optional[Set[str]] = None  # timestamps with known anomalies
    timestamp_key: str = "timestamp"
    show_warmup_span: bool = True
    # Optional fixed y-limits for value panels, keyed by model name.
    # Example: {"demand_model": (11000.0, 46000.0), "net_generation_model": (2500.0, 38000.0)}
    # If provided, these are used instead of auto/freezing from the first live window.
    value_y_lims: Optional[Dict[str, Tuple[float, float]]] = None
    # Which per-model score to plot on the anomaly panel (must be in [0,1] for the y-limits).
    # Recommended: "anomaly_probability" (aka "p").
    plot_score_key: str = "anomaly_probability"
    # Optional: record frames for README GIFs.
    record_dir: Optional[str] = None
    record_every: int = 1
    record_dpi: int = 140
    show_raw_score: bool = True
    show_group_panel: bool = False
    max_label_len: int = 14
    model_units: Optional[Dict[str, str]] = None
    model_label_colors: Optional[Dict[str, str]] = None
    model_group_membership: Optional[Dict[str, str]] = None

    @staticmethod
    def _parse_plot_ts(x: Optional[str]) -> Optional[datetime]:
        if x is None:
            return None
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H"):
            try:
                return datetime.strptime(x, fmt)
            except Exception:
                continue
        return None

    @staticmethod
    def _infer_plot_step(xs: List[PlotX]) -> Union[int, timedelta]:
        """
        Infer one x-step for end-capping spans.
        - integer axis -> int step
        - datetime axis -> timedelta step
        """
        if len(xs) < 2:
            return timedelta(hours=1) if xs and isinstance(xs[0], datetime) else 1

        dx = xs[1] - xs[0]
        if isinstance(dx, timedelta):
            return dx if dx > timedelta(0) else timedelta(hours=1)
        if isinstance(dx, (int, float)):
            return int(dx) if dx > 0 else 1
        return timedelta(hours=1) if isinstance(xs[0], datetime) else 1

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
        # predictive-warning context
        self._warn_start_t: Optional[int] = None
        self._warn_end_t: Optional[int] = None
        self._failure_t: Optional[int] = None
        # per-model series
        self._model_names: List[str] = []
        # values: model -> feature -> series
        self._val: Dict[str, Dict[str, Deque[float]]] = {}
        self._val_features: Dict[str, List[str]] = {}
        self._raw: Dict[str, Deque[float]] = {}
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
        self._value_ylim: Dict[str, Optional[Tuple[float, float]]] = {}

        self._sys_fill = None
        self._sys_line = None
        self._legend = None
        # figure-level label text handles (stable layout; avoids axes-space clipping/overlap)
        self._signal_label_texts: Dict[str, Any] = {}
        self._sys_label_text = None
        self._label_x_fig: Optional[float] = None

        # vline handles we explicitly create/remove (no try/except)
        self._hot_spans: Dict[str, List[Any]] = {}
        self._warmup_spans: List[Any] = []
        self._warning_spans: List[Any] = []
        self._failure_lines: List[Any] = []
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
        # Labeled anomaly is shown as a red span, not a dotted line.

        # Highly visible GT band (recommended for demo)
        self._gt_span_style: Dict[str, Any] = dict(
            facecolor="red",
            alpha=0.10,
            linewidth=0.0,
            zorder=1,   # behind data lines
        )
        self._warning_span_style: Dict[str, Any] = dict(
            facecolor="magenta",
            alpha=0.10,
            linewidth=0.0,
            zorder=0.4,
        )
        self._failure_line_style: Dict[str, Any] = dict(
            color="magenta",
            linewidth=2.5,
            linestyle=":",
            alpha=0.95,
            zorder=8,
        )
        # keep hot spans effectively off; they were cluttering the story and competing with labels
        self._hot_span_style: Dict[str, Any] = dict(facecolor="black", alpha=0.0, linewidth=0.0, zorder=0.5)
        plt.ion()

    def _init_model_series(self, model_names: List[str]) -> None:
        self._model_names = list(model_names)
        for m in self._model_names:
            self._val[m] = {}
            self._val_features[m] = []
            self._l_val[m] = {}
            self._raw[m] = deque(maxlen=self.window)
            self._gt[m] = deque(maxlen=self.window)
            self._gt_spans[m] = []
            self._hot[m] = deque(maxlen=self.window)
            self._hot_spans[m] = []
            preset = None
            if isinstance(self.value_y_lims, dict):
                preset = self.value_y_lims.get(m)
            self._value_ylim[m] = preset

    def _display_name(self, model_name: str) -> str:
        # Save space: "jumpsdown_model" -> "jumpsdown"
        name = model_name.removesuffix("_model")
        max_len = max(1, int(self.max_label_len))
        if len(name) <= max_len:
            return name
        if max_len <= 3:
            return name[:max_len]
        return name[: max_len - 3] + "..."

    def _display_label(self, model_name: str) -> str:
        name = self._display_name(model_name)
        unit = None
        if isinstance(self.model_units, dict):
            unit = self.model_units.get(model_name)
        if isinstance(unit, str) and unit.strip():
            return f"{name}\n({unit.strip()})"
        return name

    def _group_palette(self) -> List[str]:
        # Avoid black (signals) and blue (anomaly score)
        return [
            "tab:green",
            "tab:red",
            "tab:orange",
            "tab:purple",
            "tab:brown",
            "tab:pink",
            "tab:olive",
            "tab:cyan",
        ]

    def _model_label_color(self, model_name: str) -> str:
        """
        Configurable label color lookup.
        Resolution order:
          1) exact model name, e.g. "demand_model"
          2) base name without "_model", e.g. "demand"
          3) auto-color by group membership, if provided
          4) default black
        """
        if not isinstance(self.model_label_colors, dict):
            colors = {}
        else:
            colors = self.model_label_colors

        base = model_name.removesuffix("_model")
        if model_name in colors:
            return str(colors[model_name])
        if base in colors:
            return str(colors[base])

        if isinstance(self.model_group_membership, dict):
            group = self.model_group_membership.get(model_name)
            if group is not None:
                groups_sorted = sorted(set(str(g) for g in self.model_group_membership.values()))
                palette = self._group_palette()
                if groups_sorted:
                    idx = groups_sorted.index(str(group)) % len(palette)
                    return palette[idx]
        return "black"

    @staticmethod
    def _contiguous_true_runs(xs: List[PlotX], flags: List[float]) -> List[Tuple[PlotX, PlotX]]:
        """
        Return runs (x_start, x_end_exclusive) where flags are truthy.
        Works for either integer timestep x-values or datetime x-values.
        """
        runs: List[Tuple[PlotX, PlotX]] = []
        if not xs or not flags:
            return runs
        step = LivePlot._infer_plot_step(xs)

        start: Optional[PlotX] = None
        for x, f in zip(xs, flags):
            if f and start is None:
                start = x
            elif (not f) and start is not None:
                runs.append((start, x))
                start = None
        if start is not None:
            runs.append((start, xs[-1] + step))
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
        # Layout: (value, anom) * n, spacer, system alert
        nrows = (2 * n) + 2
        # Size the figure for legibility. GIFs are usually viewed smaller than the live window.
        # Tune height by number of models; cap so it doesn’t get absurd for many models.
        display_names = [self._display_label(m).replace("\n", " ") for m in self._model_names]
        # Base the reserved label column on the imposed cap, not the current text,
        # so layout is safe for any allowed label.
        max_display_len = int(self.max_label_len)
        # Smaller reserved label column now that labels are figure-level text,
        # but still wide enough for the capped two-line labels.
        left_margin = max(0.11, min(0.15, 0.035 + 0.0055 * max_display_len))
        alert_label_x = -(0.050 + 0.0016 * max_display_len)
        fig_w = max(12.0, 10.5 + 0.10 * max_display_len)

        fig_h = min(4.4 + (2.2 * n), 16.5)
        self._fig = plt.figure(figsize=(fig_w, fig_h))
        self._fig.patch.set_facecolor("white")

        gs = self._fig.add_gridspec(
            nrows,
            1,
            height_ratios=[3, 1] * n + [0.45, 1.4],
            hspace=0.10,
        )

        first_ax = None
        for i, m in enumerate(self._model_names):
            axv = self._fig.add_subplot(gs[2 * i, 0], sharex=first_ax)
            axa = self._fig.add_subplot(gs[2 * i + 1, 0], sharex=first_ax)
            if first_ax is None:
                first_ax = axv

            self._axes_val[m] = axv
            self._axes_anom[m] = axa

            (lr,) = axa.plot([], [], label="anomaly score", linewidth=1.2, color="tab:blue")

            lr.set_visible(bool(self.show_raw_score))
            self._l_raw[m] = lr

            # We draw labels at the figure level after layout is finalized.
            axv.set_ylabel("")
            axv.grid(True, alpha=0.3)
            axa.grid(True, alpha=0.3)
            # Let matplotlib choose plain numeric formatting now that we have enough horizontal room.
            sf = ScalarFormatter(useMathText=False)
            sf.set_scientific(False)
            sf.set_useOffset(False)
            axv.yaxis.set_major_formatter(sf)

            # If fixed limits were supplied, apply them up front so the axis never "learns"
            # a too-small range from the early part of the stream.
            preset_ylim = self._value_ylim.get(m)
            if preset_ylim is not None:
                lo, hi = preset_ylim
                axv.set_ylim(lo, hi)
            # Raw anomaly panel: keep it visually minimal.
            axa.set_ylim(-0.05, 1.05)
            axa.set_yticks([])
            axa.yaxis.set_major_locator(NullLocator())

            # Hide x tick labels everywhere except the very bottom system panel
            axv.tick_params(labelbottom=False)
            axa.tick_params(labelbottom=False)

        # spacer axis: share x, but hide everything (acts like a gap)
        self._ax_spacer = self._fig.add_subplot(gs[-2, 0], sharex=first_ax)
        self._ax_spacer.set_facecolor("white")
        self._ax_spacer.set_xticks([])
        self._ax_spacer.set_yticks([])
        self._ax_spacer.tick_params(labelbottom=False, bottom=False, left=False)
        for side in ("left", "right", "bottom"):
            self._ax_spacer.spines[side].set_visible(False)
        self._ax_spacer.spines["top"].set_visible(False)

        self._ax_sys = self._fig.add_subplot(gs[-1, 0], sharex=first_ax)
        self._ax_sys.set_facecolor("#f5f5f5")

        self._ax_sys.set_ylim(-0.05, 1.05)
        self._ax_sys.grid(True, alpha=0.3)
        self._ax_sys.set_xlabel("")
        self._ax_sys.set_yticks([])
        self._ax_sys.yaxis.set_major_locator(NullLocator())
        self._ax_sys.set_ylabel("")
        self._ax_sys.spines["top"].set_visible(False)

        locator = mdates.AutoDateLocator(minticks=6, maxticks=10)
        formatter = mdates.ConciseDateFormatter(locator)
        self._ax_sys.xaxis.set_major_locator(locator)
        self._ax_sys.xaxis.set_major_formatter(formatter)
        (self._sys_line,) = self._ax_sys.step(
            [], [], where="post", label="system anomaly", color="red", linewidth=2.5
        )
        self._fig.suptitle("HTM Monitor", y=0.99, fontsize=14)

        # Single global legend (clean, non-crowded)
        # We add a representative "value(s)" handle later once value lines exist.
        alert_legend_handle = Line2D([0], [0], color="red", linewidth=2.5)
        handles = []
        labels = []
        if self.show_raw_score:
            handles.append(lr)
            labels.append("anomaly score")
        handles.append(alert_legend_handle)
        labels.append("system anomaly")
        self._legend = self._fig.legend(
            handles,
            labels,
            loc="upper right",
            frameon=True,
            fancybox=True,
            framealpha=0.90,
        )

        # Give the plot a reserved left label column.
        self._fig.subplots_adjust(left=left_margin, right=0.99, top=0.96, bottom=0.06)

        # Figure-space label column: stable, deterministic, and independent of axes width.
        self._label_x_fig = 0.015
        self._signal_label_texts.clear()
        for m in self._model_names:
            axv = self._axes_val[m]
            bbox = axv.get_position()
            y_center = 0.5 * (bbox.y0 + bbox.y1)
            self._signal_label_texts[m] = self._fig.text(
                self._label_x_fig,
                y_center,
                self._display_label(m),
                ha="left",
                va="center",
                multialignment="center",
                color=self._model_label_color(m),
            )

        sys_bbox = self._ax_sys.get_position()
        sys_y_center = 0.5 * (sys_bbox.y0 + sys_bbox.y1)
        self._sys_label_text = self._fig.text(
            self._label_x_fig,
            sys_y_center,
            "System Anomaly",
            ha="left",
            va="center",
            color="black",
        )

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

    def save_snapshot(self, path: str, *, dpi: Optional[int] = None) -> None:
        """
        Save the current live figure as a static image.
        Useful for small demo runs where the final state is presentation-worthy.
        """
        if self._fig is None:
            return
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        self._fig.savefig(
            out,
            dpi=int(dpi or self.record_dpi),
            bbox_inches="tight",
            pad_inches=0.10,
            facecolor=self._fig.get_facecolor(),
        )

    @staticmethod
    def _safe_minmax(vals: List[float]) -> Optional[Tuple[float, float]]:
        clean: List[float] = []
        for v in vals:
            if isinstance(v, numbers.Real):
                fv = float(v)
                if fv == fv:  # not NaN
                    clean.append(fv)
        if not clean:
            return None
        return (min(clean), max(clean))

    @staticmethod
    def _pad_limits(vmin: float, vmax: float) -> Tuple[float, float]:
        if vmax < vmin:
            vmin, vmax = vmax, vmin
        span = vmax - vmin
        if span <= 0.0:
            pad = max(1.0, abs(vmin) * 0.05, 0.5)
        else:
            pad = max(span * 0.08, 0.5)
        return (vmin - pad, vmax + pad)

    def _maybe_freeze_value_ylim(self, model_name: str) -> None:
        """
        Freeze each model's value-axis limits once enough real values are present.
        This keeps the live demo visually stable across screenshots and prevents
        y-axis rescaling from changing the apparent magnitude of the same pattern.
        """
        if self._value_ylim.get(model_name) is not None:
            return

        vals: List[float] = []
        for f in self._val_features.get(model_name, []):
            vals.extend(list(self._val[model_name].get(f, [])))

        mm = self._safe_minmax(vals)
        if mm is None:
            return

        vmin, vmax = mm
        # wait until we have at least a modest range of real observations;
        # otherwise the first few points can freeze a silly axis.
        if len(vals) < 24:
            return

        lo, hi = self._pad_limits(vmin, vmax)
        self._value_ylim[model_name] = (lo, hi)
        axv = self._axes_val[model_name]
        axv.set_ylim(lo, hi)

    def _reset_and_rebuild(self, model_names: List[str]) -> None:
        if self._fig is not None:
            plt.close(self._fig)
        self._t.clear()
        self._ts.clear()
        self._alert.clear()
        self._warm.clear()
        self._gt_sys.clear()
        self._warn_start_t = None
        self._warn_end_t = None
        self._failure_t = None
        self._model_names = []
        self._val.clear()
        self._val_features.clear()
        self._raw.clear()
        self._gt.clear()
        self._hot.clear()
        self._axes_val.clear()
        self._axes_anom.clear()
        self._ax_sys = None
        self._ax_spacer = None
        self._l_val.clear()
        self._l_raw.clear()
        self._value_ylim.clear()
        self._hot_spans.clear()
        self._sys_fill = None
        self._sys_line = None
        self._signal_label_texts.clear()
        self._sys_label_text = None
        self._legend = None
        self._warmup_spans = []
        self._warning_spans = []
        self._failure_lines = []
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
        gt_system = row.get("gt_system")
        if not isinstance(gt_system, (set, frozenset)):
            gt_system = None

        warning_window = row.get("warning_window")
        if isinstance(warning_window, Mapping):
            ws = warning_window.get("start_t")
            we = warning_window.get("end_t")
            self._warn_start_t = int(ws) if isinstance(ws, (int, float)) else self._warn_start_t
            self._warn_end_t = int(we) if isinstance(we, (int, float)) else self._warn_end_t

        failure_t = row.get("failure_t")
        if isinstance(failure_t, (int, float)):
            self._failure_t = int(failure_t)

        alert = 1.0 if (isinstance(result, Mapping) and result.get("alert")) else 0.0
        hot_by_model = result.get("hot_by_model") if isinstance(result, Mapping) else None
        if not isinstance(hot_by_model, Mapping):
            hot_by_model = None

        in_warmup = bool(row.get("in_warmup", False))

        self._t.append(int(t))
        self._ts.append(str(ts) if ts is not None else None)
        self._alert.append(float(alert))
        self._warm.append(1.0 if in_warmup else 0.0)
        sys_gt_flag = bool(gt_system)
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
                        (ln,) = self._axes_val[m].plot([], [], label="_nolegend_", color="black")
                        self._l_val[m][f] = ln

            raw = out.get("raw")
            # Plot score for the non-technical story:
            # - Primary: whatever plot_score_key says (default: anomaly_probability)
            # - Back-compat fallbacks: anomaly_probability, p, likelihood

            if gt_by_model is not None:
                gt_flag = bool(gt_by_model.get(m))
            else:
                gt_flag = False

            # append per-feature values
            for f in self._val_features.get(m, []):
                vv = vmap.get(f)
                v_f = float(vv) if isinstance(vv, numbers.Real) else float("nan")
                self._val[m][f].append(v_f)
            self._raw[m].append(raw)
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

        parsed_ts = [self._parse_plot_ts(x) for x in self._ts]
        use_time_axis = bool(parsed_ts) and all(x is not None for x in parsed_ts)
        xs: List[PlotX] = parsed_ts if use_time_axis else list(self._t)

        if xs:
            if use_time_axis:
                x1 = xs[-1]
                x0 = xs[0]
            else:
                x1 = xs[-1]
                x0 = x1 - (int(self.window) - 1)

            if x0 == x1:
                if use_time_axis:
                    x1 = x1 + timedelta(hours=1)
                else:
                    x1 = x1 + 1

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

            self._maybe_freeze_value_ylim(m)

            # clear + redraw GT spans (far more visible than vlines)
            for sp in self._gt_spans[m]:
                sp.remove()
            self._gt_spans[m] = []
            for x, g in zip(xs, self._gt[m]):
                if g:
                    if use_time_axis:
                        x0 = x - timedelta(minutes=30)
                        x1 = x + timedelta(minutes=30)
                    else:
                        x0, x1 = x - 0.5, x + 0.5
                    self._gt_spans[m].append(axv.axvspan(x0, x1, **self._gt_span_style))
                    self._gt_spans[m].append(axa.axvspan(x0, x1, **self._gt_span_style))

            # clear + redraw HOT spans (simple + obvious)
            for sp in self._hot_spans[m]:
                sp.remove()
            self._hot_spans[m] = []
            runs = self._contiguous_true_runs(xs, list(self._hot[m]))
            for x0, x1 in runs:
                # highlight on the value axis so it's visible even if anom axis is busy
                self._hot_spans[m].append(axv.axvspan(x0, x1, **self._hot_span_style))

            if self._value_ylim.get(m) is None:
                axv.relim()
                # IMPORTANT: NEVER let autoscale touch x (we control x via hard sliding window).
                axv.autoscale_view(scalex=False, scaley=True)
            else:
                lo, hi = self._value_ylim[m]
                axv.set_ylim(lo, hi)

            # IMPORTANT: DO NOT autoscale anomaly axes (it causes the “weird” looking panels).
            # Keep anomaly y fixed; x is shared already.
            # axa.relim(); axa.autoscale_view()  # intentionally disabled
            axa.set_ylim(-0.05, 1.05)
            axa.set_yticks([])
            axa.yaxis.set_major_locator(NullLocator())

        # Clear + redraw SYSTEM GT spans (pink band) on system axis
        for sp in self._gt_sys_spans:
            sp.remove()
        self._gt_sys_spans = []
        if self._ax_sys is not None:
            for x, g in zip(xs, list(self._gt_sys)):
                if g:
                    if use_time_axis:
                        x0 = x - timedelta(minutes=30)
                        x1 = x + timedelta(minutes=30)
                    else:
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

        # Clear + redraw predictive warning window spans
        for sp in self._warning_spans:
            sp.remove()
        self._warning_spans = []
        if self._warn_start_t is not None and self._warn_end_t is not None and xs:
            if use_time_axis:
                t_to_ts = {tt: tsv for tt, tsv in zip(list(self._t), xs)}
                x_left = t_to_ts.get(int(self._warn_start_t))
                x_right = t_to_ts.get(int(self._warn_end_t))
                if x_left is not None and x_right is not None:
                    step = self._infer_plot_step(xs)
                    x_right_excl = x_right + step
                    if x_left < x_right_excl:
                        for m in self._model_names:
                            self._warning_spans.append(self._axes_val[m].axvspan(x_left, x_right_excl, **self._warning_span_style))
                            self._warning_spans.append(self._axes_anom[m].axvspan(x_left, x_right_excl, **self._warning_span_style))
                        if self._ax_sys is not None:
                            self._warning_spans.append(self._ax_sys.axvspan(x_left, x_right_excl, **self._warning_span_style))
            else:
                x_left = max(self._warn_start_t, xs[0])
                x_right = min(self._warn_end_t, xs[-1] + 1)
                if x_left < x_right:
                    for m in self._model_names:
                        self._warning_spans.append(self._axes_val[m].axvspan(x_left, x_right, **self._warning_span_style))
                        self._warning_spans.append(self._axes_anom[m].axvspan(x_left, x_right, **self._warning_span_style))
                    if self._ax_sys is not None:
                        self._warning_spans.append(self._ax_sys.axvspan(x_left, x_right, **self._warning_span_style))

        # Clear + redraw failure markers
        for ln in self._failure_lines:
            ln.remove()
        self._failure_lines = []
        if self._failure_t is not None and xs:
            if use_time_axis:
                t_to_ts = {tt: tsv for tt, tsv in zip(list(self._t), xs)}
                failure_x = t_to_ts.get(int(self._failure_t))
                if failure_x is not None:
                    for m in self._model_names:
                        self._failure_lines.append(self._axes_val[m].axvline(failure_x, **self._failure_line_style))
                        self._failure_lines.append(self._axes_anom[m].axvline(failure_x, **self._failure_line_style))
                    if self._ax_sys is not None:
                        self._failure_lines.append(self._ax_sys.axvline(failure_x, **self._failure_line_style))
            else:
                if xs[0] <= self._failure_t <= (xs[-1] + 1):
                    for m in self._model_names:
                        self._failure_lines.append(self._axes_val[m].axvline(self._failure_t, **self._failure_line_style))
                        self._failure_lines.append(self._axes_anom[m].axvline(self._failure_t, **self._failure_line_style))
                    if self._ax_sys is not None:
                        self._failure_lines.append(self._ax_sys.axvline(self._failure_t, **self._failure_line_style))

        # Ensure legend includes a representative "value(s)" handle (once lines exist)
        if self._legend is not None and self._model_names:
            m0 = self._model_names[0]
            any_val_line = next(iter(self._l_val.get(m0, {}).values()), None)
            if any_val_line is not None:
                alert_legend_handle = Line2D([0], [0], color="red", linewidth=2.5)

                handles = [any_val_line]
                labels = ["signal"]
                if self.show_raw_score:
                    handles.append(self._l_raw[m0])
                    labels.append("anomaly score")
                handles.append(alert_legend_handle)
                labels.append("system anomaly")

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
        # Hide the baseline red line at y=0 so the SYSTEM ALERT panel stays visually empty
        # until an actual alert occurs.
        ys_alert_only = [float(y) if float(y) > 0.0 else float("nan") for y in ys]
        self._sys_line.set_data(xs, ys_alert_only)

        # remove old fill, draw new fill (simple + very legible)
        if self._sys_fill is not None:
            self._sys_fill.remove()
        if any(float(y) > 0.0 for y in ys):
            self._sys_fill = self._ax_sys.fill_between(
                xs,
                0.0,
                ys,
                where=[float(y) > 0.0 for y in ys],
                step="post",
                alpha=0.12,
                color="red",
                zorder=0.2,
            )
        else:
            self._sys_fill = None
        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()

        # optionally record a frame for README GIFs
        self._maybe_record_frame()

        plt.pause(0.001)
