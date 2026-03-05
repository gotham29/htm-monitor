#src/htm-monitor/htm_src/htm_model.py

import math
import logging
from typing import Mapping, Union, Any, Dict, Optional, Tuple, Set

import numpy as np
from htm.bindings.algorithms import TemporalMemory
from htm.bindings.sdr import SDR

from .feature import Feature, separate_time_and_rest
from .anomalylikelihood import AnomalyLikelihood
from .types import HTMType

log = logging.getLogger(__name__)

def _sdr_sparse_tuple(sdr: SDR) -> Tuple[int, ...]:
    return tuple(int(i) for i in sdr.sparse)

def _approx_rdse_resolution(params: Mapping[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    """
    Approximate RDSE resolution from config when possible.
    Returns (resolution, min_val).

    We use common keys:
      - minVal / maxVal / numBuckets
    If missing, returns (None, None).
    """
    min_v = params.get("minVal")
    max_v = params.get("maxVal")
    nb = params.get("numBuckets")
    if isinstance(min_v, (int, float)) and isinstance(max_v, (int, float)) and isinstance(nb, (int, float)):
        nb = float(nb)
        span = float(max_v) - float(min_v)
        if nb > 0 and span > 0:
            return (span / nb, float(min_v))
    return (None, None)

class HTMmodel:
    def __init__(self,
                 features: Mapping[str, Feature],
                 models_params: dict,
                 return_pred_count: bool = False,
                 name: Optional[str] = None):

        self.iteration_ = 0
        self.name = str(name) if name is not None else None
        self._last_al_p: Optional[float] = None
        self._last_al_log: Optional[float] = None
        self.features = features
        self.models_params = models_params
        self.return_pred_count = bool(return_pred_count)
        # Will be set in init_tm()
        self.cells_per_column: Optional[int] = None
        # Cached predictive cells from the *previous* step (for diagnostics without extra TM state mutation).
        self._pred_cells_prev: Tuple[int, ...] = tuple()

        self.encoding_width = sum(feat.encoding_size for feat in self.features.values())
        self.tm = self.init_tm()
        self.al = self.init_alikelihood()

        # utility attributes
        self.single_feature = self.get_single_feature_name()
        self.feature_names = list(self.features.keys())
        # self.features_samples = {f: [] for f in self.feature_names}
        self.feature_timestamp = separate_time_and_rest(self.features.values())[0]

    @staticmethod
    def _clamp_int(x: float, lo: int, hi: int) -> int:
        return int(max(lo, min(hi, int(round(x)))))

    @staticmethod
    def _w_from_date_encoder_param(v: Any) -> int:
        """
        Extract the "w" (active bits) from DateEncoder sub-encoder params.

        We support common shapes used in configs:
          - int/float              -> w
          - dict with key "w"      -> w
          - tuple/list (w, ...)    -> w

        If we can't extract a numeric w, return 0 (no guessing).
        """
        if isinstance(v, (int, float)) and v > 0:
            return int(v)
        if isinstance(v, Mapping):
            w = v.get("w")
            if isinstance(w, (int, float)) and w > 0:
                return int(w)
            return 0
        if isinstance(v, (tuple, list)) and len(v) > 0:
            w0 = v[0]
            if isinstance(w0, (int, float)) and w0 > 0:
                return int(w0)
            return 0
        return 0

    def _expected_active_columns(self) -> int:
        """
        Estimate expected active columns per timestep from Feature params.

        - Numeric/Categoric RDSE: uses params['activeBits'] when present.
        - Datetime DateEncoder: uses sum of enabled sub-encoder widths:
            timeOfDay/weekend/dayOfWeek/holiday/season
          (In Numenta-style encoders, these are typically the "w" active bits.)

        If any piece is missing, we simply skip it (robust, no guessing).
        """
        total = 0
        for feat in self.features.values():
            # If a feature is configured to not contribute to encoding width, don't count it.
            # (This keeps TM scaling consistent when you set timestamp.encode: false.)
            if hasattr(feat, "encode_enabled") and (not bool(getattr(feat, "encode_enabled"))):
                continue
            p = dict(feat.params or {})

            # If feature is not encoded, it contributes 0 active columns.
            if not bool(p.get("encode", True)):
                continue

            if feat.type in (HTMType.Numeric, HTMType.Categoric):
                ab = p.get("activeBits")
                if isinstance(ab, (int, float)) and ab > 0:
                    total += int(ab)
            elif feat.type == HTMType.Datetime:
                for k in ("timeOfDay", "weekend", "dayOfWeek", "holiday", "season"):
                    v = p.get(k)
                    total += self._w_from_date_encoder_param(v)
        return int(total)

    def _scaled_tm_params(self) -> Dict[str, Any]:
        """
        Optionally scale count-like TM params to maintain proportions relative to
        expected active columns per timestep.
        """
        tm_cfg = dict(self.models_params.get("tm", {}) or {})
        scfg = dict(tm_cfg.get("scaling", {}) or {})
        if not bool(scfg.get("enabled", False)):
            return tm_cfg

        base_active = scfg.get("baseActiveColumns", 40)
        if not isinstance(base_active, (int, float)) or float(base_active) <= 0:
            raise ValueError("tm.scaling.baseActiveColumns must be > 0")
        base_active = float(base_active)

        expected_active = float(self._expected_active_columns())
        if expected_active <= 0:
            # Fail loud: scaling enabled but we can't estimate active columns.
            raise ValueError(
                "tm.scaling.enabled is true but expected active columns estimate is 0. "
                "Ensure features specify activeBits (RDSE) and timeOfDay/weekend/dayOfWeek/holiday/season (DateEncoder)."
            )

        s = expected_active / base_active

        clamp = dict(scfg.get("clamp", {}) or {})
        a_min = int(clamp.get("activationThresholdMin", 3))
        m_min = int(clamp.get("minThresholdMin", 2))
        n_min = int(clamp.get("newSynapseCountMin", 4))
        # Upper bounds: cannot exceed expected_active for these counts.
        hi = max(1, int(round(expected_active)))

        # Scale only the count-like fields.
        tm_cfg["activationThreshold"] = self._clamp_int(float(tm_cfg["activationThreshold"]) * s, a_min, hi)
        tm_cfg["minThreshold"] = self._clamp_int(float(tm_cfg["minThreshold"]) * s, m_min, tm_cfg["activationThreshold"])
        tm_cfg["newSynapseCount"] = self._clamp_int(float(tm_cfg["newSynapseCount"]) * s, n_min, hi)

        log.info(
            f"TM scaling enabled: expected_active={expected_active:.1f} base_active={base_active:.1f} scale={s:.3f} "
            f"-> activationThreshold={tm_cfg['activationThreshold']} minThreshold={tm_cfg['minThreshold']} newSynapseCount={tm_cfg['newSynapseCount']}"
        )
        return tm_cfg

    def init_tm(self) -> TemporalMemory:
        """
        Purpose:
            Init HTMmodel.tm
        Inputs:
            HTMmodel.models_params
                type: dict
                meaning: HTM hyperparams (user-provided in config.yaml)
        Outputs:
            HTMmodel.tm
                type: htm.core.TemporalMemory
                meaning: HTM native alg that activates & depolarizes activeColumns' cells within HTM region
        """

        column_dimensions = self.encoding_width
        tm_cfg = self._scaled_tm_params()
        # store for diagnostics (cell->column mapping)
        cpc = tm_cfg.get("cellsPerColumn")
        self.cells_per_column = int(cpc) if isinstance(cpc, (int, float)) else None
        return TemporalMemory(
            columnDimensions=(column_dimensions,),
            cellsPerColumn=tm_cfg["cellsPerColumn"],
            activationThreshold=tm_cfg["activationThreshold"],
            initialPermanence=tm_cfg["initialPerm"],
            connectedPermanence=tm_cfg["permanenceConnected"],
            minThreshold=tm_cfg["minThreshold"],
            maxNewSynapseCount=tm_cfg["newSynapseCount"],
            permanenceIncrement=tm_cfg["permanenceInc"],
            permanenceDecrement=tm_cfg["permanenceDec"],
            predictedSegmentDecrement=tm_cfg["predictedSegmentDecrement"],
            maxSegmentsPerCell=tm_cfg["maxSegmentsPerCell"],
            maxSynapsesPerSegment=tm_cfg["maxSynapsesPerSegment"],
            seed=42)

    def init_alikelihood(self) -> AnomalyLikelihood:
        """
        Purpose:
            Init HTMModel.al
        Inputs:
            HTMmodel.models_params['anomaly_likelihood']
                type: dict
                meaning: hyperparams for alikelihood
        Outputs:
            HTMmodel.al
                type: nupic.AnomalyLikelihood
                meaning: HTM native alg that for postprocessing raw anomaly scores
        """

        al_cfg = self.models_params.get("anomaly_likelihood", {}) or {}

        # Preferred explicit config (matches your new htm_defaults.yaml)
        learning_period = al_cfg.get("learningPeriod")
        estimation_samples = al_cfg.get("estimationSamples")

        # Backward-compat: allow older configs that still specify probationaryPeriod
        probationary = al_cfg.get("probationaryPeriod")
        if (learning_period is None or estimation_samples is None) and probationary is not None:
            learning_period = int(math.floor(float(probationary) / 2.0))
            estimation_samples = int(probationary) - int(learning_period)

        # Final defaults (only if neither explicit nor probationary provided)
        if learning_period is None:
            learning_period = 288
        if estimation_samples is None:
            estimation_samples = 100

        historic_window = int(al_cfg.get("historicWindowSize", 8640))
        reestimation_period = int(al_cfg.get("reestimationPeriod", 100))
        averaging_window = int(al_cfg.get("averagingWindow", 10))

        # Basic validation to catch silent misconfig
        learning_period = int(learning_period)
        estimation_samples = int(estimation_samples)
        if learning_period <= 0:
            raise ValueError("anomaly_likelihood.learningPeriod must be > 0")
        if estimation_samples <= 0:
            raise ValueError("anomaly_likelihood.estimationSamples must be > 0")
        if historic_window < estimation_samples:
            raise ValueError("anomaly_likelihood.historicWindowSize must be >= estimationSamples")
        if averaging_window <= 0:
            raise ValueError("anomaly_likelihood.averagingWindow must be > 0")

        return AnomalyLikelihood(
            learningPeriod=learning_period,
            estimationSamples=estimation_samples,
            historicWindowSize=historic_window,
            reestimationPeriod=reestimation_period,
            averagingWindow=averaging_window,
        )

    def _likelihood_value_feature(self) -> Optional[str]:
        """
        Select a stable, non-timestamp feature name to pass as `value` into
        AnomalyLikelihood.anomalyProbability(...).

        - If there is exactly one non-time feature, use it.
        - Otherwise, choose the first non-time feature (deterministic based on Feature order).
        - If no non-time features exist, return None.
        """
        _, non_time_feature_names = separate_time_and_rest(self.features.values())
        if len(non_time_feature_names) == 0:
            return None
        if len(non_time_feature_names) == 1:
            return non_time_feature_names[0]
        # Multi-feature model: pick the first non-time feature deterministically
        return non_time_feature_names[0]

    def get_alikelihood(
        self,
        value: float,
        anomaly_score: float,
        timestamp,
        *,
        timestep: Optional[int] = None,
        log_every: int = 200,
    ) -> float:
        """
        Purpose:
            Return anomaly likelihood for given input data point
        Inputs:
            value:
                type: float
                meaning: current input data point
            anomaly_score:
                type: float
                meaning: HTM raw anomaly score output from HTMmodel.tm
            timestamp:
                type: datetime or int
                meaning: timestamp feature (or HTMmodel.iteration_ if no timestamp)
        Outputs:
            anomalyLikelihood:
                type: float
                meaning: likelihood value (used to classify data as anomalous or not)
        """
        p = self.al.anomalyProbability(value, anomaly_score, timestamp)
        log_score = self.al.computeLogLikelihood(p)
        self._last_al_p = float(p)
        self._last_al_log = float(log_score)

        # Sparse debug trace (does not change outputs)
        if (
            timestep is not None
            and isinstance(log_every, int)
            and log_every > 0
            and (timestep % log_every) == 0
        ):
            st = self.al.debug_state() if hasattr(self.al, "debug_state") else {}
            log.debug(
                "AL_TRACE model=%s t=%s ts=%s value=%s raw=%s p=%s log=%s state=%s",
                (self.name or "?"),
                timestep,
                timestamp,
                value,
                anomaly_score,
                p,
                log_score,
                st,
            )

        return log_score

    def last_anomaly_probability(self) -> Optional[float]:
        return self._last_al_p

    def last_log_likelihood(self) -> Optional[float]:
        return self._last_al_log

    def get_single_feature_name(self) -> Union[None, str]:
        """
        If the model has a single feature beside the timestamp, will return the name of that feature.
        Otherwise, returns None.
        """
        _, non_time_feature_names = separate_time_and_rest(self.features.values())
        if len(non_time_feature_names) == 1:
            return non_time_feature_names[0]
        else:
            return None

    def get_encoding(self, features_data: Mapping) -> SDR:
        """
        Purpose:
            Build total concatenated encoding from all features' encoders -- for input to SP
        Inputs:
            features_data
                type: dict
                meaning: current data for each feature
            HTMmodel.feature_encoders
                type: dict
                meaning: encoder objects for each feature
            HTMmodel.timestamp_config
                type: dict
                meaning: params for timestep encoding (user-provided in config.yaml)
            HTMmodel.encoding_width
                type: int
                meaning: size in bits of total concatenated encoder (input to SP)
        Outputs:
            encoding
                type: nup.core.SDR
                meaning: total concatenated encoding -- input to SP
        """

        # Get encodings for all features
        all_encodings = [SDR(0)]
        for name, feat in self.features.items():
            if hasattr(feat, "encode_enabled") and (not feat.encode_enabled):
                continue
            all_encodings.append(feat.encode(features_data[name]))

        # Combine all features encodings into one for Spatial Pooling
        encoding = SDR(self.encoding_width).concatenate(all_encodings)
        return encoding

    def _feature_encodings(self, features_data: Mapping[str, Any]) -> Dict[str, SDR]:
        """
        Encode each feature separately (used for diagnostics).
        """
        out: Dict[str, SDR] = {}
        for name, feat in self.features.items():
            if hasattr(feat, "encode_enabled") and (not feat.encode_enabled):
                continue
            out[name] = feat.encode(features_data[name])
        return out

    @staticmethod
    def _cells_to_columns(cells: Tuple[int, ...], cells_per_column: int) -> Tuple[int, ...]:
        """
        Map cell indices -> unique column indices.
        cell_id = col*cellsPerColumn + cellWithinCol  => col = cell_id // cellsPerColumn
        """
        if cells_per_column <= 0:
            return tuple()
        cols: Set[int] = set()
        for c in cells:
            ci = int(c)
            if ci < 0:
                continue
            cols.add(ci // cells_per_column)
        return tuple(sorted(cols))

    def run_with_diagnostics(
        self,
        features_data: Mapping[str, Any],
        timestep: int,
        learn: bool,
    ) -> Tuple[float, float, Optional[float], Dict[str, Any]]:
        """
        Same semantics as run(), but returns a diag payload:
          diag["encodings"][feat] = {value, sparse, active_bits, resolution, min_val, approx_bucket}
          diag["tm"] = {pred_cells_prior, active_cells, winner_cells, pred_cells_after}
        """
        # select only relevant features
        features_data = {name: features_data[name] for name in self.feature_names}

        # --- per-feature encodings (for evidence) ---
        per_feat = self._feature_encodings(features_data)

        # Concatenate into active columns (same as get_encoding() path)
        # IMPORTANT:
        # _feature_encodings() omits encode-disabled features (e.g. timestamp.encode:false),
        # so we must NOT assume per_feat has entries for every feature in self.features.
        all_encodings = [SDR(0)]
        for name, feat in self.features.items():
            if hasattr(feat, "encode_enabled") and (not feat.encode_enabled):
                continue
            sdr = per_feat.get(name)
            if sdr is None:
                # This should never happen unless configs drift (model.features includes a feature that
                # wasn't encoded). Fail loud with context instead of KeyError.
                raise KeyError(
                    f"Missing per-feature encoding for '{name}' in run_with_diagnostics(). "
                    f"Likely cause: feature encode:false or model/features mismatch. "
                    f"model={self.name or '?'} feature_names={list(self.features.keys())}"
                )
            all_encodings.append(sdr)
        encoding = SDR(self.encoding_width).concatenate(all_encodings)
        active_columns = encoding
        # snapshot active columns (sparse) for column-level diagnostics
        active_cols_sparse = _sdr_sparse_tuple(active_columns)
        active_cols_count = int(active_columns.getSum()) if hasattr(active_columns, "getSum") else len(active_cols_sparse)

        # --- predictive cells PRIOR to compute (diagnostics) ---
        # IMPORTANT: do NOT call activateDendrites() here; that mutates TM internal state.
        # We instead use the cached predictive-cells snapshot from the previous timestep.
        pred_prior = tuple(self._pred_cells_prev)

        # prediction density (optional) WITHOUT mutating TM state:
        # define as (# predicted cells from prior state) / (# active columns in current input)
        pred_count = None
        if self.return_pred_count:
            n_pred_cells = float(len(pred_prior))
            n_active_cols = float(active_columns.getSum()) if active_columns is not None else 0.0
            pred_count = 0.0 if n_active_cols == 0.0 else (n_pred_cells / n_active_cols)

        # TM compute
        self.tm.compute(active_columns, learn=learn)
        anomaly_score = self.tm.anomaly

        # cells AFTER compute (for counts / debugging)
        # Snapshot TM state immediately after compute.
        active_cells = _sdr_sparse_tuple(self.tm.getActiveCells())
        winner_cells = _sdr_sparse_tuple(self.tm.getWinnerCells())

        # Update predictive cells for *next* timestep (and cache it for next diagnostics call).
        # We explicitly keep learn=False here: activating dendrites should not learn.
        self.tm.activateDendrites(learn=False)
        self._pred_cells_prev = _sdr_sparse_tuple(self.tm.getPredictiveCells())

        # --- column-level prediction + bursting diagnostics ---
        pred_cols_prior_count = None
        pred_col_hit_rate = None
        burst_frac = None
        if isinstance(self.cells_per_column, int) and self.cells_per_column > 0:
            pred_cols_prior = self._cells_to_columns(pred_prior, self.cells_per_column)
            pred_cols_prior_count = len(pred_cols_prior)
            # Column hit: do predicted columns cover active columns?
            if active_cols_count > 0:
                pred_col_hit = len(set(pred_cols_prior) & set(active_cols_sparse))
                pred_col_hit_rate = float(pred_col_hit) / float(active_cols_count)
                # Burst proxy: fraction of cells active per active column
                # predicted regime ~1/cpc; full burst ~1.0
                burst_frac = float(len(active_cells)) / float(active_cols_count * self.cells_per_column)
            else:
                pred_col_hit_rate = 0.0
                burst_frac = 0.0

        # AnomalyLikelihood
        f1 = self._likelihood_value_feature()
        if self.feature_timestamp:
            # Ensure datetime-like timestamps are parsed even if timestamp.encode is false.
            timestamp = self.features[self.feature_timestamp].parse(features_data[self.feature_timestamp])
        else:
            timestamp = self.iteration_
        if f1 is None:
            raise ValueError("No non-timestamp feature available for anomaly likelihood value")

        anomaly_likelihood = self.get_alikelihood(
            value=features_data[f1],
            timestamp=timestamp,
            anomaly_score=anomaly_score,
            timestep=timestep,
        )

        # Increment iteration
        self.iteration_ += 1

        # --- diag payload ---
        enc_diag: Dict[str, Any] = {}
        for fname, sdr in per_feat.items():
            feat = self.features[fname]
            v = features_data.get(fname)
            v_f = float(v) if isinstance(v, (int, float)) else None

            resolution = None
            min_val = None
            approx_bucket = None

            if feat.type in (HTMType.Numeric, HTMType.Categoric):
                r, mn = _approx_rdse_resolution(feat.params or {})
                resolution = r
                min_val = mn
                if v_f is not None and resolution is not None and min_val is not None and resolution > 0:
                    approx_bucket = int(math.floor((v_f - min_val) / resolution))

            enc_diag[fname] = {
                "value": v_f,
                "sdr": sdr,  # keep as SDR; writer will extract sparse
                "active_bits": int(sdr.getSum()) if hasattr(sdr, "getSum") else len(_sdr_sparse_tuple(sdr)),
                "resolution": resolution,
                "min_val": min_val,
                "approx_bucket": approx_bucket,
            }

        diag = {
            "encodings": enc_diag,
            "tm": {
                "pred_cells_prior": pred_prior,
                "active_cells": active_cells,
                "winner_cells": winner_cells,
                # Added: column-level + burst evidence
                "active_cols": int(active_cols_count),
                "pred_cols_prior_count": pred_cols_prior_count,
                "pred_col_hit_rate": pred_col_hit_rate,
                "burst_frac": burst_frac,
            },
        }

        return anomaly_score, anomaly_likelihood, pred_count, diag

    def get_predcount(self, active_columns: SDR) -> float:
        """
        Purpose:
            Get number of predictions made by TM at current timestep
        Inputs:
            HTMmodel.tm
                type: nupic.core.TemporalMemory
                meaning: TemporalMemory component of HTMmodel
            HTMmodel.models_params
                type: dict
                meaning: HTM hyperparams (user-provided in config.yaml)
            active_columns
                type: SDR
                meaning: active columns SDR for the *current* input (used only as a stable denominator)
        Outputs:
            pred_count
                type: float
                meaning: prediction density for the *current timestep*:
                         (# predicted cells from prior state) / (# active columns in current input)
        """
        # Ensure predictive cells are up to date based on *prior* timestep state.
        self.tm.activateDendrites(learn=False)

        # Count number of predicted cells
        n_pred_cells = self.tm.getPredictiveCells().getSum()
        n_active_cols = active_columns.getSum() if active_columns is not None else 0

        # Normalize to number of predictions
        pred_count = 0.0 if n_active_cols == 0 else float(n_pred_cells) / float(n_active_cols)
        return pred_count

    def run(self,
            features_data: Mapping,
            timestep: int,
            learn: bool,
            ) -> (float, float, float, dict):
        """
        Purpose:
            Run HTMmodel -- yielding all outputs & updating model (if 'learn'==True)
        Inputs:
            features_data
                type: dict
                meaning: current data for each feature
            timestep
                type: int
                meaning: current timestep
            learn
                type: bool
                meaning: whether learning is enabled in HTMmodel
        Outputs:
            anomaly_score
                type: float
                meaning: anomaly metric - from HMTModel.tm
            anomaly_likelihood
                type: float
                meaning: anomaly metric - from HMTModel.anomaly_history
            pred_count
                type: float
                meaning: number of predictions made by TM at current timestep (# predicted cells / # active columns)
            steps_predictions
                type: dict
                meaning: predicted feature values for each of 'n_steps_ahead'
        """
        # select only relevant features
        features_data = {name: features_data[name] for name in self.feature_names}

        # ENCODERS
        # Call the encoders to create bit representations for each feature
        encoding = self.get_encoding(features_data)
        active_columns = encoding

        # TEMPORAL MEMORY
        # Get prediction density
        pred_count = self.get_predcount(active_columns) if self.return_pred_count else None
        self.tm.compute(active_columns, learn=learn)
        # Get anomaly metrics
        anomaly_score = self.tm.anomaly
        # Choose feature for AnomalyLikelihood.value (never use timestamp)
        f1 = self._likelihood_value_feature()
        # Get timestamp data if available
        if self.feature_timestamp:
            # Ensure datetime-like timestamps are parsed even if timestamp.encode is false.
            timestamp = self.features[self.feature_timestamp].parse(features_data[self.feature_timestamp])
        else:
            timestamp = self.iteration_
        if f1 is None:
            raise ValueError("No non-timestamp feature available for anomaly likelihood value")

        anomaly_likelihood = self.get_alikelihood(
            value=features_data[f1],
            timestamp=timestamp,
            anomaly_score=anomaly_score,
            timestep=timestep,
        )

        # Do NOT hard-fail on pred_count==0; it can happen legitimately early in learning
        # or in ambiguous regimes. Keep as debug trace only.
        if (
            self.return_pred_count
            and isinstance(pred_count, (int, float))
            and pred_count == 0
            and anomaly_score < 1.0
        ):
            log.debug("pred_count==0 with anomaly_score=%s (model=%s t=%s)", anomaly_score, (self.name or "?"), timestep)

        # Increment iteration
        self.iteration_ += 1

        return anomaly_score, anomaly_likelihood, pred_count