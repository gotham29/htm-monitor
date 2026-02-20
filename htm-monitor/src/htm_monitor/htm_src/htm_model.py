import math
import logging
from typing import Mapping, Union, Any, Dict, Optional

import numpy as np
from htm.bindings.algorithms import TemporalMemory
from htm.bindings.sdr import SDR

from .feature import Feature, separate_time_and_rest
from .anomalylikelihood import AnomalyLikelihood
from .types import HTMType

log = logging.getLogger(__name__)


class HTMmodel:
    def __init__(self,
                 features: Mapping[str, Feature],
                 models_params: dict,
                 return_pred_count: bool = False):

        self.iteration_ = 0
        self.features = features
        self.models_params = models_params
        self.return_pred_count = bool(return_pred_count)
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
            p = dict(feat.params or {})
            if feat.type in (HTMType.Numeric, HTMType.Categoric):
                ab = p.get("activeBits")
                if isinstance(ab, (int, float)) and ab > 0:
                    total += int(ab)
            elif feat.type is HTMType.Datetime:
                for k in ("timeOfDay", "weekend", "dayOfWeek", "holiday", "season"):
                    v = p.get(k)
                    if isinstance(v, (int, float)) and v > 0:
                        total += int(v)
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

    def get_alikelihood(self, value, anomaly_score, timestamp) -> float:
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
        anomalyScore = self.al.anomalyProbability(value, anomaly_score, timestamp)
        logScore = self.al.computeLogLikelihood(anomalyScore)
        return logScore

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
            all_encodings.append(feat.encode(features_data[name]))

        # Combine all features encodings into one for Spatial Pooling
        encoding = SDR(self.encoding_width).concatenate(all_encodings)
        return encoding

    def get_predcount(self) -> float:
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
        Outputs:
            pred_count
                type: float
                meaning: number of predictions made by TM at current timestep (# predicted cells / # active columns)
        """
        self.tm.activateDendrites(learn=False)

        # Count number of predicted cells
        n_pred_cells = self.tm.getPredictiveCells().getSum()
        n_cols_per_pred = self.tm.getWinnerCells().getSum()

        # Normalize to number of predictions
        pred_count = 0 if n_cols_per_pred == 0 else float(n_pred_cells) / n_cols_per_pred
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
        pred_count = self.get_predcount() if self.return_pred_count else None
        self.tm.compute(active_columns, learn=learn)
        # Get anomaly metrics
        anomaly_score = self.tm.anomaly
        # Choose feature for value arg
        f1 = self.single_feature if self.single_feature else self.feature_names[0]
        # Get timestamp data if available
        timestamp = features_data[self.feature_timestamp] if self.feature_timestamp else self.iteration_
        anomaly_likelihood = self.get_alikelihood(value=features_data[f1],
                                                  timestamp=timestamp,
                                                  anomaly_score=anomaly_score)

        # Ensure pred_count > 0 when anomaly_score < 1.0
        if anomaly_score < 1.0 and pred_count == 0:
            raise RuntimeError(f"0 preds with anomaly={anomaly_score}")

        # Increment iteration
        self.iteration_ += 1

        return anomaly_score, anomaly_likelihood, pred_count