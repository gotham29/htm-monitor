import math
import logging
from typing import Mapping, Union

import numpy as np
from htm.bindings.algorithms import TemporalMemory
from htm.bindings.sdr import SDR

from .feature import Feature, separate_time_and_rest
from .anomalylikelihood import AnomalyLikelihood


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
        return TemporalMemory(
            columnDimensions=(column_dimensions,),
            cellsPerColumn=self.models_params["tm"]["cellsPerColumn"],
            activationThreshold=self.models_params["tm"]["activationThreshold"],
            initialPermanence=self.models_params["tm"]["initialPerm"],
            connectedPermanence=self.models_params["tm"]["permanenceConnected"],
            minThreshold=self.models_params["tm"]["minThreshold"],
            maxNewSynapseCount=self.models_params["tm"]["newSynapseCount"],
            permanenceIncrement=self.models_params["tm"]["permanenceInc"],
            permanenceDecrement=self.models_params["tm"]["permanenceDec"],
            predictedSegmentDecrement=self.models_params["tm"]["predictedSegmentDecrement"],
            maxSegmentsPerCell=self.models_params["tm"]["maxSegmentsPerCell"],
            maxSynapsesPerSegment=self.models_params["tm"]["maxSynapsesPerSegment"],
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