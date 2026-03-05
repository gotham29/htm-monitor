#htm_src/feature.py

from datetime import datetime
from typing import Any, Iterable, Mapping, Optional, Tuple, Union

from htm.bindings.sdr import SDR

from .types import HTMType, to_htm_type
from .encoding import EncoderFactory


class _NullEncoder:
    """
    Encoder stub used when feature params specify encode: false.
    Guarantees:
      - size == 0 (so it does not contribute to concatenated encoding width)
      - encode(...) returns SDR(0)
    """
    size = 0

    def encode(self, _data: Any) -> SDR:
        return SDR(0)


class Feature:
    def __init__(self, name: str, params: dict):
        """
        This is a wrapper class for easy type checking and encoding of features
        """
        self._type = to_htm_type(params['type'])
        self._name = name
        self._params = params
        # If encode=false, feature exists for parsing/timebase/metadata,
        # but contributes zero bits to model input.
        self._encode = bool(self._params.get("encode", True))

        # If encode is disabled, we still keep the feature for typing + timestamp usage,
        # but it contributes ZERO bits and is never passed through a real encoder.
        if not self._encode:
            self._encoder = _NullEncoder()
        else:
            self._encoder = EncoderFactory.get_encoder(self._type, params)

        if self.type == HTMType.Datetime:
            if "format" not in self._params:
                raise ValueError(f"Datetime-like feature `{self.name}` must have a `format` parameter")
            self._dt_format = self._params["format"]

    def parse(self, data: Union[str, int, float, datetime]) -> Union[int, float, datetime]:
        """
        Parse raw input into the canonical python type for this feature.
        (Used even when encode is disabled, e.g. timestamp passed to AnomalyLikelihood.)
        """
        if self.type == HTMType.Datetime:
            if isinstance(data, datetime):
                return data
            if isinstance(data, str):
                return datetime.strptime(data, self._dt_format)
            raise ValueError(f"Datetime-like feature `{self.name}` expected str or datetime; got {type(data)}")
        # numeric/categoric encoders can accept int/float directly; leave as-is.
        return data  # type: ignore[return-value]

    def encode(self, data: Union[str, int, float, datetime]) -> SDR:
        """
        Encodes input `data` with the appropriate encoder, based on `params` given in init
        """
        if not self._encode:
            return SDR(0)

        data = self.parse(data)
        return self._encoder.encode(data)

    def __eq__(self, other) -> bool:
        return self.name == other.name

    def __repr__(self) -> str:
        return f"Feature(name={self._name}, dtype={self._type}, params={self._params})"

    @property
    def encode_enabled(self) -> bool:
        return bool(self._encode)

    @property
    def type(self) -> HTMType:
         return self._type

    @property
    def name(self) -> str:
        return self._name

    @property
    def params(self) -> Mapping[str, Any]:
        return self._params

    @property
    def encoding_size(self) -> int:
        if not self._encode or self._encoder is None:
            return 0
        return int(self._encoder.size)

def separate_time_and_rest(features: Iterable[Feature], strict: bool = True) -> Tuple[Optional[str], Tuple[str, ...]]:
    """
    Given any iterable of Features, will separate the time-like feature from the rest and return the feature names:
    time_feature, (other_f_1, ...)

    If `strict` is set to True, will raise an exception if more than 1 time-like feature is found
    """
    time = None
    non_time = list()
    for feat in features:
        if feat.type == HTMType.Datetime:
            if strict and time is not None:
                raise ValueError(f"More than a single time-like feature found: {time, feat.name}")
            else:
                time = feat.name
        else:
            non_time.append(feat.name)

    return time, tuple(non_time)