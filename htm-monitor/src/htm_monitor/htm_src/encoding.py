import logging

from htm.encoders.date import DateEncoder
from htm.encoders.rdse import RDSE_Parameters, RDSE
from .types import HTMType

log = logging.getLogger(__name__)


def init_rdse(rdse_params, max_fail=3):
    encoder = None
    counter = 0
    while encoder is None:
        try:
            encoder = RDSE(rdse_params)
        except RuntimeError as e:
            counter += 1
            if counter == max_fail:
                log.error(
                    msg=f"Failed RDSE random collision check {max_fail} times\n  change rdse params --> {rdse_params}")
                raise RuntimeError(e)
            pass
    return encoder


class EncoderFactory:
    @staticmethod
    def get_encoder(dtype: HTMType, encoder_params: dict):
        """
        Returns the appropriate encoder based on given HTMType and parameters dict
        """
        # Numeric + Categoric share RDSE; Categoric sets category=True.
        if dtype in [HTMType.Numeric, HTMType.Categoric]:
            rdse_params = RDSE_Parameters()
            rdse_params.seed = encoder_params['seed']
            rdse_params.size = encoder_params['size']
            rdse_params.activeBits = encoder_params["activeBits"]

            if dtype is HTMType.Numeric:
                # Allow explicit resolution, otherwise derive from (minVal,maxVal,numBuckets).
                if "resolution" in encoder_params:
                    rdse_params.resolution = float(encoder_params["resolution"])
                else:
                    try:
                        nb = float(encoder_params["numBuckets"])
                        min_val = float(encoder_params["minVal"])
                        max_val = float(encoder_params["maxVal"])
                    except KeyError as e:
                        raise ValueError(
                            "Numeric RDSE feature must specify either "
                            "'resolution' OR ('numBuckets' + 'minVal' + 'maxVal')"
                        ) from e

                    if nb <= 0:
                        raise ValueError("numBuckets must be > 0")
                    span = max_val - min_val
                    if span <= 0:
                        raise ValueError("maxVal must be > minVal")

                    rdse_params.resolution = span / nb
                    log.info(
                        f"RDSE[{encoder_params.get('name','?')}]: span={span} numBuckets={nb} -> resolution={rdse_params.resolution}"
                    )
            else:  # dtype is HTMType.Categoric
                rdse_params.category = True
            encoder = init_rdse(rdse_params)

        elif dtype is HTMType.Datetime:
            encoder = DateEncoder(timeOfDay=encoder_params["timeOfDay"],
                                  weekend=encoder_params["weekend"],
                                  dayOfWeek=encoder_params["dayOfWeek"],
                                  holiday=encoder_params["holiday"],
                                  season=encoder_params["season"])

        # future implementations here..

        else:
            raise NotImplementedError(f"Encoder not implemented for '{dtype}'")

        return encoder