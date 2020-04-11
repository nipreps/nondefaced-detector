import numpy as np


def standardize_volume(volume, mask=None):
    """
	volume: volume which needs to be normalized
	mask: brain mask, only required if you prefer not to
		consider the effect of air in normalization
    """
    if mask != None:
        volume = volume * mask

    mean = np.mean(volume[volume > 5])
    std = np.std(volume[volume > 5])
    return (volume - mean) / std


def normalize_volume(volume, mask=None, _type="MinMax"):
    """
        volume: volume which needs to be normalized
        mask: brain mask, only required if you prefer not to
		consider the effect of air in normalization
        _type: {'Max', 'MinMax', 'Sum'}
    """
    if mask != None:
        volume = mask * volume
    min_vol = np.min(volume)
    max_vol = np.max(volume)
    sum_vol = np.sum(volume)

    if _type == "MinMax":
        return (volume - min_vol) / (max_vol - min_vol)
    elif _type == "Max":
        return volume / max_vol
    elif _type == "Sum":
        return volume / sum_vol
    else:
        raise ValueError(
            "Invalid _type, allowed values are: {}".format("Max, MinMax, Sum")
        )
