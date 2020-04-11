import numpy as np
import nibabel as nb
from .conform import conform_data


def align(template, volume):

    axis_combination = []
    errors = []

    transformed_volume = volume.copy()
    template = conform_data(template, out_size=(32, 32, 32))
    volume = conform_data(template, out_size=(32, 32, 32))

    for ix in list(range(3)):
        axis = list(range(3))
        axis.pop(ix)  # remove first axis
        for jy in axis:
            axis = list(range(3))
            axis.pop(ix)
            axis.pop(jy)
            axis_combination.append((ix, jy, axis[0]))
            errors.append(
                np.linalg.norm(template - volume.transpose(axis_combination[-1]))
            )
    i = np.argmin(errors)

    return (transformed_volume.transpose(axis_combination[i]), axis_combination[i])
