from preprocessing import clip, normalize, standardize
from preprocessing.conform import conform_data

def preprocessing(vol_path, conform_path=None):

    if not conform_path:

