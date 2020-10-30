#ADD CES_DATA CITATION

import pandas as pd
import numpy as np
from scipy.io import loadmat

def load_llds(filepath):
    with open(filepath)as f:
        line = f.readline()
    header=None
    if line[:4] == 'name':
        header='infer'
    sep=';'
    if ',' in line:
        sep=','

    num_features = len(pd.read_csv(filepath, sep=sep, header=header).columns)-2

    F = np.array(pd.read_csv(filepath, sep=sep, header=header, usecols=range(2, 2+num_features)).values)
    
    return F

def summarise_audio(audio_features):
    return np.array(pd.DataFrame(audio_features).groupby(np.arange(len(audio_features))//4).mean())


def load_deep_feature(filepath):
    features = loadmat(filepath)['feature']
    return np.array(pd.DataFrame(features))