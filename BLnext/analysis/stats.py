from pathlib import Path
import sys
sys.path.extend([Path(__file__).parent.parent])
import logging
import pandas as pd
from scipy.io import loadmat
import os
import joblib
import numpy as np
import pingouin
from BLnext.analysis.performance import loadHumanPerformance
from pingouin import ttest

def reportSampleStats():
    dataAgeGender = loadmat(Path(__file__).parent.parent.parent / 'resources' / 'humanBehaviour' / 'data_AgeGender.mat')

    dataCond = loadmat(Path(__file__).parent.parent.parent / 'resources' / 'humanBehaviour' / 'data_transform.mat')

    df={}
    df['N'] = np.sum(dataCond['stats']['imageCond'][0][0][:, 0] == 0)
    df['AverageAge'] = np.nanmean(dataAgeGender['stats']['age'][0][0][dataCond['stats']['imageCond'][0][0][:, 0] == 0])
    df['SDAge'] = np.nanstd(dataAgeGender['stats']['age'][0][0][dataCond['stats']['imageCond'][0][0][:, 0] == 0])
    df['MinAge'] = np.nanmin(dataAgeGender['stats']['age'][0][0][dataCond['stats']['imageCond'][0][0][:, 0] == 0])
    df['MaxAge'] = np.nanmax(dataAgeGender['stats']['age'][0][0][dataCond['stats']['imageCond'][0][0][:, 0] == 0])
    df['AgeMissing'] = np.sum(np.isnan(dataAgeGender['stats']['age'][0][0][dataCond['stats']['imageCond'][0][0][:, 0] == 0]))

    return df


def reportdPrimeStats(durations = None):
    df = loadHumanPerformance()

    if durations != None:
        durations = [0.0130, 0.040, 0.080, 0.16]

    results = pd.DataFrame()
    for duration in durations:
        result = ttest(df.loc[df['Presentation duration'] == duration, 'dPrime'], 0).round(6)

        results = pd.concat([results, result], ignore_index=True)

    results['Duration'] = durations
    return results