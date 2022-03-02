from pathlib import Path
import sys
sys.path.extend([Path(__file__).parent.parent])
import logging
from scipy.io import loadmat
import joblib
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from BLnext.analysis.performance import perform
import matplotlib.pyplot as plt
import seaborn as sns

cm = 1 / 2.54

_logger = logging.getLogger(__name__)

def loadHumanPredictions():
    data = loadmat(Path(__file__).parent.parent.parent / 'resources' / 'humanBehaviour' / 'data_item.mat')
    return data["dataItem"]


def computeNoiseCeilings(humanPredictions, durations):

    labelCond = humanPredictions['labelCond'][0][0]
    durationCond = humanPredictions['duration'][0][0]
    accuracy = humanPredictions['accuracy'][0][0]
    tpresent = humanPredictions['tpresent'][0][0]
    imageCond = humanPredictions['imageCond'][0][0]

    subjects = labelCond.shape[1]

    noiseCeilings = {}
    for d, duration in enumerate(durations):
        lower_NC = []
        upper_NC = []
        for p in range(subjects):
            remaining = np.setdiff1d(np.arange(subjects), p)
            n_p = ((durationCond == duration) & (imageCond == 0) & (labelCond == 0))[:, p]
            if np.sum(n_p) > 0:
                hits_p = ((tpresent == 1) & (durationCond == duration) & (imageCond == 0) & (labelCond == 0) & (
                            accuracy == 1))[:, p]
                fa_p = ((tpresent == 0) & (durationCond == duration) & (imageCond == 0) & (labelCond == 0) & (
                            accuracy == 0))[:, p]
                det_p = fa_p + hits_p

                n = np.sum((durationCond == duration) & (imageCond == 0) & (labelCond == 0), axis=1)
                hits = np.sum((tpresent == 1) & (durationCond == duration) & (imageCond == 0) & (labelCond == 0) & (
                            accuracy == 1), axis=1)
                fa = np.sum((tpresent == 0) & (durationCond == duration) & (imageCond == 0) & (labelCond == 0) & (
                            accuracy == 0), axis=1)
                detections = np.divide((hits + fa), n)

                nan_idx = np.isnan(detections) == False

                upper_NC.append(spearmanr(detections[n_p & nan_idx], det_p[n_p & nan_idx])[0])

                n_low = np.sum(((durationCond == duration) & (imageCond == 0) & (labelCond == 0))[:, remaining], axis=1)
                hits_low = np.sum(
                    ((tpresent == 1) & (durationCond == duration) & (imageCond == 0) & (labelCond == 0) & (
                                accuracy == 1))[:, remaining],
                    axis=1)
                fa_low = np.sum(
                    ((tpresent == 0) & (durationCond == duration) & (imageCond == 0) & (labelCond == 0) & (
                                accuracy == 0))[:, remaining],
                    axis=1)
                detections_low = np.divide((hits_low + fa_low), n_low)

                nan_idx = np.isnan(detections_low) == False

                lower_NC.append(spearmanr(detections_low[n_p & nan_idx], det_p[n_p & nan_idx])[0])

        noiseCeilings[duration] = {}
        noiseCeilings[duration]['lowerNC'] = np.mean(lower_NC)
        noiseCeilings[duration]['upperNC'] = np.mean(upper_NC)

    return noiseCeilings

def obtainNoiseCeilings(durations):

    file_name = Path(__file__).parent.parent / 'results'/ f"NoiseCeilings.pkl"
    if file_name.is_file():
        # Check if this exists already
        NC = joblib.load(file_name)
    else:
        humanPredictions = loadHumanPredictions()
        NC = computeNoiseCeilings(humanPredictions, durations)
        joblib.dump(NC, file_name)

    return NC


def drawNoiseCeilings(g, durations):
    noiseCeilings = obtainNoiseCeilings(durations)

    for d, duration in enumerate(durations):
        g.axes[0][d].axhspan(noiseCeilings[duration]['lowerNC'], noiseCeilings[duration]['upperNC'], color='grey',
                             label='Noise ceiling',
                             alpha=0.1)

    return g

def computeSingleTrialCorr(PPC, humanPredictions, duration, num_bootstraps=1000, bootstrap_size=1, random_seed=0,
                           sample=None, model=None, prototype=None, return_data=False):
    labelCond = humanPredictions['labelCond'][0][0]
    durationCond = humanPredictions['duration'][0][0]
    accuracy = humanPredictions['accuracy'][0][0]
    tpresent = humanPredictions['tpresent'][0][0]
    imageCond = humanPredictions['imageCond'][0][0]

    subjects = labelCond.shape[1]
    np.random.seed(random_seed)

    if num_bootstraps == 0:
        n_current = np.sum(((durationCond == duration) & (imageCond == 0) & (labelCond == 0)), axis=1)

        hits_current = np.sum(((tpresent == 1) & (durationCond == duration) & (imageCond == 0) & (labelCond == 0) & (
                accuracy == 1)), axis=1)
        fa_current = np.sum(((tpresent == 0) & (durationCond == duration) & (imageCond == 0) & (labelCond == 0) & (
                accuracy == 0)), axis=1)

        detectionrate_current = np.divide((hits_current + fa_current), n_current)

        nan_idx_current = np.isnan(detectionrate_current) == False

        result = pd.DataFrame(
            [{'Model': model, 'Samples': sample, 'Prototype': prototype, 'Duration': duration,
             'Rho': spearmanr(PPC[nan_idx_current], detectionrate_current[nan_idx_current])[0]}])

        if return_data == True:
            result = [result, pd.DataFrame({'PPC':PPC[nan_idx_current],
                           'Response rate': detectionrate_current[nan_idx_current]})]

    else:
        bootstraps = []

        for i in range(num_bootstraps):
            sel_idx = np.random.choice(np.arange(subjects)[imageCond[0, :] == 0], size=int(bootstrap_size * subjects))

            n_current = np.sum(((durationCond == duration) & (imageCond == 0) & (labelCond == 0))[:, sel_idx], axis=1)
            hits_current = np.sum(((tpresent == 1) & (durationCond == duration) & (imageCond == 0) & (labelCond == 0) & (
                    accuracy == 1))[:, sel_idx], axis=1)
            fa_current = np.sum(((tpresent == 0) & (durationCond == duration) & (imageCond == 0) & (labelCond == 0) & (
                    accuracy == 0))[:, sel_idx], axis=1)

            detectionrate_current = np.divide((hits_current + fa_current), n_current)

            nan_idx_current = np.isnan(detectionrate_current) == False

            bootstraps.append({'Model': model, 'Samples': sample, 'Prototype':prototype, 'Bootstrap': i, 'Duration': duration,
                               'Rho': spearmanr(PPC[nan_idx_current], detectionrate_current[nan_idx_current])[0]})

        result = pd.DataFrame(bootstraps)

    return result



def predict(model, sample, prototype, duration, alphas=[0], beta=0., num_bootstraps=1000):
    humanPredictions = loadHumanPredictions()
    file_name = Path(__file__).parent.parent / 'results' / f"RSVP-performance_{model}_Samples-{sample}_{prototype}.pkl"

    if file_name.is_file():
        # Check if this exists already
        tracker = joblib.load(file_name)
    else:
        print('Evaluating ' + model + ' model with ' + str(sample) + ' samples')
        # Otherwise, run the experiment
        tracker = perform(model, sample, prototype, alphas=alphas, beta=beta)
        joblib.dump(tracker, file_name, compress=True)

    PPC = np.max(tracker['TargetCorrs'], axis=1)
    df = computeSingleTrialCorr(PPC, humanPredictions, duration, sample=sample, model=model, prototype=prototype,
                                num_bootstraps=num_bootstraps)

    return df


def collectPredictivity(models, samples, durations, prototype='sel10'):

    data = pd.DataFrame()
    for model in models:
        for sample in samples:
            if model in ['b', 'b_d']:  # Only evaluate these models once since they don't have any time steps.
                sample_current = 1
            else:
                sample_current = sample

            if (model =='bl') & (sample > 8):
                pass
            else:
                for duration in durations:
                    file_name = Path(__file__).parent.parent / 'results'/ f"RSVP-predictivity_{model}_Samples-{sample_current}_Duration-{duration}s_{prototype}.pkl"

                    if file_name.is_file():
                        # Check if this exists already
                        df = pd.read_pickle(file_name)
                        df['Samples'] = sample
                    else:
                        print('Evaluating predictivity for ' + model + ' model with ' + str(sample) + ' samples at ' + str(duration) + ' s...')
                        # Otherwise, run the experiment
                        if model in ['b', 'b_d', 'bl']:
                            df = predict(model, sample, prototype, duration)
                        else:
                            df = predict(model, sample, prototype, duration, alphas=models[model]['Alpha'],
                                         beta=models[model]['Beta'])

                        df.to_pickle(file_name)

                    data = pd.concat([data, df], ignore_index=True)

    return data


def drawCI_area(x, y, **kwargs):
    bounds = y.groupby([x]).quantile((0.025, 0.975)).unstack()
    plt.fill_between(x=bounds.index, y1=bounds.iloc[:, 0], y2=bounds.iloc[:, 1],  alpha=0.1, **kwargs)

def drawCI_bar(x, y,  **kwargs):
    bounds = y.groupby([x]).quantile((0.025, 0.975)).unstack()

    # get rid of these arguments for compatibility with plt.
    if 'hue' in kwargs:
        kwargs.pop('hue')

    if 'order' in kwargs:
        kwargs.pop('order')

    if 'hue_order' in kwargs:
        kwargs.pop('hue_order')

    plt.errorbar(x=bounds.index, y=y.mean(), yerr=np.abs(bounds.values - y.mean()).T, **kwargs)



def showPredictivity(models, samples, durations,figureName,  data=None, colors=None):
    data = data if data is not None else collectPredictivity(models, samples, durations)

    fig_dir = Path(__file__).parent.parent / 'figures'

    # Make figure
    if colors is not None:
        sns.set_palette(np.array(colors))

    g = sns.FacetGrid(data, col="Duration", hue='Model', height=7*cm, aspect=0.7)

    g.map(sns.lineplot, 'Samples', 'Rho')
    g.map(drawCI_area, 'Samples', 'Rho')

    g = drawNoiseCeilings(g, durations=durations)

    (g.set_axis_labels("Model steps/image", "Spearman's ρ")
     .set(xticks=[1, 2, 4, 6, 8, 10, 12], yticks=[0, 0.2, 0.4, 0.6, 0.8])
     .tight_layout(w_pad=0))

    for duration, ax in g.axes_dict.items():
        ax.set_title(str(int(duration * 1000)) + ' ms', fontweight='bold', fontsize=10)

    sns.despine()
    plt.tight_layout()

    plt.savefig(fig_dir / f"{figureName}.pdf", dpi=300, transparent=True)
    plt.savefig(fig_dir / f"{figureName}.png")



def comparePredictivity_samples(models, samples, durations,data=None, colors=None):
    data = data if data is not None else collectPredictivity(models, samples, durations)

    fig_dir = Path(__file__).parent.parent / 'figures'

    # Make figure
    if colors is not None:
        sns.set_palette(np.array(colors))

    bootstraps = data['Bootstrap'].max() + 1
    df = []
    for model in models:
        print(model)
        for duration in durations:
            for b in range(bootstraps):
                df.append({'Model': model, 'Duration': duration, 'Bootstrap': b, 'bestSample': data.loc[
                    (data['Model'] == model) & (data['Duration'] == duration) & (data['Bootstrap'] == b)].groupby(
                    ['Samples'])['Rho'].mean().idxmax()})

    df = pd.DataFrame(df)
    del data

    markers = ['o' for m in range(len(models))]

    g = sns.FacetGrid(df, hue='Model', height=6 * cm, aspect=0.85)

    g.map(sns.lineplot, 'Duration', 'bestSample', markers=markers)
    g.map(drawCI_area, 'Duration', 'bestSample')

    (g.set_axis_labels("Presentation rate (ms)", "Most explanatory model step")
     .set(yticks=[1, 2, 4, 6, 8, 10, 12], xticks=durations,xticklabels= np.array(np.array(durations) * 1000,dtype=int))
     .tight_layout(w_pad=0))

    plt.savefig(fig_dir / f"RSVP-predictivty_comparisonSamples.pdf", dpi=300, transparent=True)
    plt.savefig(fig_dir / f"RSVP-predictivty_comparisonSamples.png")


def comparePredictivity_VE(models, samples, durations,data=None, colors=None):
    data = data if data is not None else collectPredictivity(models, samples, durations)

    fig_dir = Path(__file__).parent.parent / 'figures'

    NC = obtainNoiseCeilings(durations)

    bootstraps = data['Bootstrap'].max() + 1
    df = []
    for model in models:
        print(model)
        for d, duration in enumerate(durations):
            for b in range(bootstraps):
                df.append({'Model': model, 'Duration': duration, 'Bootstrap': b,
                           'bestSampleRhoProp': data.loc[(data['Model'] == model) & (data['Duration'] == duration) & (
                                                            data['Bootstrap'] == b)].groupby(
                               ['Samples'])['Rho'].mean().max() / NC[duration]['lowerNC']})

    df = pd.DataFrame(df)
    df2 = df.groupby(['Model', 'Bootstrap'], sort=False)['bestSampleRhoProp'].mean()
    df2 = df2.reset_index()
    del data, df

    # Make figure
    if colors is not None:
        sns.set_palette(np.array(colors))

    g = sns.catplot(data=df2, x='Model', y='bestSampleRhoProp', hue='Model', height=6.1 * cm, aspect=0.95,
                    kind='point', ci=None, order=list(models), hue_order=list(models), facet_kws={'hue':'Model'})

    g.map(drawCI_bar, 'Model','bestSampleRhoProp')

    (g.set_axis_labels("Model", "Proportion explained")
     .set(ylim=[0.65, 1], yticks=[0.7, 0.8, 0.9, 1], xlim=[-1,6])  # , xticklabels=durations * 1000)
     .tight_layout(w_pad=0))

    g.set_xticklabels([' ', ' ', ' ', ' ', ' ', ' '])

    #g.set_xticklabels(['BLnext', 'BLnext (exp.)', 'BLnext (pow.)','BLnet', 'B', 'B-D'], rotation = 40)
    plt.tight_layout()

    plt.savefig(fig_dir / f"RSVP-predictivty_comparisonVE.pdf", dpi=300, transparent=True)
    plt.savefig(fig_dir / f"RSVP-predictivty_comparisonVE.png")



def showSingleTrialCorr(model, sample, duration, prototype='sel10', colors=None):
    humanPredictions = loadHumanPredictions()

    model_name = list(model)[0]
    file_name = Path(__file__).parent.parent / 'results' / f"RSVP-performance_{model_name}_Samples-{sample}_{prototype}.pkl"

    if file_name.is_file():
        # Check if this exists already
        tracker = joblib.load(file_name)
    else:
        print('Evaluating ' + model_name + ' model with ' + str(sample) + ' samples')
        # Otherwise, run the experiment
        tracker = perform(model_name, sample, prototype, alphas=model['Alpha'], beta=model['Beta'])
        joblib.dump(tracker, file_name, compress=True)

    PPC = np.max(tracker['TargetCorrs'], axis=1)

    df, points = computeSingleTrialCorr(PPC, humanPredictions, duration, sample=sample, model=model_name, prototype=prototype,
                                num_bootstraps=0, return_data=True)

    # Make figure
    fig_dir = Path(__file__).parent.parent / 'figures'

    if colors is not None:
        sns.set_palette(np.array(colors))


    g = sns.lmplot(data=points, x='PPC', y='Response rate', height=4 * cm, aspect=1.2,scatter_kws={'alpha':0.5})

    (g.set(ylim=[-0.05,1.05], xlim=[0.2,0.9],yticks=[0, 0.5, 1]).tight_layout(w_pad=0))

    plt.savefig(fig_dir / f"RSVP-predictivty_scatterExample.pdf", dpi=300, transparent=True)
    plt.savefig(fig_dir / f"RSVP-predictivty_scatterExample.png")