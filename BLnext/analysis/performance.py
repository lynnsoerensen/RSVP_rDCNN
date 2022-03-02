from pathlib import Path
import sys
sys.path.extend([Path(__file__).parent.parent])
import logging
import numpy as np
import pandas as pd
import joblib
from scipy.io import loadmat
from BLnext.models.blnext import makeModel_blnext
from BLnext.models.controls import makeModel_control
from BLnext.analysis.trials import loadRSVPdesign, loadTrial
from BLnext.analysis.prototypes import obtainPrototypeActivations_selection
from scipy import stats


import matplotlib.pyplot as plt
import seaborn as sns

cm = 1 / 2.54
_logger = logging.getLogger(__name__)

def loadHumanPerformance():
    data = loadmat(Path(__file__).parent.parent.parent / 'resources'/'humanBehaviour'/'data_transform.mat')

    durations = np.array([0.013, 0.040, 0.080, 0.160])

    dPrimes = data['stats']['dprime_loglinear'][0][0][:, 0][:, data['stats']['imageCond'][0][0][:, 0] == 0]

    num_subjects = dPrimes.shape[1]
    df_human = pd.DataFrame({'Presentation duration': np.repeat(durations, num_subjects),
                             'Subject': np.tile(np.arange(num_subjects), (len(durations), 1)).flatten(),
                             'dPrime': dPrimes.flatten()})

    return df_human


def obtainPerformanceMeasures(target_corrs, targetVector, thresholds = np.arange(0, 1.01, 0.01)):
    # d' is derived from here:
    # http: // www.mbfys.ru.nl / ~robvdw / DGCN22 / PRACTICUM_2011 / LABS_2011 / ALTERNATIVE_LABS / Lesson_9.html
    tracker = {}
    tracker['TargetCorrs'] = target_corrs
    PPC = np.max(target_corrs, axis=1)

    hit_rate = np.zeros((len(thresholds)))
    fa_rate = np.zeros((len(thresholds)))

    # sample ROC curve
    for i_t, t in enumerate(thresholds):
        hit_rate[i_t] = (PPC >= t)[targetVector == 1].mean()
        fa_rate[i_t] = (PPC >= t)[targetVector == 0].mean()

    tracker['hit_rate'] = hit_rate
    tracker['fa_rate'] = fa_rate

    # intgrate area under the curve definec by hit and fa rate
    tracker['AUC'] = -np.trapz(hit_rate, fa_rate)

    # inverseNormalCDF for d'
    tracker['dPrime'] = np.sqrt(2) * stats.norm.ppf(tracker['AUC'])

    return tracker

def perform_isolatedImages(model_name, sample, protoType, dataset='ecoset'):

    # load the design
    design = loadRSVPdesign()
    num_images = 6

    # load the model
    model = makeModel_control(model_name,  trainingSet=dataset)

    # load the protoTypes
    protoPath = Path(
        __file__).parent.parent / 'prototypes' / 'activations' / f"activations500_bl_FC_{dataset}_{protoType}.pkl"
    if protoPath.is_file():
        protoTypes = joblib.load(protoPath)
    else:
        protoTypes = obtainPrototypeActivations_selection(model_name, selection=int(protoType[3:]), dataset=dataset)

    np.random.seed(5)
    target_corrs = np.zeros((len(design), num_images))
    # loop over trials
    for idx in range(len(design)):
        images = loadTrial(design, idx, 1, offset=0)
        predictions = model.predict(np.squeeze(np.array(images)))

        # This is to achieve compatibility for both B and BL models
        if len(np.array(predictions).shape) == 2:
            predictions = np.array(predictions)[np.newaxis,:,:] # introduces an empty time dimension.
        else:
            predictions = np.array(predictions)

        # Compute the category centroid
        centroid = protoTypes[design.loc[idx, 't']].mean(axis=0)

        for t in range(num_images):
            target_corrs[idx, t] = np.corrcoef(predictions[sample - 1, t], centroid)[0, 1]

    tracker = obtainPerformanceMeasures(target_corrs, design['tpres'])

    return tracker


def perform_sequentialImages(model_name, samples, protoType, offset=4, dataset='ecoset',alphas=[0], beta=0):

    # load the design
    design = loadRSVPdesign()
    num_images = 6

    # load the model
    trial_duration = samples * num_images + offset
    model = makeModel_blnext(n_times=trial_duration,  beta=beta, alpha=alphas, trainingSet=dataset)

    # load the protoTypes
    protoPath = Path(__file__).parent.parent / 'prototypes' /'activations'/ f"activations500_bl_FC_{dataset}_{protoType}.pkl"

    if protoPath.is_file():
        protoTypes = joblib.load(protoPath)
    else:
        protoTypes = obtainPrototypeActivations_selection(model_name, selection=int(protoType[3:]), dataset=dataset)

    np.random.seed(5)
    target_corrs = np.zeros((len(design), trial_duration))
    # loop over trials
    for idx in range(len(design)):
        images = loadTrial(design, idx, samples)
        predictions = model.predict(images)

        # Compute the category centroid
        centroid = np.array(protoTypes[design.loc[idx, 't']]).mean(axis=0)

        for t in range(trial_duration):
            target_corrs[idx, t] = np.corrcoef(predictions[t], centroid)[0, 1]

    tracker = obtainPerformanceMeasures(target_corrs, design['tpres'])

    return tracker

def perform(model_name, samples, protoType, dataset='ecoset',alphas=[0], beta=0):

    if model_name in ['b', 'b_d', 'bl']:
        tracker = perform_isolatedImages(model_name, samples, protoType, dataset=dataset)
    else:
        tracker = perform_sequentialImages(model_name, samples, protoType, dataset=dataset,alphas=alphas, beta=beta)

    tracker['model'] = model_name
    tracker['samples'] = samples
    tracker['prototype'] = protoType

    return tracker


def collectPerformance(models, samples, prototype='sel10'):
    data = []
    for model in models:

        for sample in samples:

            if model in ['b', 'b_d']:  # Only evaluate these models once since they don't have any time steps.
                sample_current = 1
            else:
                sample_current = sample

            if (model =='bl') & (sample > 8):
                pass
            else:
                file_name = Path(
                    __file__).parent.parent / 'results' / f"RSVP-performance_{model}_Samples-{sample_current}_{prototype}.pkl"

                if file_name.is_file():
                    # Check if this exists already
                    tracker = joblib.load(file_name)
                else:
                    # Otherwise, run the experiment
                    print('Evaluating ' + model + ' model with ' + str(sample) + ' samples on ' + prototype)
                    if model in ['b', 'b_d', 'bl']:
                        tracker = perform_isolatedImages(model, sample, prototype)
                    else:
                        tracker = perform_sequentialImages(model, sample, prototype, alphas=models[model]['Alpha'],
                                                           beta=models[model]['Beta'])

                    tracker['model'] = model
                    tracker['samples'] = sample
                    tracker['prototype'] = prototype
                    joblib.dump(tracker, file_name, compress=True)

                data.append({'Model': model, 'Samples': sample, 'Prototype': prototype, 'dPrime': tracker['dPrime'],
                             'AUC': tracker['AUC']})

    return pd.DataFrame(data)


def collectPPCs(models, samples, prototype='sel10'):
    data = pd.DataFrame()
    design = loadRSVPdesign()
    targetVector = design['tpres']
    for model in models:

        for sample in samples:

            if model in ['b', 'b_d']:  # Only evaluate these models once since they don't have any time steps.
                sample_current = 1
            else:
                sample_current = sample

            if (model =='bl') & (sample > 8):
                pass
            else:
                file_name = Path(
                    __file__).parent.parent / 'results' / f"RSVP-performance_{model}_Samples-{sample_current}_{prototype}.pkl"

                if file_name.is_file():
                    # Check if this exists already
                    tracker = joblib.load(file_name)
                else:
                    # Otherwise, run the experiment
                    print('Evaluating ' + model + ' model with ' + str(sample) + ' samples on ' + prototype)
                    if model in ['b', 'b_d', 'bl']:
                        tracker = perform_isolatedImages(model, sample, prototype)
                    else:
                        tracker = perform_sequentialImages(model, sample, prototype, alphas=models[model]['Alpha'],
                                                           beta=models[model]['Beta'])

                    tracker['model'] = model
                    tracker['samples'] = sample
                    tracker['prototype'] = prototype
                    joblib.dump(tracker, file_name, compress=True)


                df = pd.DataFrame(np.max(tracker['TargetCorrs'], axis=1), columns=['PPC'])
                df['Target'] = targetVector
                df['Model'] = model
                df['Samples'] = sample
                df['Prototype'] = prototype

                data = pd.concat([data, df], ignore_index=True)

    return data


def showPerformance(models, samples, figureName, data=None, colors=None):
    data = data if data is not None else collectPerformance(models, samples)

    fig_dir = Path(__file__).parent.parent / 'figures'

    # Make figure
    if colors is not None:
        sns.set_palette(np.array(colors))

    markers = ['o' for m in range(len(models))]

    fig, ax = plt.subplots(figsize=(6.5*cm,8*cm))#(6.3*cm, 6.8*cm))

    ax = drawHumanPerformance_ci(ax, durations=[0.013, 0.04, 0.08, 0.16])
    sns.lineplot(data=data, style='Model', hue='Model', x='Samples', y='dPrime', ax=ax,
                 dashes=False, markers=markers, legend=False)

    ax.set_xticks([1, 2, 4, 6, 8, 10, 12])
    ax.set_yticks([0, 1, 2, 3])

    ax.set_ylim([-0.15, 3])
    ax.set_ylabel("Sensitivity (d')")
    ax.set_xlabel("Model steps/image")

    ax.axhline(0, c='gray', ls='--', alpha=0.4)

    ax.set_title('RSVP performance', fontweight='bold', fontsize=10)
    sns.despine()
    plt.tight_layout()

    plt.savefig(fig_dir/ f"{figureName}.pdf", dpi=300, transparent=True)
    plt.savefig(fig_dir/ f"{figureName}.png")



def drawHumanPerformance_ci(ax, durations =[0.013, 0.04, 0.08, 0.16]):

    df_human = loadHumanPerformance()

    for duration in durations:
        ci = sns.utils.ci(
            sns.algorithms.bootstrap(df_human.loc[df_human['Presentation duration'] == duration, 'dPrime'], func="mean", seed=0))
        ax.axhspan(*ci, color='gray', alpha=0.2)

    return ax


def showHumanPerformance(colors=None):
    fig_dir = Path(__file__).parent.parent / 'figures'

    df_human = loadHumanPerformance()

    # Make figure
    if colors is not None:
        sns.set_palette(np.array(colors))

    fig, ax = plt.subplots(figsize=(4.25 * cm, 4 * cm))

    sns.lineplot(data=df_human, x='Presentation duration', y='dPrime', ax=ax,
                 dashes=False, err_kws={'fmt':'-o', 'markersize':1.5}, legend=False, err_style='bars')


    ax.set_xticks([0.013, 0.04, 0.08, 0.16])
    ax.set_xticklabels([13, 40, 80, 160])
    ax.set_ylabel("Sensitivity (d')")
    ax.set_xlabel("Presentation rate")
    #ax.set_xlabel("Steps/image")

    ax.axhline(0, c='gray', ls='--', alpha=0.4)

    plt.tight_layout()

    plt.savefig(fig_dir / 'RSVP-performance_humans.pdf', dpi=300, transparent=True)
    plt.savefig(fig_dir / 'RSVP-performance_humans.png')

def showHumanPerformanceDetailed(colors=None):
    fig_dir = Path(__file__).parent.parent / 'figures'

    df_human = loadHumanPerformance()
    durations = np.array([0.013, 0.040, 0.080, 0.160])
    df_extended = df_human.copy()
    durations_ext = np.arange(0.005, 0.17, 0.001).round(3)

    ticks_id = []
    for d in durations:
        df_extended.loc[df_extended['Presentation duration'] == d, 'id'] = np.where(durations_ext == d)[0][0]
        ticks_id.append(np.where(durations_ext == d)[0][0])

    for d in durations_ext:
        if d not in durations:
            df_extended = df_extended.append({'Presentation duration': d, 'dPrime': np.nan, 'Subject': np.nan,
                                              'id': np.where(durations_ext == d)[0][0]}, ignore_index=True)

    if colors is not None:
        sns.set_palette(np.array(colors))


    fig, ax = plt.subplots(figsize=(6.3*cm, 6.8*cm))

    sns.lineplot(data=df_extended, x='id', y='dPrime', dashes=False, legend=False, markers='o',
                 err_style="bars", ci=95, ax=ax, color=colors[0],
                 err_kws={'fmt': '-o', 'barsabove': True, 'zorder': 6})

    sns.stripplot(x='id', y='dPrime', data=df_extended, jitter=3, alpha=.15, zorder=1, ax=ax, color=colors[0])

    ax.axhline(0, c='gray', ls='--', alpha=0.4)
    ax.set_xticks(ticks_id)
    ax.set_yticks([-1, 0, 1, 2, 3])
    ax.set_xticklabels(np.array(durations * 1000, dtype=int))
    ax.set_ylabel("Sensitivity (d')")
    ax.set_xlabel('Presentation rate (ms)')
    sns.despine()
    plt.tight_layout()

    plt.savefig(fig_dir / 'RSVP-performance_humans_detailed.pdf', dpi=300, transparent=True)
    plt.savefig(fig_dir / 'RSVP-performance_humans_detailed.png')

def showPPCs(models, samples, data=None, colors=None):
    data = data if data is not None else collectPPCs(models, samples)

    fig_dir = Path(__file__).parent.parent / 'figures'

    # Make figure
    if colors is not None:
        sns.set_palette(np.array(colors))

    fig, ax = plt.subplots(figsize=(4.9 * cm , 3.5 * cm ))

    sns.distplot(x=data.loc[data['Target'] == 1, 'PPC'], ax=ax)
    sns.distplot(x=data.loc[data['Target'] == 0, 'PPC'], ax=ax)

    ax.set_xlabel('PPCs (all trials)')
    ax.set_xlim([0.15, 1])
    ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(fig_dir / 'PPC-distribution_example.pdf', dpi=300, transparent=True)
    plt.savefig(fig_dir / 'PPC-distribution_example.png')


def comparePrototypePerformance(models, samples, prototypes, figureName, data=None, colors=None):

    if data is not None:
        data = data
    else:
        data = pd.DataFrame()
        for prototype in prototypes:
            df = collectPerformance(models, samples, prototype=prototype)
            data = pd.concat([data, df], ignore_index=True)

    fig_dir = Path(__file__).parent.parent / 'figures'

    # Make figure
    if colors is not None:
        sns.set_palette(np.array(colors))

    g = sns.FacetGrid(data, col="Prototype", hue='Model', height=6 * cm, aspect=0.75)

    g.map(sns.lineplot, 'Samples', 'dPrime')

    (g.set_axis_labels("Model steps/image", "Sensitivity (d')")
     .set(xticks=[1, 2, 4, 6, 8, 10, 12], yticks=[0, 1, 2, 3])
     .tight_layout(w_pad=0))

    for prototype, ax in g.axes_dict.items():
        ax = drawHumanPerformance_ci(ax, durations=[0.013, 0.04, 0.08, 0.16])
        ax.axhline(0, c='gray', ls='--', alpha=0.4)
        ax.set_title(prototype[3:] + ' images/prototype', fontweight='bold', fontsize=10)

    plt.tight_layout()

    plt.savefig(fig_dir / f'{figureName}.pdf', dpi=300, transparent=True)
    plt.savefig(fig_dir / f'{figureName}.png')