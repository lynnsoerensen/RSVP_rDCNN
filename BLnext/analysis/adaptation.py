from pathlib import Path
import sys
sys.path.extend([Path(__file__).parent.parent])
import logging
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

cm = 1 / 2.54
_logger = logging.getLogger(__name__)

def simulateSingleUnit(alphas, beta, signal):
    alphas = np.array(alphas)
    r = np.zeros((len(signal)))
    s = np.zeros((len(alphas),len(signal)))

    for t in range(1, len(signal)-1):
        r[t] = np.maximum(signal[t] - beta* (np.sum(s[:, t], axis=0)), 0)
        if t != (len(signal)):
            s[:, t + 1] = (alphas * s[:, t] + (1 - alphas) * r[t])

    return r, s

def collectSimulations(models, signal):
    data={}

    for model in models:
        r, s = simulateSingleUnit(models[model]['Alpha'],models[model]['Beta'], signal)

        data[model] = {}
        data[model]['r'] = r
        data[model]['s'] = s

    return data


def showExampleAdaptation(models, colors=None):
    fig_dir = Path(__file__).parent.parent / 'figures'

    # Make figure
    if colors is not None:
        sns.set_palette(np.array(colors))

    time_steps = 80
    signal = np.zeros(time_steps + 1)

    signal[5:30] = 1
    signal[55:] = 1

    data = collectSimulations(models, signal)

    plt.figure(figsize=(8*cm, 4.5*cm))
    plt.fill_between(np.arange(time_steps + 1), np.zeros(time_steps + 1), signal, color='gray', alpha=0.25,
                     label='Signal')
    for m, model in enumerate(models):

        plt.plot(np.arange(time_steps + 1), data[model]['r'], label='Activation ' + model, c=colors[m])
        if model != 'NoAdaptation':

            plt.plot(np.arange(time_steps + 1), np.mean(data[model]['s'], axis=0), label='Suppression '+ model, c=colors[m], ls='--')

    plt.yticks([0, 0.5, 1])
    plt.xlabel('Model steps')
    plt.ylabel('Normalized\nactivation')
    plt.xlim([0, time_steps])
    plt.ylim([0, 1.05])

    sns.despine()
    plt.tight_layout()
    plt.savefig(fig_dir / 'Adaptation_SingleUnit_Example.pdf', dpi=300, transparent=True)
    plt.savefig(fig_dir / 'Adaptation_SingleUnit_Example.png')


