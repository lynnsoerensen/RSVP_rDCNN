from pathlib import Path
import sys
import logging
sys.path.extend(['/mnt/Googleplex2/PycharmProjects/BLnext', '/mnt/Googleplex2/PycharmProjects/BLnext/BLnext',
                 '/mnt/Googleplex2/PycharmProjects/BLnext/resources','/mnt/Googleplex2/PycharmProjects/BLnext/resources/rcnn_sat' ])

import matplotlib.font_manager as font_manager
font_files = font_manager.findSystemFonts(fontpaths='/usr/share/fonts/truetype/msttcorefonts')
font_list = font_manager.createFontList(font_files)
font_manager.fontManager.ttflist.extend(font_list)

import matplotlib as plt
import seaborn as sns
import numpy as np
from BLnext.analysis.adaptation import showExampleAdaptation
from BLnext.analysis.trials import showExampleImageTrace, showExamplePrototypeTrace, showExampleImageTraces, showExampleCategoryTrace
from BLnext.analysis.ecoset import showPerformanceEcoset
from BLnext.analysis.performance import showPerformance, showHumanPerformance, showHumanPerformanceDetailed, showPPCs, comparePrototypePerformance
from BLnext.analysis.predictivity import showPredictivity, comparePredictivity_samples, comparePredictivity_VE, showSingleTrialCorr
from BLnext.analysis.stats import reportSampleStats, reportdPrimeStats
from BLnext.analysis.prototypes import countCategoryExamples
_logger = logging.getLogger(__name__)

def paper_figures():

    # Plotting arguments
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False

    sns.set_context('paper', rc={'font.size': 10,  'xtick.labelsize': 8, 'ytick.labelsize': 8,
                                 'figure.titleweight': 'bold', 'axes.labelsize': 10, 'axes.titlesize': 12})

    colors = sns.color_palette('colorblind')
    blnext_colors = []
    [blnext_colors.append(colors[c]) for c in [1, -1, 0]]

    control_colors =[]
    [control_colors.append(colors[c]) for c in [3, 6, 4]]

    human_color = []
    [human_color.append(colors[c]) for c in [2]]

    all_colors = []
    [all_colors.append(colors[c]) for c in [1, -1, 0, 3, 6, 4]]

    old_colors = []
    [old_colors.append(colors[c]) for c in [0, 2, 3, 6, 4]]

    sdt_colors = sns.xkcd_palette(['teal', 'orange'])

    seq_colors = []
    [seq_colors.append(colors[c]) for c in [1, -1, 2, 3, 4, -2]]

    # Models
    models = {'NoAdaptation': {'Alpha': [0], 'Beta': 0},
              'Exponential': {'Alpha': [0.96], 'Beta': 0.7},
              'Power': {'Alpha': [0.96, 0.75], 'Beta': 0.15},
              'bl': {'Alpha': [0], 'Beta': 0},
              'b':  {'Alpha': [0], 'Beta': 0},
              'b_d': {'Alpha': [0], 'Beta': 0}}

    models_blnext = {'NoAdaptation': {'Alpha': [0], 'Beta': 0},
              #'Exponential': {'Alpha': [0.96], 'Beta': 0.7},
              'Power': {'Alpha': [0.96, 0.75], 'Beta': 0.15}}

    controlModels = ['bl', 'b', 'b_d']

    # Samples per image for simulations
    samples = np.arange(1, 13)
    samplesControl = np.arange(1, 9)

    # Presentation durations for participants
    durations = [0.0130, 0.040, 0.080]

    # Figures
    _logger.info("Figure: Illustrations")
    showExampleAdaptation(models_blnext, colors=blnext_colors)
    showExampleImageTrace(models_blnext, colors=blnext_colors)

    showExampleCategoryTrace({'NoAdaptation': {'Alpha': [0], 'Beta': 0}}, category='flower', matchingImage=1, colors=seq_colors)

    traces = showExampleImageTraces({'NoAdaptation': {'Alpha': [0], 'Beta': 0}}, 'ImageTraces_Example', samples=4, colors=seq_colors)
    showExampleImageTraces({'NoAdaptation': {'Alpha': [0], 'Beta': 0}}, 'ImageTraces_Example_1st', images=[0], samples=4, colors=seq_colors, data=traces)

    showExamplePrototypeTrace({'NoAdaptation': {'Alpha': [0], 'Beta': 0}}, samples=4, colors=[(0.5, 0.5, 0.5)])

    showPPCs({'NoAdaptation': {'Alpha': [0], 'Beta': 0}}, [8], colors=sdt_colors)

    showPerformanceEcoset(models, samples, 'Ecoset-performance_BLnext', colors=all_colors)

    _logger.info("Figure: RSVP performance for humans")
    showHumanPerformance(colors=human_color)
    showHumanPerformanceDetailed(colors=human_color)

    _logger.info("Figure: RSVP performance for models")
    showPerformance(models_blnext, samples, 'RSVP-performance_BLnext', colors=blnext_colors)
    showPerformance(controlModels, samplesControl, 'RSVP-performance_controls',colors=control_colors)
    showPerformance(models, samples, 'RSVP-performance', colors=all_colors)

    _logger.info("Figure: RSVP performance for models across prototypes")
    prototypes = ['sel10', 'sel50', 'sel100']
    comparePrototypePerformance(models_blnext, samples, prototypes, 'RSVP-performance_BLnext_prototypes', colors=blnext_colors)
    comparePrototypePerformance(controlModels, samplesControl, prototypes, 'RSVP-performance_controls_prototypes', colors=control_colors)

    _logger.info("Figure: Model predictivity for human single trials")

    # Hit (model + human)
    showExamplePrototypeTrace({'Power': {'Alpha': [0.96, 0.75], 'Beta': 0.15}}, samples=8, colors=[(0.5, 0.5, 0.5)],
                              trial=(1 + 11), filename='PrototypeTrace_Trial1')
    # CR (model + human)
    showExamplePrototypeTrace({'Power': {'Alpha': [0.96, 0.75], 'Beta': 0.15}}, samples=8, colors=[(0.5, 0.5, 0.5)],
                              trial=(12 + 11), filename='PrototypeTrace_Trial12')
    # Incorret model, correct human)
    showExamplePrototypeTrace({'Power': {'Alpha': [0.96, 0.75], 'Beta': 0.15}}, samples=8, colors=[(0.5, 0.5, 0.5)],
                             trial=(164 + 11), filename='PrototypeTrace_Trial164')

    # correct model, incorrect human)
    showExamplePrototypeTrace({'Power': {'Alpha': [0.96, 0.75], 'Beta': 0.15}}, samples=8, colors=[(0.5, 0.5, 0.5)],
                             trial=(91 + 11), filename='PrototypeTrace_Trial91')

    showSingleTrialCorr({'Power': {'Alpha': [0.96, 0.75], 'Beta': 0.15}}, 8, 0.08, colors=[(0.5, 0.5, 0.5)])

    showPredictivity(models_blnext, samples, durations, 'RSVP-predictivity_BLnext_presentation',
                     colors=old_colors[:2])  # blnext_colors)
    showPredictivity(controlModels, samples, durations, 'RSVP-predictivity_controls', colors=control_colors)

    comparePredictivity_samples(models, samples, durations, colors=all_colors)
    comparePredictivity_VE(models, samples, durations, colors=all_colors)


    sampleStats = reportSampleStats()
    _logger.info(sampleStats)

    dPrime = reportdPrimeStats()
    N_images = countCategoryExamples()



if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    paper_figures()

