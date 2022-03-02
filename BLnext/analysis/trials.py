from pathlib import Path
import sys
sys.path.extend([Path(__file__).parent.parent])
import logging
import joblib
import numpy as np
import pandas as pd
from rcnn_sat import preprocess_image
from keras_preprocessing import image
from BLnext.models.blnext import makeModel_blnext
from BLnext.models.controls import makeModel_control

import matplotlib.pyplot as plt
import seaborn as sns

cm = 1 / 2.54
_logger = logging.getLogger(__name__)

def loadRSVPdesign(skip_practiceTrials = True, practice=11):
    design = pd.read_excel(Path(__file__).parent.parent.parent / 'resources' / 'stimuli' / 'FD6GS-StimFile1-withHeader_altLabels_word2vecLabels.xlsx', 'Sheet1',
                           index_col=None, na_values=['NA'], engine='openpyxl')
    if skip_practiceTrials:
        design = design.loc[practice:].reset_index()

    return design

def loadTrial(design, trial,samples, num_images = 6, offset=4):
    image_dir = Path(__file__).parent.parent.parent / 'resources' / 'stimuli' / 'stimuliCropped'

    images = []
    for n in range(num_images):
        item = image.load_img(image_dir / f"{design.loc[trial, 'item' + str(n + 1)]}.jpg", target_size=(128, 128))

        item = image.img_to_array(item, dtype=np.uint8)
        item = preprocess_image(np.expand_dims(item, axis=0))
        [images.append(item) for j in range(samples)]

    [images.append(preprocess_image(np.ones((1, 128, 128, 3), dtype=np.uint8) * 128)) for i in range(offset)]
    return images

def loadPretrainedCategories(training_set='ecoset'):
    df = pd.read_csv(Path(__file__).parent.parent.parent / 'resources' / 'rcnn_sat' / 'pretrained_output_categories'/ f'{training_set}_categories.txt', sep=" ", header=None)

    return df.rename(columns={0 :'categories'})

def translatePretrainedCategories(categories, training_set='ecoset'):
    pretrainedCategories = loadPretrainedCategories(training_set=training_set)

    # get rid of leading numbers
    pretrainedCategories['categoryStrings'] = pretrainedCategories['categories'].str[5:]

    category_ids = []
    for category in categories:
        category_id = pretrainedCategories.loc[pretrainedCategories['categoryStrings'] ==category].index.tolist()

        if len(category_id) == 1:
            category_ids.append(category_id[0])
        else:
            raise ValueError('The indicated category ' + category + ' does not exist. Choose of these categories:\n' + str(pretrainedCategories['categoryStrings'].to_list()))

    return category_ids

def traceImage_sequentialImages(model_name, samples, trial_id, images, alphas=[0], beta=0, num_images = 6, offset=4):
    design = loadRSVPdesign(skip_practiceTrials = False)
    trial = loadTrial(design, trial_id, samples, num_images=num_images, offset=offset)

    trial_duration = samples * num_images + offset
    model = makeModel_blnext(n_times=trial_duration, beta=beta, alpha=alphas)
    model_base = makeModel_control('bl')

    predictions = model.predict(trial)
    predictions_base = model_base.predict(np.array(trial).squeeze())

    traces = np.zeros((len(images), trial_duration))
    for i, image in enumerate(images):

        for t in range(trial_duration):
            traces[i, t] = np.corrcoef(predictions[t], predictions_base[7][samples * image])[0, 1]

    return traces

def traceCategory_sequentialImages(model_name, samples, trial_id, category_ids, alphas=[0], beta=0, num_images = 6, offset=4):
    design = loadRSVPdesign(skip_practiceTrials = False)
    trial = loadTrial(design, trial_id, samples, num_images=num_images, offset=offset)

    trial_duration = samples * num_images + offset
    model = makeModel_blnext(n_times=trial_duration, beta=beta, alpha=alphas)

    predictions = model.predict(trial)

    traces = np.zeros((len(category_ids), trial_duration))
    for c, category_id in enumerate(category_ids):

        target_vector = np.zeros(predictions[0].shape[1])
        target_vector[category_id] = 1

        for t in range(trial_duration):
            traces[c, t] = np.corrcoef(predictions[t], target_vector)[0, 1]

    return traces


def tracePrototype_sequentialImages(model_name, samples, trial_id, alphas=[0], beta=0, num_images = 6, offset=4,
                                    dataset='ecoset', prototype='sel10'):
    design = loadRSVPdesign(skip_practiceTrials = False)
    images = loadTrial(design, trial_id, samples, num_images=num_images, offset=offset)

    trial_duration = samples * num_images + offset
    model = makeModel_blnext(n_times=trial_duration, beta=beta, alpha=alphas)

    # load the protoType images
    protoTypes = joblib.load(
        Path(__file__).parent.parent / 'prototypes' / 'activations' / f"manifolds500_bl_FC_{dataset}_{prototype}.pkl")

    predictions = model.predict(images)

    centroid = protoTypes[design.loc[trial_id, 't']].mean(axis=0)

    trace = np.zeros((trial_duration))
    for t in range(trial_duration):
        trace[t] = np.corrcoef(predictions[t], centroid)[0, 1]

    return trace


def collectImageTraces(models, samples, trial, images):
    traces = {}
    for model in models:
        print('Collecting trace for ' + model + ' model.')
        traces[model] = traceImage_sequentialImages(model, samples, trial, images, alphas=models[model]['Alpha'],
                                            beta=models[model]['Beta'])

    return traces

def collectCategoryTraces(models, samples, trial, categories):
    category_ids = translatePretrainedCategories(categories)

    traces = {}
    for model in models:
        print('Collecting trace for ' + model + ' model.')
        traces[model] = traceCategory_sequentialImages(model, samples, trial, category_ids, alphas=models[model]['Alpha'],
                                            beta=models[model]['Beta'])

    return traces



def collectPrototypeTraces(models, samples, trial):
    traces = {}
    for model in models:
        print('Collecting trace for ' + model + ' model.')
        traces[model] = tracePrototype_sequentialImages(model, samples, trial, alphas=models[model]['Alpha'],
                                            beta=models[model]['Beta'])

    return traces

def showExampleImageTrace(models, image=1, samples=6, trial=5, colors=None, data=None):
    data = data if data is not None else collectImageTraces(models, samples, trial, [image])

    fig_dir = Path(__file__).parent.parent / 'figures'

    if colors is not None:
        sns.set_palette(np.array(colors))

    time_steps = data[list(models)[0]].shape[1]

    x_ticks = [t * samples for t in range(7)]
    plt.figure(figsize=(8*cm, 4.5*cm))

    plt.axvspan(samples * image, samples * (image+1), color='gray', alpha=0.5)

    plt.axvspan(0, samples * image, color='gray', alpha=0.2)
    plt.axvspan(samples * (image+1), time_steps, color='gray', alpha=0.2)

    for model in models:
        plt.plot(np.arange(time_steps), data[model])

    plt.yticks([0, 0.5, 1])
    plt.xticks(x_ticks)
    plt.xlabel('Model steps')
    plt.ylabel('Representational\nstrength')
    plt.xlim([0, time_steps])
    sns.despine()
    plt.tight_layout()

    plt.savefig(fig_dir / 'ImageTrace_Example.pdf', dpi=300, transparent=True)
    plt.savefig(fig_dir / 'ImageTrace_Example.png')

def showExampleCategoryTrace(model, category='flower', samples=4, trial=5, matchingImage=None, colors=None, data=None):

    data = data if data is not None else collectCategoryTraces(model, samples, trial, [category])

    fig_dir = Path(__file__).parent.parent / 'figures'

    if colors is not None:
        sns.set_palette(np.array(colors))

    time_steps = data[list(model)[0]].shape[1]

    x_ticks = [t * samples for t in range(7)]
    plt.figure(figsize=(5.6 * cm, 5 * cm))
    if matchingImage is not None:
        target_color = colors[matchingImage]
        plt.axvspan(samples * matchingImage, samples * (matchingImage+1), color=target_color, alpha=0.1)

        not_shown_images = np.setdiff1d(np.arange(6), matchingImage)
        for n in not_shown_images:
            plt.axvspan(samples * n, samples * (n + 1), alpha=0.1, color='grey')


    else:
        target_color = 'grey'

    plt.plot(np.arange(time_steps), data[list(model)[0]][0], color=target_color)

    plt.yticks([0, 1])
    #plt.ylim([0, 0.2])
    plt.xticks(x_ticks)
    plt.xlabel('Model steps')
    plt.ylabel('Representational\nstrength')
    plt.xlim([0, time_steps])
    sns.despine()
    plt.tight_layout()

    plt.savefig(fig_dir / 'CategoryTrace_Example_softmax.pdf', dpi=300, transparent=True)
    plt.savefig(fig_dir / 'CategoryTrace_Example_softmax.png')

def showExampleImageTraces(model, figurename, images=np.arange(6), samples=6, trial=5, colors=None, data=None):
    data = data if data is not None else collectImageTraces(model, samples, trial, images)

    fig_dir = Path(__file__).parent.parent / 'figures'

    if colors is not None:
        sns.set_palette(np.array(colors))

    time_steps = data[list(model)[0]].shape[1]

    x_ticks = [t * samples for t in range(7)]
    plt.figure(figsize=(5.6*cm, 5*cm))

    for image in images:
        plt.axvspan(samples * image, samples * (image+1), alpha=0.2, color=colors[image])
        plt.plot(np.arange(time_steps), data[list(model)[0]][image], color=colors[image])

    if len(images) < 6:
        not_shown_images = np.setdiff1d(np.arange(6),images)
        for n in not_shown_images:
            plt.axvspan(samples * n, samples * (n + 1), alpha=0.1, color='grey')

    plt.yticks([0, 1])
    plt.xticks(x_ticks)
    plt.xlabel('Model steps')
    plt.ylabel('Representational\nstrength')
    plt.xlim([0, time_steps])
    plt.ylim([0, 1])
    sns.despine()
    plt.tight_layout()

    plt.savefig(fig_dir / f"{figurename}.pdf", dpi=300, transparent=True)
    plt.savefig(fig_dir / f"{figurename}.png")

    return data



def showExamplePrototypeTrace(models, samples=6, trial=5, colors=None, data=None):
    data = data if data is not None else collectPrototypeTraces(models, samples, trial)

    fig_dir = Path(__file__).parent.parent / 'figures'

    if colors is not None:
        sns.set_palette(np.array(colors))

    time_steps = data[list(models)[0]].shape[0]

    x_ticks = [t * samples for t in range(7)]
    plt.figure(figsize=(5.5*cm, 4*cm))

    for model in models:
        plt.plot(np.arange(time_steps), data[model], ls='--')
    for x in x_ticks:
        plt.axvline(x, color='gray',alpha=0.2)

    plt.yticks([0, 1])
    plt.xticks(x_ticks)
    plt.xlabel('Model steps')
    plt.ylabel('Representational\nstrength')
    plt.xlim([0, time_steps])
    plt.ylim([0, 1])
    sns.despine()
    plt.tight_layout()

    plt.savefig(fig_dir / 'PrototypeTrace_Example.pdf', dpi=300, transparent=True)
    plt.savefig(fig_dir / 'PrototypeTrace_Example.png')

