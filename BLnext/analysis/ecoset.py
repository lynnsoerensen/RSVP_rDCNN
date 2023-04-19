from sklearn.metrics import top_k_accuracy_score

from pathlib import Path
import sys
sys.path.extend([Path(__file__).parent.parent])
import os
import logging
import numpy as np
import pandas as pd
import joblib

from BLnext.models.blnext import makeModel_blnext
from BLnext.models.controls import makeModel_control
from BLnext.models.controls_adaptation import makeModel_control_adaptation


from BLnext.analysis.trials import loadPretrainedCategories
from rcnn_sat import preprocess_image
from keras_preprocessing import image
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns


cm = 1 / 2.54
_logger = logging.getLogger(__name__)



def loadImage_centerCrop(path, targetSize = 128, shortSide=146):
    item = image.load_img(path)  # , target_size=(128, 128))

    width, height = item.size

    if height == min(width, height):
        height_target = shortSide
        width_target = int(np.round((shortSide / height) * width))
    elif width == min(width, height):
        width_target = shortSide
        height_target = int(np.round((shortSide / width) * height))

    item = item.resize((width_target, height_target), resample= Image.NEAREST)

    # center crop
    left = (width_target - targetSize) // 2
    top = (height_target - targetSize) // 2
    right = left + targetSize
    bottom = top + targetSize
    item = item.crop((left, top, right, bottom))
    return item


def makeTrial(design, trial,samples, num_images = 6, offset=4, image_dir='/mnt/Googolplex/ecoset/test'):

    images = []
    targets = []
    for n in range(num_images):

        item = loadImage_centerCrop(image_dir + f"/{design.loc[trial, 'Category' + str(n + 1)]}" + '/' + f"{design.loc[trial, 'Item' + str(n + 1)]}")

        item = image.img_to_array(item, dtype=np.uint8)
        item = preprocess_image(np.expand_dims(item, axis=0))
        [images.append(item) for j in range(samples)]
        [targets.append(design.loc[trial, 'CategoryID' + str(n + 1)]) for _ in range(samples)]

    [images.append(preprocess_image(np.ones((1, 128, 128, 3), dtype=np.uint8) * 128)) for _ in range(offset)]
    [targets.append(design.loc[trial, 'CategoryID' + str(n + 1)]) for _ in range(offset)]

    return images, targets

def getDesign(n=500, num_images=6, image_dir='/mnt/Googolplex/ecoset/test', seed=None, filename=None):

    if filename.is_file():
        design = pd.read_pickle(filename)

    else:
        if seed != None:
            np.random.seed(seed)

        # load all possible categories
        categories = loadPretrainedCategories('ecoset')

        design = []
        for t in range(n):

            trial = {}
            trial['Trial'] = t
            for i in range(num_images):
                # pick a random category
                current_catID = np.random.randint(len(categories))

                if i > 0:
                    while trial['CategoryID'  + str(i)] == current_catID:
                        current_catID = np.random.randint(len(categories))

                current_cat = categories.loc[current_catID].values[0]

                image_files = os.listdir(image_dir + f'/{current_cat}')

                current_img = image_files[np.random.randint(len(image_files))]

                trial['CategoryID' + str(i + 1)] = current_catID
                trial['Category' + str(i+1)] = current_cat

                trial['Item' + str(i + 1)] = current_img

            design.append(trial)

        design = pd.DataFrame(design)
        if filename != None:
            pd.to_pickle(design, filename)

    return design




def perform_isolatedImages(model_name, dataset='ecoset', alphas=[0], beta=0):

    # load the design
    num_images = 6
    design = getDesign(n=500, num_images=num_images,
                       filename=Path(__file__).parent.parent / 'results' / 'EcosetDesign.pkl')

    # load the model
    if beta == 0:
        model = makeModel_control(model_name,  trainingSet=dataset)
    else:
        # Control models with adaptation
        model = makeModel_control_adaptation(model_name, trainingSet=dataset, alphas=alphas, beta=beta)

    np.random.seed(5)
    if model_name.startswith('bl'):
        accuracies_top1 = np.zeros((len(design), 8))
        accuracies_top5 = np.zeros((len(design), 8))
    else:
        accuracies_top1 = np.zeros((len(design)))
        accuracies_top5 = np.zeros((len(design)))

    # loop over trials
    for idx in range(len(design)):
        #images = makeTrial(design, idx, 1, offset=0)
        #predictions = model.predict(np.squeeze(np.array(images)))

        images, targets = makeTrial(design, idx, 1, offset=0)
        predictions = model.predict(np.squeeze(np.array(images)))

        # Compute accuracies

        if model_name.startswith('bl'):

            for t in range(len(predictions)):
                accuracies_top1[idx, t] = top_k_accuracy_score(np.array(targets), np.squeeze(predictions[t]), k=1,
                                                       labels=np.arange(predictions[0].shape[-1], dtype='int'))
                accuracies_top5[idx, t] = top_k_accuracy_score(np.array(targets), np.squeeze(predictions[t]), k=5,
                                                               labels=np.arange(predictions[0].shape[-1], dtype='int'))
        else:
            accuracies_top1[idx] = top_k_accuracy_score(np.array(targets), predictions, k=1,
                                                           labels=np.arange(predictions.shape[-1], dtype='int'))
            accuracies_top5[idx] = top_k_accuracy_score(np.array(targets), predictions, k=5,
                                                           labels=np.arange(predictions.shape[-1], dtype='int'))


    tracker = {}

    tracker['accuracy_top1'] = accuracies_top1
    tracker['accuracy_top5'] = accuracies_top5

    return tracker


def perform_sequentialImages(model_name, samples, offset=4, dataset='ecoset',alphas=[0], beta=0):

    # load the design
    num_images = 6
    design = getDesign(n=500, num_images=num_images, filename=Path(__file__).parent.parent / 'results' /'EcosetDesign.pkl')

    # load the model
    trial_duration = samples * num_images + offset
    model = makeModel_blnext(n_times=trial_duration,  beta=beta, alpha=alphas, trainingSet=dataset)
    model_base = makeModel_control('bl',  trainingSet=dataset)
    np.random.seed(5)
    accuracies_top5 = np.zeros(len(design))
    accuracies_top1 = np.zeros(len(design))
    imageCorrs = np.zeros((len(design), num_images, trial_duration))

    # loop over trials
    for idx in range(len(design)):
        images, targets = makeTrial(design, idx, samples)
        predictions = model.predict(images)

        # Compute accuracies
        accuracies_top5[idx] = top_k_accuracy_score(np.array(targets), np.squeeze(predictions), k=5, labels=np.arange(predictions[0].shape[-1], dtype='int'))
        accuracies_top1[idx] = top_k_accuracy_score(np.array(targets), np.squeeze(predictions), k=1,
                                                    labels=np.arange(predictions[0].shape[-1], dtype='int'))

        # Quality
        images_base, targets_base = makeTrial(design, idx, 1, offset=0)
        predictions_base = model_base.predict(np.squeeze(np.array(images_base)))

        for target in range(num_images):
            for t in range(trial_duration):
                imageCorrs[idx, target, t] = np.corrcoef(predictions[t], predictions_base[-1][target])[0, 1]

    tracker = {}

    tracker['accuracy_top1'] = accuracies_top1
    tracker['accuracy_top5'] = accuracies_top5

    tracker['imageCorrelations'] = imageCorrs

    return tracker


def collectPerformance(models, samples):
    data = []
    for model_name in models:
        model = models[model_name]['model']
        for sample in samples:

            if model in ['b', 'b_d', 'bl']:  # Only evaluate these models once since they don't have any time steps.
                sample_current = 1
            else:
                sample_current = sample

            if (model =='bl') & (sample >= 8):
                pass
            else:
                file_name = Path(
                    __file__).parent.parent / 'results' / f"Ecoset-performance_{model_name}_Samples-{sample_current}.pkl"

                if file_name.is_file():
                    # Check if this exists already
                    tracker = joblib.load(file_name)
                else:
                    # Otherwise, run the experiment
                    print('Evaluating ' + model_name + ' model with ' + str(sample) + ' samples ')
                    if model == 'bl':
                        tracker = perform_isolatedImages(model, alphas=models[model_name]['Alpha'],
                                                           beta=models[model_name]['Beta'])
                    elif (model in ['b', 'b_d']) & (models[model_name]['Beta'] == 0):

                        tracker = perform_isolatedImages(model, alphas=models[model_name]['Alpha'],
                                                         beta=models[model_name]['Beta'])

                    else:
                        tracker = perform_sequentialImages(model, sample, alphas=models[model_name]['Alpha'],
                                                           beta=models[model_name]['Beta'])

                    tracker['model'] = model_name
                    tracker['samples'] = sample
                    joblib.dump(tracker, file_name, compress=True)

                if model == 'bl':
                    data.append(
                        {'Model': model_name, 'Samples': sample, 'accuracy_top1': tracker['accuracy_top1'][:, sample-1].mean(),
                         'accuracy_top5': tracker['accuracy_top5'][:, sample-1].mean()})
                else:
                    data.append({'Model': model_name, 'Samples': sample, 'accuracy_top1': tracker['accuracy_top1'].mean(),
                             'accuracy_top5': tracker['accuracy_top5'].mean()})

    return pd.DataFrame(data)

def collectItemCorrs(models, samples):
    data = []
    for model in models:

        for sample in samples:

            if model in ['b', 'b_d', 'bl']:  # Only evaluate these models once since they don't have any time steps.
                pass
            else:
                file_name = Path(
                    __file__).parent.parent / 'results' / f"Ecoset-performance_{model}_Samples-{sample}.pkl"

                if file_name.is_file():
                    # Check if this exists already
                    tracker = joblib.load(file_name)
                else:
                    # Otherwise, run the experiment
                    print('Evaluating ' + model + ' model with ' + str(sample) + ' samples ')
                    if model in ['b', 'b_d', 'bl']:
                        tracker = perform_isolatedImages(model, sample)
                    else:
                        tracker = perform_sequentialImages(model, sample, alphas=models[model]['Alpha'],
                                                           beta=models[model]['Beta'])

                    tracker['model'] = model
                    tracker['samples'] = sample
                    joblib.dump(tracker, file_name, compress=True)

                data.append({'Model': model, 'Samples': sample, 'imageCorrelations': tracker['imageCorrelations'].max(axis=-1).mean()})

    return pd.DataFrame(data)



def showPerformanceEcoset(models, samples, figureName, data=None, colors=None):
    data = data if data is not None else collectPerformance(models, samples)

    fig_dir = Path(__file__).parent.parent / 'figures'

    # Make figure
    if colors is not None:
        sns.set_palette(np.array(colors))

    markers = ['o' for m in range(len(models))]

    fig, ax = plt.subplots(figsize=(6.5*cm,8*cm), sharey=True)#(6.3*cm, 6.8*cm))

    sns.lineplot(data=data, style='Model', hue='Model', x='Samples', y='accuracy_top5', ax=ax,
                 dashes=False, markers=markers, legend=False)

    ax.set_xticks([1, 2, 4, 6, 8, 10, 12])
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])

    #ax.set_ylim([-0.15, 3])
    ax.set_ylabel("Top5 accuracy")
    ax.set_xlabel("Model steps/image")

    ax.axhline(1/565, c='gray', ls='--', alpha=0.4)

    ax.set_title('Object recognition\n(Ecoset test set)', fontweight='bold', fontsize=10)
    sns.despine()
    plt.tight_layout()

    plt.savefig(fig_dir/ f"{figureName}.pdf", dpi=300, transparent=True)
    plt.savefig(fig_dir/ f"{figureName}.png")


def showPerformanceEcoset_v2(models, samples, figureName, data=None, colors=None, dashes = False ):
    data = data if data is not None else collectPerformance(models, samples)

    fig_dir = Path(__file__).parent.parent / 'figures'

    # Make figure
    if colors is not None:
        sns.set_palette(np.array(colors))

    markers = ['o' for m in range(len(models))]

    fig, ax = plt.subplots(figsize=(6.5*cm,8*cm), sharey=True)#(6.3*cm, 6.8*cm))

    sns.lineplot(data=data, style='Model', hue='Model', x='Samples', y='accuracy_top5', ax=ax,
                 dashes=dashes, markers=markers, legend=False)

    ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8])#, 10, 12])
    ax.set_yticks([0.6, 0.7, 0.8, 0.9, 1])

    #ax.set_ylim([-0.15, 3])
    ax.set_ylabel("Top5 accuracy")
    ax.set_xlabel("Model steps/image")

    #ax.axhline(1/565, c='gray', ls='--', alpha=0.4)

    ax.set_title('Object recognition\n(Ecoset test set)', fontweight='bold', fontsize=10)
    sns.despine()
    plt.tight_layout()

    plt.savefig(fig_dir/ f"{figureName}.pdf", dpi=300, transparent=True)
    plt.savefig(fig_dir/ f"{figureName}.png")



def showItemCorrsEcoset(models, samples, figureName, data=None, colors=None):
    data = data if data is not None else collectItemCorrs(models, samples)

    fig_dir = Path(__file__).parent.parent / 'figures'

    # Make figure
    if colors is not None:
        sns.set_palette(np.array(colors))

    markers = ['o' for m in range(len(models))]

    fig, ax = plt.subplots(figsize=(6.5*cm,8*cm), sharey=True)#(6.3*cm, 6.8*cm))

    sns.lineplot(data=data, style='Model', hue='Model', x='Samples', y='imageCorrelations', ax=ax,
                 dashes=False, markers=markers, legend=False)

    ax.set_xticks([1, 2, 4, 6, 8, 10, 12])
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1])

    #ax.set_ylim([-0.15, 3])
    ax.set_ylabel("Representational strength")
    ax.set_xlabel("Model steps/image")

    ax.axhline(1, c='gray', ls=':', alpha=0.4)

    ax.set_title('Image recovery\n(Ecoset test set)', fontweight='bold', fontsize=10)
    sns.despine()
    plt.tight_layout()

    plt.savefig(fig_dir/ f"{figureName}.pdf", dpi=300, transparent=True)
    plt.savefig(fig_dir/ f"{figureName}.png")
