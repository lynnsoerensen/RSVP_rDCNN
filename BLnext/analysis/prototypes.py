from pathlib import Path
import sys
sys.path.extend([Path(__file__).parent.parent])
import logging
import pandas as pd
import os
import joblib
import numpy as np
import google_images_download

from rcnn_sat import preprocess_image
from keras_preprocessing import image
from BLnext.analysis.trials import loadRSVPdesign
from BLnext.models.controls import makeModel_control

cm = 1 / 2.54
_logger = logging.getLogger(__name__)

def downloadCategoryExamples(limit=500):

    response = google_images_download.googleimagesdownload()  # class instantiation

    output_dir = Path(__file__).parent.parent / 'prototypes' / 'categoryExamples'

    design = loadRSVPdesign()

    print('Downloading images')
    for targets in [design['t']]:  # Loop over categories.
        for category in targets:

            if not (output_dir / f"{category}").is_dir():
                print(category)
                arguments = {"keywords": category,
                             "limit": limit,
                             "print_urls": True,
                             "output_directory": output_dir,
                             "format": "jpg",
                             "color_type": "full-color",
                             "usage_rights": "labeled-for-noncommercial-reuse-with-modification",
                             # "usage_rights": "labeled-for-reuse-with-modifications,labeled-for-reuse,labeled-for-noncommercial-reuse-with-modification,labeled-for-nocommercial-reuse",
                             "type": "photo",
                             "prefix": category,
                             "chromedriver": '/usr/bin/chromedriver',
                             }  # creating list of arguments
                paths = response.download(arguments)  # passing the arguments to the function


def countCategoryExamples(directory=None):
    if directory is None:
        image_dir = Path(directory)
    else:
        image_dir = Path(__file__).parent.parent / 'prototypes' / 'categoryExamples'

    design = loadRSVPdesign()
    N = []

    for category in design['t']:
        print(category)
        directory = image_dir / f"{category}"
        img_names = os.listdir(directory)
        N.append(len(img_names))
        print(len(img_names))
    return N


def obtainPrototypeActivations(model_name, dataset='ecoset'):
    output_dir = Path(__file__).parent.parent / 'prototypes' / 'activations'

    file_name = output_dir /f"activations500_{model_name}_{dataset}.pkl"

    if file_name.is_file():
        activations = joblib.load(file_name)
    else:
        activations={}

    image_dir = Path(__file__).parent.parent / 'prototypes' / 'categoryExamples'

    design = loadRSVPdesign(skip_practiceTrials=False)
    model = makeModel_control(model_name, trainingSet=dataset)

    for category in design['t']:
        if category not in activations:
            print(category)
            directory = image_dir/f"{category}"
            if directory.is_dir():
                img_names = os.listdir(directory)

                img_batch = []
                for img in img_names:
                    item = image.load_img(directory / f"{img}", target_size=(128, 128))
                    item = image.img_to_array(item, dtype=np.uint8)
                    item = preprocess_image(np.expand_dims(item, axis=0))
                    img_batch.extend(item)

                activations[category] = model.predict(np.array(img_batch))

                joblib.dump(activations, file_name, compress=True)
            else:
                downloadCategoryExamples()

    return activations

def obtainPrototypeActivations_selection(model_name, selection=10, dataset='ecoset'):

    output_dir = Path(__file__).parent.parent / 'prototypes' / 'activations'

    if model_name in ['NoAdaptation', 'Exponential', 'Power']:  # For sequential models, replace with the right control model.
        model_name = 'bl'

    file_name = output_dir / f"activations500_{model_name}_FC_{dataset}.pkl"

    if file_name.is_file():
        activations = joblib.load(file_name)
    else:
        activations = obtainPrototypeActivations(model_name)

    image_dir = Path(__file__).parent.parent.parent / 'resources' / 'stimuli' / 'stimuliCropped'
    design = loadRSVPdesign(skip_practiceTrials=False)
    model = makeModel_control(model_name, trainingSet=dataset)

    activations_sel = {}
    targetTraceSimilarity = []
    for idx in range(len(design)):
        predictions = []
        for side in ['L', 'R']:
            item = image.load_img(image_dir / f"{design.loc[idx, side + 'Tname']}.jpg",
                                  target_size=(128, 128))
            item = image.img_to_array(item, dtype=np.uint8)
            item = preprocess_image(np.expand_dims(item, axis=0))

            item_pred = model.predict(item)
            if model_name == 'bl':
                predictions.extend(item_pred[7])
            else:
                predictions.extend(item_pred)

        if model_name == 'bl':
            rdm = np.corrcoef(np.vstack([np.mean(predictions, axis=0), activations[design.loc[idx, 't']][7]]))

        else:
            rdm = np.corrcoef(np.vstack([np.mean(predictions, axis=0), activations[design.loc[idx, 't']]]))

        selected_manifolds = np.argsort(rdm[0])[-(selection + 1):-1] - 1  # +1 because the RDM still has the diagonal

        if model_name == 'bl':
            activations_sel[design.loc[idx, 't']] = activations[design.loc[idx, 't']][7][selected_manifolds]
        else:
            activations_sel[design.loc[idx, 't']] = activations[design.loc[idx, 't']][selected_manifolds]
        print(design.loc[idx, 't'])
        print(np.mean(rdm[0][selected_manifolds + 1]))
        targetTraceSimilarity.append(rdm[0][selected_manifolds + 1])

    activations_sel['TargetTraceSimilarity'] = targetTraceSimilarity

    joblib.dump(activations_sel, output_dir / f"activations500_{model_name}_FC_{dataset}_sel{selection}.pkl",
                compress=True)

    return activations_sel




