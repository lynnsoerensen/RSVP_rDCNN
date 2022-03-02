# Mechanisms of human dynamic object recognition revealed by sequential deep neural networks
by Lynn K.A. Sörensen, Sander M. Bohté, Heleen A. Slagter, & H. Steven Scholte


![](BLnext/figures/Figures_paper_Figure1.png)
This figure is part of this [preprint]().

### Overview
This is the code to reproduce the results of this [paper](https://www.biorxiv.org/content/). All paper figures and analyses can be reproduced using `main.py`.  

### Dependencies
Implementation for tensorflow (1.12).

All model implementations rely on the [rcnn_sat](https://github.com/cjspoerer/rcnn-sat) repository. Make sure to clone it to a folder in `resources/rcnn_sat`. 
The weights for the pretrained models can be accessed [here](https://osf.io/mz9hw/).


All experimental behavioural data can be downloaded [here](https://osf.io/c9gs8/) and should be placed in a directory `resources/humanBehaviour`.
All analyses result files can be downloaded [here](https://osf.io/c9gs8/). Please make sure to add the files to folder `BLnext/results` to reproduce the Figures using `__main__.py`.


