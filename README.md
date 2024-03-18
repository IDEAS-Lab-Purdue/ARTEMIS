# ARTEMIS - AI-driven Robotic Triage Labeling and Emergency Medical Information System

Video: [youtube.com/watch?v=4FU4FRxNwmY](https://www.youtube.com/watch?v=4FU4FRxNwmY)

site: [hrishikeshvish.github.io/assets/projects/artemis.html](https://hrishikeshvish.github.io/assets/projects/artemis.html)

ArXiv: [arxiv.org/abs/2002.09405](https://arxiv.org/abs/2002.09405)

If you use the code here please cite this paper:

    @inproceedings{sanchezgonzalez2020learning,
      title={Learning to Simulate Complex Physics with Graph Networks},
      author={Alvaro Sanchez-Gonzalez and
              Jonathan Godwin and
              Tobias Pfaff and
              Rex Ying and
              Jure Leskovec and
              Peter W. Battaglia},
      booktitle={International Conference on Machine Learning},
      year={2020}
    }


## Example usage: Deploy Champ framework for simulating MCIs in Gazebo

![Earthquake](demos/indoor.gif)
![Construction](demos/test2.gif)

## `mimic`
Contains `scripts` and `figs`. Works on the `triage.csv` table from the [MIMIC-IV-ED database](https://physionet.org/content/mimic-iv-ed/2.2/).

### `scripts`
Contains all the python scripts and ipy notebooks used to preprocess the MIMIC dataset and train models. 

#### Phase 1: Data has not been normalized and has not been synthetically augmented using SMOTE
- `ARTEMIS.ipynb`: We read in and visualize the data. We trained a Random Forest to classify the acuity level of patients given vital signs. We also used embeddings generated for the `pain` attribute (which is textual) using OpenAI's text-embedding-3-small API to train an MLP.
- `generate_embeddings.py`: Generates embeddings for the `pain` attribute (which is textual) using OpenAI's text-embedding-3-small API.
- `mlp.py` and `train.py` together define a model and train an MLP on the dataset with the new `pain` embeddings as well as the original vital signs.

#### Phase 2: Data has not been normalized but has been synthetically augmented using SMOTE
- `mlp_smote.py` and `train_smote.py` together define a model and train an MLP on a dataset that has been synthetically augmented using under-sampling and over-sampling (SMOTE) strategies.

#### Phase 3: Data has both been normalized and been synthetically augmented using SMOTE
- `NN_smote_mimic.ipynb`: Performs upsampling of all non-majority classes and then trains the 5-Layer MLP that achieves 59% accuracy.

### `figs`
- Plots for the 5-Layer MLP: accuracy vs epochs, loss vs epochs, confusion matrix

## `yale`
Contains `scripts` and `figs`. Works on the dataset obtained from [Yale School of Medicine](https://www.kaggle.com/datasets/maalona/hospital-triage-and-patient-history-data) (to be referred to here as Y-MED).

### `scripts`
Contains all the ipy notebooks used to preprocess Y-MED and train the 5-Layer MLP.
-`NN_smote_yale.ipynb` performs synthetic data augmentation of the original Y-MED dataset, along with other data cleaning and normalization.
- Contains other scripts to evaluate MLP, Gaussian Naive Bayes, SVM, Random Forest, and Ensemble models on the data generated through SMOTE.
- A acuity level classification accuracy of 74% was achieved with the MLP.

### `figs`
- Contains plots for the MLP: confusion matrix, loss vs epochs and accuracy vs epochs.
- Contains confusion matrix plots for the other models trained on the Y-MED.

## `models`
- Contains the Gaussian Naive Bayes model and the MLP model trained on the Yale Dataset.
- Contains the MLP model trained on the MIMIC dataset in Phase 3 described above in the `mimic` section.

NB. The Random Forest models were too large to upload here. The SVM one over one model was not saved either.

## `GUI`
- Contains code for the triage display website we created as part of ARTEMIS.
