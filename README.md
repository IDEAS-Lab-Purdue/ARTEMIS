# ARTEMIS
IROS 2024 ARTEMIS Codebase

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

## `models`
- The Random Forest model was too large to upload here
- The SVM one over one model was not saved either

## `yale`

### `scripts`
Contains all the python scripts and ipy notebooks used to preprocess the MIMIC dataset and train the 5-Layer MLP. 

- `ARTEMIS.ipynb`: A Random Forest was also attempted on MIMIC. We also attempted to

### `figs`
