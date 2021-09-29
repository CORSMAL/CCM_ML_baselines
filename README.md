# Audio-based classification of the content in a container: Baselines

This repository contains 12 uni-modal baselines that use only audio as input data 
to solve the classification of filling level and filling type on the 
[CORSMAL Containers Manipulation](http://corsmal.eecs.qmul.ac.uk/containers_manip.html) 
dataset at the [CORSMAL Challenge](https://corsmal.eecs.qmul.ac.uk/challenge.html).

The baselines compute different types of features from the input signals and 
provide the features as input to three classifiers, namely k-Nearest Neighbour (kNN),
Support Vectot Machine (SVM), and Random Forest (RF). 
All these baselines classify filling type and level jointly. 

The features considered are spectrograms, Zero-Crossing Rate (ZRC), 
Mel-Frequency Cepustrum Coefficients (MFCC), chromagram, mel-scaled spectrogram, 
spectral contrast, and tonal centroid features (tonnetz). For MFCC, the 1st to 
the 13th coefficients are used, while the 0th coefficient is discarded.

Similar to the baselines for the [Environmental Sound Classification](https://github.com/karolpiczak/ESC-50), 
3 baselines use the mean and standard deviation of the MFCC and ZCR features 
across multiple audio frames. Other 3 baselines extract a feature vector 
consisting of 193 coefficients from the mean and standard deviation of the MFCC, 
chromagram, mel-scaled spectrogram, spectral contrast, and tonnetz across 
multiple audio frames. 3 baselines compute 2D spectrograms from the input audio
and reshape the spectrogram into a vector of dimension 9,216 (see [ACC](https://github.com/CORSMAL/ACC)).
Three baselines perform dimensionality reduction with Principal Component 
Analysis on the reshaped spectrograms to remove redundant and less relevant information. 
The first 128 components are retained for these baselines.


[[arXiv](https://arxiv.org/abs/2107.12719)]
[[webpage](http://corsmal.eecs.qmul.ac.uk/containers_manip.html)]
[[CCM dataset](http://corsmal.eecs.qmul.ac.uk/containers_manip.html)]
[[ACM-S2 dataset](https://zenodo.org/record/4770439#.YKPacSbTU5k)]
[[evaluation toolkit](https://github.com/CORSMAL/CORSMALChallengeEvalToolkit)]

---

> Legend:
>   - MFCC: Mel-Frequency Cepustrum Coefficients
>   - ZCR: Zero-Crossing Rate
>   - tonnetz: tonal centroid features
>   - A5F: audio 5 features (MFCC, 
chromagram, mel-scaled spectrogram, spectral contrast, and tonnetz)
>   - PCA: Principal Component Analysis
>   - kNN: k-Nearest Neighbour
>   - SVM: Support Vector Machine
>   - RF: Random Forest

Features | Classifier 
---------|------------
MFCC + ZCR | kNN
MFCC + ZCR | SVM
MFCC + ZCR | RF
A5F | kNN
A5F | SVM
A5F | RF
Reshaped spectrogram | kNN
Reshaped spectrogram | SVM
Reshaped spectrogram | RF
Reshaped spectrogram + PCA | kNN
Reshaped spectrogram + PCA | SVM
Reshaped spectrogram + PCA | RF

---

## Installation

### Requirements
* python=3.8.3
* librosa=0.8.0
* pandas=1.2.3
* scikit-image=0.17.2
* scikit-learn=0.24.1
* tqdm=4.60.0

### Instructions
0. Clone repository

<code>git clone https://github.com/CORSMAL/ml_baselines</code>

1. From a terminal or an Anaconda Prompt, go to project's root directory and run: 

<code>conda env create -f environment.yml</code>

<code>conda activate ml_basel_corsmal</code>

This will create a new conda environment (<code>ml_basel_corsmal</code>) and install all software dependencies.

If this step failed, in a Linux machine (also from the project's root directory) run the following commans and try again:

* <code>module load anaconda3</code>
* <code>source activate</code>

**Alternatively** or if the installation file failed, run the following lines for a manual installation:

* <code>conda create --name ml_basel_corsmal python=3.8.3</code>
* <code>conda activate ml_basel_corsmal</code>
* <code>conda install -c conda-forge librosa==0.8.0</code>
* <code>conda install -c conda-forge pandas==1.2.3</code>
* <code>conda install -c conda-forge scikit-learn==0.24.1</code>
* <code>conda install -c conda-forge scikit-image==0.18.1</code>
* <code>conda install -c conda-forge tqdm==4.60.0</code>

---
## Demo
Once finished with the installation, run the demo to check that everything is installed correctly and works as expected. Note that this repository contains everything needed to run the demo.
The demo will check:
- Environment and dependencies
- Data loading
- Data pre-processing
- Models initialization
- Pre-trained models loading

To run the demo, simply execute:

<code>python demo.py</code>

The demo will show if each test runs correctly, ending with the message `[i] end_of_script`. Otherwise, the installation has not been successful.

---

## Run baselines
(From the project's root directory):

The only setup in the code is to point to the [CCM dataset](http://corsmal.eecs.qmul.ac.uk/containers_manip.html). Just add the path to the dataset in: <code>ml_eval.py line 45</code>. Then, run:

<code>python ml_eval.py --data [val, putest, prtest, sec_setup] --features [spectro, pca, esc, full] </code>

- --data: CCM datasplit for the evaluation. Validation is a random 20% of the training data. For sec_setup, it is required additional data that can be found [here](https://zenodo.org/record/4770439#.YPb0pjMYBPZ)
- --features: set of features to extract from the data, explained below.

### Features options:
- <code>spectro</code>: *Reshaped spectrogram*
- <code>pca</code>: *Reshaped spectrogram + PCA*
- <code>esc</code>: *MFCC + ZCR*
- <code>full</code>: *A5F*

This will first generate the preprocessed data as .npy files in `./data/` (if neccesary). This will take some minutes.
Then it will run and export the evaluation for the 3 ML baselines (KNN, SVM, and RF). The estimation CSVs will be exported to `./output/`. The evaluation toolkit can be found [here.](https://github.com/CORSMAL/CORSMALChallengeEvalToolkit)


### Note for private test evaluations
In line 41 from files `feat_hand.py` and `feat_spect.py` (`./script/`)
add the name of the directory where the private test data is located within the CCM dataset, if you have access to it.

### Data annotations
The manual annotations (`./data/man_annotations.csv`) of *start* and *end of action* were collected by the [HVRL team](https://github.com/YuichiNAGAO/ICPRchallenge2020), one of the participants in the [2020 CORSMAL Challenge](http://corsmal.eecs.qmul.ac.uk/ICPR2020challenge.html#results).


---

## Enquiries, Question and Comments

If you have any further enquiries, question, or comments, please contact <email>s.donaher@qmul.ac.uk</email> or <email>a.xompero@qmul.ac.uk</email>. 
If you would like to file a bug report or a feature request, use the Github issue tracker. 


## License

This work is licensed under the MIT License.  To view a copy of this license, see
[LICENSE](https://opensource.org/licenses/MIT).
