# Unsupervised Anomaly Detection in LHC Event Data

**Saqib Umair**  
MSc Particle and Nuclear Physics, University of Edinburgh

This project uses a PyTorch autoencoder to test whether Beyond Standard Model (BSM) event samples can be separated from Standard Model (SM) background using reconstruction error alone.

The network is trained only on SM events. At inference time, the mean-squared reconstruction error is used as the anomaly score: events that are poorly reconstructed are treated as less SM-like.

## Dataset representation

Each event is converted into a 39-element feature vector:

- 5 particle counts: electrons, muons, jets, b-jets and photons;
- missing transverse energy: `log(MET)` and `METphi`;
- the 8 highest-energy reconstructed particles, each represented by `log(E)`, `log(pT)`, `eta` and `phi`.

The SM sample contains **340,268 events**. An 80/20 split gives:

- **272,214 training events**
- **68,054 test events**

The scaler is fitted on the SM training sample only and then applied to the SM test set and all BSM samples.

## Autoencoder

The model is deliberately compact:

```text
39 inputs -> 32 -> 8 latent features -> 32 -> 39 outputs
```

ReLU activations are used in the hidden layers and a sigmoid output is used because the input features are scaled to `[0, 1]`.

Training uses Adam with a learning rate of `1e-3`, mean-squared reconstruction loss and 30 epochs. The best saved validation loss in the submitted run was **0.004652**.

![Training and validation loss](figures/training_loss_plot.png)

## Anomaly score

For each event,

```text
anomaly score = mean((reconstruction - input)^2)
```

The same score is evaluated for the held-out SM test sample and nine BSM benchmark samples.

## ROC performance

![Combined ROC curves](figures/roc_curves_combined.png)

The response is strongly signal-dependent:

| BSM sample | ROC AUC |
|---|---:|
| gluino_1000.0_neutralino_1.0 | **0.982** |
| stlp_st1000 | **0.957** |
| chacha_cha600_neut200 | 0.679 |
| chacha_cha400_neut60 | 0.636 |
| pp24mt_50 | 0.621 |
| chaneut_cha200_neut50 | 0.578 |
| chaneut_cha250_neut150 | 0.570 |
| chacha_cha300_neut140 | 0.557 |
| pp23mt_50 | **0.490** |

The autoencoder separates the gluino and stau-like benchmark samples well, but several other signals overlap strongly with the SM background. The `pp23mt_50` sample is effectively indistinguishable from background with this reconstruction-error score.

That spread in AUC is an important result: a background-trained autoencoder does not provide uniform sensitivity to every BSM topology.

## Example anomaly-score distributions

### Strong separation: gluino benchmark

![SM versus gluino anomaly score](figures/anomaly_scores/anomaly_histogram_gluino_1000.0_neutralino_1.0.png)

### Strong separation: stlp benchmark

![SM versus stlp anomaly score](figures/anomaly_scores/anomaly_histogram_stlp_st1000.png)

### Moderate separation: chacha benchmark

![SM versus chacha anomaly score](figures/anomaly_scores/anomaly_histogram_chacha_cha600_neut200.png)

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── notebooks/
│   └── lhc_bsm_autoencoder_anomaly_detection.ipynb
├── src/
│   ├── model.py
│   └── evaluation.py
├── models/
│   └── best_autoencoder.pt
├── figures/
│   ├── training_loss_plot.png
│   ├── roc_curves_combined.png
│   └── anomaly_scores/
├── results/
│   └── auc_summary.csv
└── data/
    └── README.md
```

## Reproducibility and data

The raw event files and cached processed arrays are not redistributed in this repository. They were supplied for the course project and their public redistribution status is not assumed.

The notebook documents the preprocessing pipeline and expected file names. The trained model weights and the analysis figures from the saved run are included.

## Tools

Python · PyTorch · NumPy · pandas · scikit-learn · Matplotlib · iminuit

## Project context

This work was completed as a particle-physics data-analysis and machine-learning course project during the MSc Particle and Nuclear Physics programme at the University of Edinburgh.
