# Methodology

## Dataset and task

The analysis uses simulated proton-proton collision events from the Dark Machines anomaly-detection benchmark. The autoencoder is trained only on Standard Model (SM) background events and evaluated on nine Beyond Standard Model (BSM) benchmark samples.

The task is unsupervised: BSM labels are not used during training.

## Event representation

Each event is converted to a fixed 39-dimensional vector:

- five particle multiplicities: electrons, muons, jets, b-jets and photons;
- log(MET) and MET azimuth;
- log(E), log(pT), eta and phi for each of the eight highest-energy reconstructed objects.

Objects are sorted by energy. Events with more than eight objects are truncated; events with fewer than eight are zero padded.

## Train/test split and scaling

The SM sample contains 340,268 events and is split 80:20:

- 272,214 training events
- 68,054 held-out SM events

A MinMaxScaler is fitted only on the SM training sample and then applied to the held-out SM and BSM samples.

## Autoencoder

The network architecture is:

```text
39 -> 32 -> 8 -> 32 -> 39
```

The model contains 3,119 trainable parameters. Hidden layers use ReLU activations and the decoder output uses a sigmoid activation.

Training settings:

- optimiser: Adam
- learning rate: 1e-3
- batch size: 64
- reconstruction loss: mean squared error
- training epochs: 30

The best saved validation loss from the recorded run was 0.004652.

## Anomaly score

The event-level anomaly score is the mean squared reconstruction error across the 39 input features:

```text
score = mean((x_reconstructed - x_input)^2)
```

Larger scores indicate events that are less well represented by the structure learned from the SM training sample.

## Evaluation

For each BSM benchmark, SM test scores and BSM scores are combined to construct a ROC curve. The ROC AUC measures how effectively the reconstruction error separates that signal sample from SM background.
