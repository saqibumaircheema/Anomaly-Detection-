# Data

The raw collision-event datasets and cached processed arrays are not included in this public repository.

The analysis follows the **channel-2b** setup used in the Dark Machines Anomaly Score Challenge.

## Expected inputs

The workflow expects:

- one Standard Model background sample;
- nine BSM benchmark samples used for evaluation.

The original analysis used a Standard Model sample containing 340,268 events.

## Processed event representation

Each event is converted to 39 numerical features:

- 5 particle multiplicities;
- 2 missing-transverse-energy variables;
- 8 leading reconstructed objects x 4 kinematic variables.

The Standard Model sample is split 80/20 into training and test sets. Feature scaling is fitted only on the training split and is then applied to the held-out SM data and all BSM datasets.

Large raw CSV files and cached `.npy` arrays are excluded from version control.
