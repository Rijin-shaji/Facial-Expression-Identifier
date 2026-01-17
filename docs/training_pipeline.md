# Training Pipeline

Training is performed in two stages:

## Stage 1 – Feature Extraction
- Base model frozen
- Train classifier head

## Stage 2 – Fine Tuning
- Entire model trainable
- Lower learning rate

## Callbacks Used
- EarlyStopping
- ReduceLROnPlateau
- ModelCheckpoint
