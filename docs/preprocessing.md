# Data Preprocessing

Before training, images undergo several preprocessing steps:

## Steps
1. Resize images to 224 × 224
2. Convert images to RGB
3. Normalize pixel values using MobileNetV2 preprocessing
4. Apply data augmentation (training only)

## Augmentation Techniques
- Rotation
- Zoom
- Width/height shift
- Brightness adjustment
- Horizontal flip

These steps improve robustness and generalization.
